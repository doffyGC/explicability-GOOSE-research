"""Run a model strictly from persisted grouped splits.

Checklist B.1/B.3/B.5.  This is the canonical revision runner: it verifies the
prepared-dataset audit and split-file hash, never creates a message-level split,
and records one prediction with its ``split_group`` and fold identifier.

Use ``--max-rows-per-group-class`` for a fast technical smoke test.  A capped
run validates wiring only and must not be reported as scientific performance.

``--balance {none,downsample,smote}`` implements checklist E: each fold's
TRAIN partition is rebalanced (test is always the untouched original
distribution). ``none`` is the already-reported unbalanced baseline;
``downsample``/``smote`` require ``pip install imbalanced-learn``.

``--model`` covers checklist D.3's model-family comparison (decision-tree,
xgboost, random-forest, logistic-regression).  ``--max-train-rows-per-fold``
is the documented per-fold train cap that comparison needs on this machine;
see ``subsample_train`` and ``ablations_baselines.md`` SS7 for the policy and
the measurements behind it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


IDENTIFIER_COLUMNS = {
    "run_id", "trace_id", "event_id", "scenario_id", "seed",
    "attack_variant", "loss_rate", "burst_size", "traffic_rate",
    "substation_config", "split_group", "batch_index",
    "message_index",
    # Card-C benign-degradation provenance. impairment_mode is a trivial leak
    # if it ever reached the feature matrix - it directly encodes whether a
    # row is `benign_degradation` (see benign_controls.md SS6). The other two
    # are its run-level config, discarded for the same reason loss_rate/
    # burst_size are.
    "impairment_mode", "impairment_rate", "impairment_intensity_ms",
}
BASE_DISCARD_COLUMNS = {
    "ethDst", "ethSrc", "gocbRef", "datSet", "goID", "test", "ndsCom",
    "protocol", "ethType", "TPID", "gooseAppid",
}


class GroupedRunError(ValueError):
    pass


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_artifacts(dataset, preparation_report, splits_payload):
    dataset_abs = os.path.abspath(dataset)
    digest = sha256_file(dataset)
    if preparation_report.get("status") != "pass":
        raise GroupedRunError("preparation report did not pass")
    if os.path.abspath(preparation_report.get("output", "")) != dataset_abs:
        raise GroupedRunError("preparation report belongs to a different dataset")
    if preparation_report.get("output_sha256") != digest:
        raise GroupedRunError("prepared dataset hash differs from preparation report")
    if splits_payload.get("dataset_sha256") != digest:
        raise GroupedRunError("split file belongs to a different dataset version")
    if splits_payload.get("open_set_diagnostic"):
        raise GroupedRunError(
            "open-set LOETO splits are diagnostic and cannot be used for standard multiclass metrics"
        )
    return digest


def load_frame(path, columns=None, group_column=None, target_column=None):
    import pandas as pd
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if path.lower().endswith((".parquet", ".pq")):
        # `columns=None` reads every column, as before. The uncapped path in
        # `main()` passes only what survives `feature_matrix`'s own discard
        # sets plus group/target - the ~18 string identifier/discard columns
        # (ethDst, gocbRef, datSet, ...) are dropped there anyway, so reading
        # them from disk only to throw them away doubled peak RSS for no
        # benefit on a 20M-row dataset.
        #
        # The remaining numeric columns are stored as float64/int64 but are
        # cast to float32 by `feature_matrix` regardless, so they are cast on
        # the way in instead - row group by row group, so the full float64
        # table never exists at once. The group and target columns become
        # dictionary-encoded (pandas Categorical) rather than one Python str
        # object per row. Measured on the 20.8M-row pool: peak RSS during the
        # load drops from 11.7 GB to 7.3 GB and the resident frame from
        # ~8.7 GB to ~3.5 GB, which is the headroom a Random Forest's fitted
        # trees need (see `subsample_train`). Values are unchanged - the
        # float32 rounding is the same one `feature_matrix` already applied.
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        names = list(columns) if columns is not None else list(schema.names)
        fields = []
        for name in names:
            field = schema.field(name)
            if name in (group_column, target_column) and pa.types.is_string(field.type):
                fields.append(pa.field(name, pa.dictionary(pa.int32(), pa.string())))
            elif pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
                fields.append(pa.field(name, pa.float32()))
            else:
                fields.append(field)
        target_schema = pa.schema(fields)
        chunks = [
            parquet_file.read_row_group(index, columns=names).cast(target_schema).to_pandas()
            for index in range(parquet_file.metadata.num_row_groups)
        ]
        return pd.concat(chunks, ignore_index=True)
    raise GroupedRunError("dataset must be .csv or .parquet")


def technical_sample(frame, group_column, target_column, cap, seed):
    if not cap:
        return frame
    # Sampling happens independently inside group x class strata. It is only a
    # speed control; the report is marked technical_smoke and never used as a
    # substitute for original-distribution evaluation.
    shuffled = frame.sample(frac=1, random_state=seed)
    sampled = (
        shuffled.groupby([group_column, target_column], group_keys=False, observed=True)
        .head(cap)
        .sort_index()
    )
    return sampled


def load_technical_sample(path, group_column, target_column, cap, seed):
    """Read a capped, per-(group, class) sample without loading the dataset.

    Every `split_group` value this pipeline produces - whatever
    `--split-level` it was derived at - lives entirely inside one Parquet
    row group: `add_experiment_metadata.py`'s native path and
    `prepare_grouped_dataset.py` both write one row group per run, a run is
    one trace, and an event never spans two traces. So every (group, class)
    stratum `technical_sample` draws from is complete within a single row
    group, and sampling row group by row group - instead of loading
    everything and shuffling once - draws from the exact same eligible pool
    per stratum and respects the same `cap`, without ever materialising the
    full (possibly tens-of-millions-of-row) table just to throw most of it
    away. It does not reproduce the identical specific rows a single
    whole-dataset shuffle with the same seed would have picked (each chunk
    is shuffled independently - see the per-chunk seed offset below); that
    was never a documented guarantee of a technical smoke sample, only the
    cap and coverage are. Only called for a `.parquet` dataset; see `main()`.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    samples = []
    row_offset = 0
    for index in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_group(index).to_pandas()
        # `to_pandas()` restarts the index at 0 per row group; re-anchoring
        # it to the row's true position in the full prepared dataset keeps
        # `row_index` values recorded downstream (predictions,
        # `benign_confusion_report.py`'s positional rejoin) valid.
        chunk.index = pd.RangeIndex(row_offset, row_offset + len(chunk))
        row_offset += len(chunk)
        # Offset per chunk (same pattern as `classifier(model_name, seed +
        # fold_index)` below) so same-sized chunks don't all draw the same
        # relative shuffle from a shared seed - each group/class stratum
        # still gets exactly `cap` rows, just not the same specific ones a
        # single whole-dataset shuffle would have picked.
        sampled_chunk = technical_sample(chunk, group_column, target_column, cap, seed + index)
        if len(sampled_chunk):
            samples.append(sampled_chunk)
    if not samples:
        return pd.DataFrame(columns=[])
    return pd.concat(samples).sort_index()


def feature_matrix(frame, target_column, extra_discard):
    discard = IDENTIFIER_COLUMNS | BASE_DISCARD_COLUMNS | {target_column} | set(extra_discard)
    features = frame.drop(columns=[c for c in discard if c in frame.columns])
    non_numeric = list(features.select_dtypes(exclude="number").columns)
    if non_numeric:
        raise GroupedRunError(
            "non-numeric feature columns remain; discard or encode them explicitly: %s" % non_numeric
        )
    if features.empty:
        raise GroupedRunError("no model features remain")
    # float32 throughout: every remaining feature is a small protocol counter
    # (StNum/SqNum/frameLen/...) or a delta/electrical measurement, all well
    # inside float32's exact-integer range (2**24) and precision. Halves the
    # size of the dense array pandas/sklearn materialise per fold - on the
    # full 20M-row dataset that array alone was the actual OOM (see
    # run_grouped_validation.py history around 2026-09-12). sklearn's own
    # tree splitter already runs in float32 internally, so this changes
    # nothing about the fitted model, only how much RAM getting there needs.
    # copy=False so a frame that `load_frame` already cast on the way in is
    # not duplicated here; a float64 frame (CSV, or the technical-sample path)
    # is still converted exactly as before.
    return features.astype("float32", copy=False)


MODEL_CHOICES = ("decision-tree", "xgboost", "random-forest", "logistic-regression")


def classifier(name, seed, n_jobs=-1):
    """Build one model family at its library defaults (checklist D.3).

    D.3 is the *untuned* family comparison: apart from ``decision-tree``'s
    ``max_depth=8`` - inherited unchanged from the card-E runs this card has
    to stay comparable with - no hyperparameter is set away from its library
    default here.  Tuning is card D.4's job and happens inside train folds.

    ``n_jobs`` is a throughput knob, not a hyperparameter: it changes how many
    threads fit the model, not what is fitted.  It is recorded in the run
    report anyway, because XGBoost's histogram reduction order depends on the
    thread count and can move the last decimals of a score.  The scikit-learn
    estimators are seeded per tree from ``random_state`` and are unaffected.
    """
    if name == "decision-tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=8, random_state=seed)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "random-forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    if name == "logistic-regression":
        # Wrapped in a scaler for the same reason the baseline pipeline wraps
        # it (model/train.py): the feature matrix mixes raw electrical
        # magnitudes with protocol counters, and an unscaled lbfgs run on that
        # is dominated by whichever column happens to have the largest units.
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed)),
        ])
    raise GroupedRunError("unknown model: %s" % name)


PREDICT_CHUNK_ROWS = 1_000_000


def predict_in_chunks(model, X, positions, chunk_rows=PREDICT_CHUNK_ROWS):
    """Predict a fold's test partition in bounded blocks.

    Predictions are row-independent, so this returns exactly what
    ``model.predict(X[positions])`` returns - it only bounds the transient
    memory getting there.  Predicting a whole partition at once allocates
    several temporaries proportional to its size, and the largest fold here
    is 5.58M rows: the fancy-indexed slice, then a copy per pipeline step
    (``StandardScaler.transform`` copies), then - for the linear model -
    a float64 promotion of the whole block, because ``coef_`` is float64
    while the feature matrix is float32.  Together that overran RAM on the
    ``logistic-regression`` run even though its fit used only ~96k rows.
    A fixed block size makes the peak independent of fold size and model
    family.
    """
    import numpy as np

    chunks = [
        model.predict(X[positions[start:start + chunk_rows]])
        for start in range(0, len(positions), chunk_rows)
    ]
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def predict_proba_in_chunks(model, X, positions, chunk_rows=PREDICT_CHUNK_ROWS):
    """The posterior counterpart of ``predict_in_chunks``, at float32.

    Same bounded-block reasoning, one extra constraint: the result is kept
    for the whole fold rather than consumed chunk by chunk, so it is cast
    down to float32 as it arrives.  A 5.58M-row fold is 134 MB at float32
    against 268 MB at the float64 scikit-learn returns, and that array has
    to coexist with the fitted model until the fold's scores are written.
    """
    import numpy as np

    chunks = [
        np.asarray(model.predict_proba(X[positions[start:start + chunk_rows]]),
                   dtype="float32")
        for start in range(0, len(positions), chunk_rows)
    ]
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def labels_from_proba(model, X, positions, proba, chunk_rows=PREDICT_CHUNK_ROWS):
    """Derive the fold's hard labels from the posteriors, and prove it is the same thing.

    Predicting twice - once for the labels, once for the posteriors - would
    double the predict phase of every scored run, so the labels are taken as
    ``argmax(proba)`` instead.  For every family here that *is* what
    ``predict`` computes, but "essentially always identical" is not a claim
    this pipeline gets to make without checking: ties and a booster that
    resolves the argmax inside its own C++ rather than in numpy could both
    break it, and a silent disagreement would mean the persisted scores
    describe a different classifier than `grouped_predictions.csv` does.

    So the first block is verified against ``model.predict``.  On any
    mismatch the run falls back to ``predict`` for the whole fold - the
    labels stay exactly what an unscored run would have written - and the
    disagreement is recorded in the report rather than hidden.
    """
    import numpy as np

    predicted = np.argmax(proba, axis=1).astype("int64")
    checked = int(min(len(positions), chunk_rows))
    if checked == 0:
        return predicted, {"rows_checked": 0, "argmax_mismatches": 0,
                           "fell_back_to_predict": False}
    reference = np.asarray(model.predict(X[positions[:checked]]), dtype="int64")
    mismatches = int(np.count_nonzero(reference != predicted[:checked]))
    if mismatches:
        predicted = np.asarray(predict_in_chunks(model, X, positions), dtype="int64")
    return predicted, {
        "rows_checked": checked,
        "argmax_mismatches": mismatches,
        "fell_back_to_predict": bool(mismatches),
    }


SCORES_FILENAME = "grouped_scores.parquet"


class ScoreWriter:
    """Persists the per-row class posteriors that any threshold question needs.

    `grouped_predictions.csv` records only the argmax.  That fixes the
    operating point at "whatever prior the training partition happened to
    have", which is exactly why this revision's two published configurations
    - `none` (attack recall ~0) and `downsample` (attack recall ~0.70 at a
    ~39% false-positive rate on ideal traffic) - cannot be compared as
    operating points: they are the same score read at two thresholds
    separated by the balancing, with the rest of the curve never measured.
    Recovering it from hard labels is impossible; recovering it from
    posteriors is arithmetic (`grouped_pr_curves.py`).

    Written as a separate Parquet file rather than as extra CSV columns for
    two reasons: `grouped_predictions.csv` stays byte-identical to what
    earlier runs produced, which `validation_protocol.md` relies on as a
    regression check, and 23.2M x 6 float32 costs ~0.4 GB compressed instead
    of ~1.5 GB of text.  One row group per fold, so a reader's peak is one
    fold rather than the whole run.
    """

    def __init__(self, path, classes, group_labels):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self.classes = [str(name) for name in classes]
        self._groups = pa.array([str(label) for label in group_labels], type=pa.string())
        self._labels = pa.array(self.classes, type=pa.string())
        fields = [
            pa.field("split_id", pa.dictionary(pa.int32(), pa.string())),
            pa.field("row_index", pa.int64()),
            pa.field("split_group", pa.dictionary(pa.int32(), pa.string())),
            pa.field("y_true", pa.dictionary(pa.int32(), pa.string())),
        ]
        fields += [pa.field("p_%s" % name, pa.float32()) for name in self.classes]
        self.schema = pa.schema(fields)
        self._writer = pq.ParquetWriter(path, self.schema, compression="snappy")

    def write_fold(self, split_id, row_index, group_codes, y_codes, proba):
        import numpy as np

        pa = self._pa
        rows = len(row_index)
        if proba.shape != (rows, len(self.classes)):
            raise GroupedRunError(
                "%s produced a %r posterior block for %d rows and %d classes"
                % (split_id, proba.shape, rows, len(self.classes)))
        columns = [
            pa.DictionaryArray.from_arrays(
                pa.array(np.zeros(rows, dtype="int32")),
                pa.array([str(split_id)], type=pa.string())),
            pa.array(np.asarray(row_index, dtype="int64")),
            pa.DictionaryArray.from_arrays(
                pa.array(np.asarray(group_codes, dtype="int32")), self._groups),
            pa.DictionaryArray.from_arrays(
                pa.array(np.asarray(y_codes, dtype="int32")), self._labels),
        ]
        columns += [
            pa.array(np.ascontiguousarray(proba[:, index]), type=pa.float32())
            for index in range(proba.shape[1])
        ]
        self._writer.write_table(pa.Table.from_arrays(columns, schema=self.schema))

    def close(self):
        self._writer.close()


def fit_diagnostics(model):
    """Whatever the fitted model can say about whether its fit actually finished.

    Only iterative solvers answer: `logistic-regression`'s lbfgs reports
    `n_iter_`, and a value equal to `max_iter` means it stopped on the
    iteration budget rather than on convergence. Recording it keeps a
    non-converged run from being reported as a converged baseline - the
    honest fix for a D.3 family comparison is to report the caveat, since
    raising `max_iter` would be tuning and tuning is card D.4.
    """
    estimator = model
    steps = getattr(model, "named_steps", None)
    if steps is not None:
        estimator = steps.get("clf", model)
    n_iter = getattr(estimator, "n_iter_", None)
    if n_iter is None:
        return None
    try:
        iterations = [int(value) for value in n_iter]
    except TypeError:
        iterations = [int(n_iter)]
    max_iter = getattr(estimator, "max_iter", None)
    return {
        "n_iter": iterations,
        "max_iter": int(max_iter) if max_iter is not None else None,
        "converged": None if max_iter is None else bool(max(iterations) < max_iter),
    }


def class_counts(y, class_names):
    import numpy as np
    counts = np.bincount(y, minlength=len(class_names))
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


def average_block(report):
    """Both averaging schemes, each named for what it actually is.

    Checklist E.4: a single unlabelled "F1" is exactly what makes an
    imbalanced-class result unreadable. `macro` is the unweighted mean over
    classes (every class counts the same, so the four rare attack classes
    dominate the average as much as `normal` does); `weighted` averages the
    same per-class numbers weighted by support (so it tracks `normal` and
    reads far higher). `accuracy`, reported separately per fold, is the
    overall/micro figure. Per-class values stay in `per_class`, never folded
    into either average.
    """
    return {
        scheme: {
            key: float(report["%s avg" % scheme][key])
            for key in ("precision", "recall", "f1-score")
        }
        for scheme in ("macro", "weighted")
    }


def subsample_train(strata, cap, seed):
    """Positions of a proportional, group- and class-stratified train subsample.

    Checklist D.3's documented per-fold train cap.  This is a *scale* control,
    not a balancing one: every ``(split_group, class)`` stratum keeps the same
    share of the fold's train partition it had before, so the subsample is a
    shrunken replica of the original train distribution rather than a
    reweighting of it.  ``resample_train`` (checklist E) is the only thing in
    this script that deliberately changes class proportions, and it runs
    *after* this.  The test partition is never subsampled - every metric in
    the run is still measured on the untouched original distribution.

    Why it exists: a ``RandomForestClassifier`` at library defaults stores
    4.0-4.4 tree nodes per training row across its 100 trees (measured on this
    dataset at 0.5M and 2.28M rows) at ~112 bytes a node, i.e. ~490 bytes of
    fitted forest per training row.  A full ~16M-row train partition therefore
    needs ~6.7 GB of forest on top of the resident feature matrix, which does
    not fit the 15.6 GB machine the revision runs on.  ``ablations_baselines.md``
    SS7 records the measurements and which runs the cap was applied to.

    Strata that would otherwise round down to zero rows keep one row, so a cap
    can never silently delete a rare class from training; the returned sample
    can exceed ``cap`` by at most the number of such strata.
    """
    import numpy as np

    total = len(strata)
    if not cap or total <= cap:
        return None
    fraction = cap / float(total)
    rng = np.random.RandomState(seed)
    # lexsort's last key is primary: rows are grouped by stratum, and the
    # random key shuffles within each stratum so the head of every stratum is
    # an unbiased draw from it.
    order = np.lexsort((rng.random_sample(total), strata))
    _, starts, counts = np.unique(strata[order], return_index=True, return_counts=True)
    keep = np.concatenate([
        order[start:start + max(1, int(count * fraction))]
        for start, count in zip(starts, counts)
    ])
    keep.sort()
    return keep


def resample_train(X_train, y_train, class_names, strategy, seed,
                    smote_factor, smote_max_target):
    """Rebalance one fold's TRAIN partition only (checklist E).

    Never touches the test partition - callers must apply this after
    splitting and before ``fit()``. ``strategy="none"`` is a no-op (the
    already-reported unbalanced baseline). Both real strategies are
    deliberately bounded rather than aiming for exact parity with the
    majority class: on this dataset the majority (`normal`) fold-train count
    is ~16M rows against ~14-50k for the rarest attack classes, so plain
    SMOTE-to-parity would synthesise on the order of 60M+ rows - infeasible
    on a 16GB-RAM machine (the same class of failure fixed in this script's
    2026-09-12 OOM). See validation_protocol.md, "Balancing scenarios".
    """
    if strategy == "none":
        return X_train, y_train
    try:
        import imblearn  # noqa: F401
    except ImportError as exc:
        raise GroupedRunError(
            "balance=%r requires the 'imbalanced-learn' package (pip install imbalanced-learn)"
            % strategy
        ) from exc

    if strategy == "downsample":
        # Undersample every class down to the size of the smallest class
        # present in this fold's train partition. Fast and bounded by
        # construction: the resampled train set never exceeds
        # num_classes * min_class_count rows.
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(sampling_strategy="not minority", random_state=seed)
        return sampler.fit_resample(X_train, y_train)

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        counts = class_counts(y_train, class_names)
        targets = {}
        for label, name in enumerate(class_names):
            count = counts[name]
            if count >= smote_max_target:
                continue  # already at/above the cap - leave untouched, never downsampled here
            target = min(int(count * smote_factor), smote_max_target)
            if target <= count:
                continue
            if count <= 5:
                raise GroupedRunError(
                    "class %r has only %d train rows in this fold - too few for SMOTE's "
                    "default 5 neighbours" % (name, count)
                )
            targets[label] = target
        if not targets:
            return X_train, y_train
        sampler = SMOTE(sampling_strategy=targets, random_state=seed)
        return sampler.fit_resample(X_train, y_train)

    raise GroupedRunError("unknown balance strategy: %s" % strategy)


def prepare_arrays(frame, group_column, target_column, extra_discard):
    """Turn the loaded frame into the compact arrays the fold loop needs.

    Split out from ``run_folds`` so ``main`` can drop its reference to the
    DataFrame before the first fit.  On the 20.8M-row pool the frame and the
    feature matrix are ~5.7 GB and ~3.3 GB, and holding both for the whole run
    left no headroom for a Random Forest's fitted trees (see
    ``subsample_train``).  Keeping only the arrays halves the resident
    footprint for every model family.

    ``row_index`` preserves the frame's own index, so the ``row_index`` column
    written to ``grouped_predictions.csv`` keeps identifying the same dataset
    row it always did - including under the technical-sample path, where
    ``load_technical_sample`` re-anchors the index to true dataset positions.
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    # int16/int32 rather than the default intp: these are held for the whole
    # run next to a multi-GB feature matrix, and neither the class count (6)
    # nor the group count (205) comes close to needing 64 bits.
    y = encoder.fit_transform(frame[target_column].astype(str)).astype("int16")
    group_labels, group_codes = np.unique(frame[group_column].astype(str).to_numpy(),
                                          return_inverse=True)
    group_codes = group_codes.astype("int32")
    row_index = frame.index.to_numpy()
    features = feature_matrix(frame, target_column, extra_discard)
    feature_names = list(features.columns)
    # Every remaining column is float32 by now, so pandas holds them in one
    # block and this is a view rather than another full-size copy.
    X = features.to_numpy(dtype="float32", copy=False)
    return {
        "X": X, "y": y, "group_codes": group_codes, "group_labels": group_labels,
        "row_index": row_index, "classes": list(encoder.classes_),
        "features": feature_names,
    }


def feature_column_names(names, target_column, extra_discard):
    """The feature columns, decided from names alone.

    Split out so `load_grouped_arrays` can reach the same decision
    `feature_matrix` reaches, without needing a DataFrame to reach it. The
    discard sets are the single source of truth for both; the regression test
    requires the two paths to agree column for column.
    """
    discard = IDENTIFIER_COLUMNS | BASE_DISCARD_COLUMNS | {target_column} | set(extra_discard)
    return [name for name in names if name not in discard]


def load_grouped_arrays(path, group_column, target_column, extra_discard):
    """Read a Parquet dataset straight into the arrays the fold loop needs.

    The path this replaces went dataset -> per-row-group DataFrames ->
    `pd.concat` -> `feature_matrix`'s `drop` -> numpy, and **three** of those
    four steps hold a full copy of the data at once:

      - `pd.concat` holds the chunk list and the concatenated frame together,
      - `frame.drop(columns=...)` copies every surviving float32 column into a
        new frame while the original is still referenced by the caller,
      - only then is the numpy view taken.

    Measured on the 265-run/23,226,530-row pool: **10.02 GB peak**, against
    3.46 GB of actual feature matrix. That was fine at 205 runs (the 7.3 GB
    in `ablations_baselines.md` SS7) and stopped being fine when the pool grew
    11.7%; every full run on a 15.6 GB machine now dies during the load,
    whatever model or flags follow it.

    Filling a preallocated `float32` array row group by row group removes all
    three copies: the peak becomes the array itself plus one row group,
    ~3.5 GB, and it no longer scales with anything except the feature matrix
    it has to produce anyway.

    Values are identical to the old path, not merely equivalent - same
    columns in the same order, the same float32 cast `load_frame` already
    applied on the way in, and the same sorted label ordering
    `np.unique`/`LabelEncoder` produced. `test_validation_protocol.py` asserts
    that against the DataFrame path rather than trusting this docstring.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    schema = handle.schema_arrow
    if group_column not in schema.names or target_column not in schema.names:
        raise GroupedRunError("dataset is missing group or target column")

    names = feature_column_names(schema.names, target_column, extra_discard)
    non_numeric = [name for name in names
                   if not (pa.types.is_floating(schema.field(name).type)
                           or pa.types.is_integer(schema.field(name).type))]
    if non_numeric:
        raise GroupedRunError(
            "non-numeric feature columns remain; discard or encode them explicitly: %s"
            % non_numeric)
    if not names:
        raise GroupedRunError("no model features remain")

    rows = handle.metadata.num_rows
    if rows == 0:
        raise GroupedRunError("dataset holds no rows")
    features = np.empty((rows, len(names)), dtype="float32")
    # Local codes first, global ones after: a row group's dictionary covers
    # only the labels that row group happens to hold, so the global ordering
    # is not knowable until every row group has been seen. Remapping at the
    # end costs one int32 lookup per row and avoids either a second pass over
    # the file or 23M Python strings.
    group_codes = np.empty(rows, dtype="int32")
    target_codes = np.empty(rows, dtype="int32")
    group_dictionaries, target_dictionaries, spans = [], [], []
    encoded_type = pa.dictionary(pa.int32(), pa.string())

    offset = 0
    for number in range(handle.num_row_groups):
        table = handle.read_row_group(number, columns=names + [group_column, target_column])
        count = table.num_rows
        for position, name in enumerate(names):
            features[offset:offset + count, position] = (
                table.column(name).to_numpy(zero_copy_only=False))
        for column, dictionaries, destination in (
            (group_column, group_dictionaries, group_codes),
            (target_column, target_dictionaries, target_codes),
        ):
            series = table.column(column).cast(encoded_type).to_pandas()
            local = series.cat.codes.to_numpy()
            if (local < 0).any():
                raise GroupedRunError(
                    "column %s holds nulls, which have no class or group" % column)
            dictionaries.append([str(value) for value in series.cat.categories])
            destination[offset:offset + count] = local
        spans.append((offset, count))
        offset += count
        del table

    def globalise(dictionaries, codes):
        # sorted(), not np.unique(): the DataFrame path compares Python str
        # objects (pandas `astype(str)` yields an object column), and this has
        # to reproduce that ordering exactly, not merely a defensible one.
        labels = sorted({label for dictionary in dictionaries for label in dictionary})
        index = {label: position for position, label in enumerate(labels)}
        for (start, count), dictionary in zip(spans, dictionaries):
            lookup = np.asarray([index[label] for label in dictionary], dtype="int32")
            codes[start:start + count] = lookup[codes[start:start + count]]
        return labels

    group_labels = globalise(group_dictionaries, group_codes)
    classes = globalise(target_dictionaries, target_codes)
    return {
        "X": features,
        "y": target_codes.astype("int16"),
        "group_codes": group_codes,
        "group_labels": group_labels,
        # The DataFrame path's index came from `pd.concat(ignore_index=True)`,
        # so it was always 0..N-1; `grouped_predictions.csv` keeps meaning the
        # same dataset row it always did.
        "row_index": np.arange(rows, dtype="int64"),
        "classes": classes,
        "features": list(names),
    }


def run_folds(arrays, splits, model_name, seed, predictions_writer,
              balance_strategy="none", smote_factor=20.0, smote_max_target=200_000,
              max_train_rows=0, n_jobs=-1, scores_writer=None):
    import gc

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report

    X = arrays["X"]
    y = arrays["y"]
    group_codes = arrays["group_codes"]
    group_labels = arrays["group_labels"]
    row_index = arrays["row_index"]
    classes = np.asarray(arrays["classes"], dtype=object)
    all_labels = list(range(len(classes)))
    label_to_code = {label: code for code, label in enumerate(group_labels)}
    metrics = []

    for fold_index, split in enumerate(splits):
        def mask_for(partition):
            codes = [label_to_code[g] for g in split[partition] if g in label_to_code]
            return np.isin(group_codes, np.asarray(codes, dtype=group_codes.dtype))

        train_mask = mask_for("train_groups")
        test_mask = mask_for("test_groups")
        if (train_mask & test_mask).any():
            raise GroupedRunError("%s has row-level train/test overlap" % split["split_id"])
        if not train_mask.any() or not test_mask.any():
            raise GroupedRunError("%s has an empty partition" % split["split_id"])

        train_classes = set(y[train_mask].tolist())
        test_classes = set(y[test_mask].tolist())
        if not test_classes.issubset(train_classes):
            missing = [classes[code] for code in sorted(test_classes - train_classes)]
            raise GroupedRunError("%s test-only classes: %s" % (split["split_id"], missing))

        train_positions = np.flatnonzero(train_mask)
        rows_available = int(len(train_positions))
        # Stratify the cap by (group, class) jointly: group codes are dense
        # from np.unique, so this pairing is injective.
        strata = (group_codes[train_positions].astype("int32") * len(classes)
                  + y[train_positions])
        keep = subsample_train(strata, max_train_rows, seed + fold_index)
        if keep is not None:
            train_positions = train_positions[keep]
        rows_sampled = int(len(train_positions))
        del strata, keep, train_mask
        gc.collect()

        X_train, y_train = X[train_positions], y[train_positions]
        del train_positions
        counts_before = class_counts(y_train, classes)
        X_train, y_train = resample_train(
            X_train, y_train, classes, balance_strategy,
            seed + fold_index, smote_factor, smote_max_target,
        )
        counts_after = class_counts(y_train, classes)

        model = classifier(model_name, seed + fold_index, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        rows_trained = int(len(y_train))
        diagnostics = fit_diagnostics(model)
        # Released before predicting: on an uncapped fold this array is up to
        # ~2.6 GB and nothing downstream needs it, while the fitted model (a
        # forest especially) still has to share RAM with the test slice.
        del X_train, y_train
        gc.collect()

        test_positions = np.flatnonzero(test_mask)
        del test_mask
        if scores_writer is None:
            predicted = predict_in_chunks(model, X, test_positions)
            proba, score_diagnostics = None, None
        else:
            proba = predict_proba_in_chunks(model, X, test_positions)
            predicted, score_diagnostics = labels_from_proba(
                model, X, test_positions, proba)
        del model
        gc.collect()
        y_test = y[test_positions]
        report = classification_report(
            y_test, predicted, labels=all_labels,
            target_names=classes, output_dict=True, zero_division=0,
        )
        metrics.append({
            "split_id": split["split_id"],
            "train_rows": rows_available,
            "test_rows": int(len(test_positions)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "averages": average_block(report),
            "per_class": {
                label: {
                    key: float(report[label][key])
                    for key in ("precision", "recall", "f1-score", "support")
                }
                for label in classes
            },
            "subsample": {
                "max_train_rows_per_fold": max_train_rows or None,
                "train_rows_available": rows_available,
                "train_rows_sampled": rows_sampled,
                "applied": rows_sampled < rows_available,
            },
            "fit": diagnostics,
            "scores": score_diagnostics,
            "balance": {
                "strategy": balance_strategy,
                "train_rows_resampled": rows_trained,
                "train_class_counts_before": counts_before,
                "train_class_counts_after": counts_after,
            },
        })
        true_labels = classes[y_test]
        predicted_labels = classes[np.asarray(predicted, dtype=int)]
        # Written straight to disk instead of accumulated in a list: on a
        # full run every dataset row is a test row in exactly one fold, so
        # the list would otherwise hold one dict per row for the whole run.
        for position, truth, prediction in zip(test_positions, true_labels, predicted_labels):
            predictions_writer.writerow({
                "split_id": split["split_id"], "row_index": int(row_index[position]),
                "split_group": group_labels[group_codes[position]],
                "y_true": truth, "y_pred": prediction,
            })
        if scores_writer is not None:
            scores_writer.write_fold(
                split["split_id"], row_index[test_positions],
                group_codes[test_positions], y_test, proba,
            )
        del test_positions, y_test, predicted, true_labels, predicted_labels, proba
        gc.collect()
    return metrics, list(classes), list(arrays["features"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default="decision-tree",
                        help="Checklist D.3: model family, at library defaults. "
                             "Tuning is card D.4 and happens inside train folds.")
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-group-class", type=int, default=0,
                        help="Non-zero enables a technical smoke sample.")
    parser.add_argument("--discard-column", action="append", default=[])
    parser.add_argument("--balance", choices=["none", "downsample", "smote"], default="none",
                        help="Checklist E: rebalance each fold's TRAIN partition only; "
                             "test is always evaluated on the original distribution.")
    parser.add_argument("--smote-oversample-factor", type=float, default=20.0,
                        help="balance=smote only: oversample a minority class up to this "
                             "many times its original per-fold count (see --smote-max-target).")
    parser.add_argument("--smote-max-target", type=int, default=200_000,
                        help="balance=smote only: absolute cap on a class's oversampled "
                             "count, regardless of --smote-oversample-factor.")
    parser.add_argument("--max-train-rows-per-fold", type=int, default=0,
                        help="Checklist D.3: cap each fold's TRAIN partition at this many "
                             "rows, sampled proportionally within every (split_group, class) "
                             "stratum. Unlike --max-rows-per-group-class this is NOT a smoke "
                             "flag: the test partition and the evaluated distribution are "
                             "untouched, so the run stays a full_grouped_run. Applied before "
                             "--balance. See ablations_baselines.md SS7.")
    parser.add_argument("--save-scores", action="store_true",
                        help="Also write grouped_scores.parquet: the per-row class "
                             "posteriors behind y_pred. Required by grouped_pr_curves.py "
                             "- PR/DET curves, a fixed alert budget and undoing a "
                             "training-prior shift are all unanswerable from hard labels "
                             "alone. Costs ~0.4 GB per full run and does not change "
                             "grouped_predictions.csv.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Threads for the model that supports it (-1 = all cores). "
                             "A throughput knob, not a hyperparameter; recorded in the report.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Both staging paths are named before anything that can fail, so the
    # cleanup handler never has to ask whether they exist as names.
    predictions_tmp = scores_tmp = None
    try:
        preparation = load_json(args.preparation_report)
        split_payload = load_json(args.splits)
        digest = verify_artifacts(args.dataset, preparation, split_payload)

        if args.max_rows_per_group_class and args.dataset.lower().endswith((".parquet", ".pq")):
            # A capped (technical smoke) run only ever needs a small sample,
            # so it is read chunked - see load_technical_sample - and the
            # full dataset is never loaded. Column existence is checked from
            # the file's schema alone, before any row is read.
            import pyarrow.parquet as pq
            columns = pq.ParquetFile(args.dataset).schema_arrow.names
            if args.group_column not in columns or args.target_column not in columns:
                raise GroupedRunError("dataset is missing group or target column")
            arrays = None
            frame = load_technical_sample(
                args.dataset, args.group_column, args.target_column,
                args.max_rows_per_group_class, args.seed,
            )
        elif args.dataset.lower().endswith((".parquet", ".pq")):
            # The full-training path never builds a DataFrame at all: it fills
            # the feature array straight from the row groups. See
            # `load_grouped_arrays` for the measurement that forced this - the
            # DataFrame route peaked at 10.02 GB on the 265-run pool, which no
            # longer fits on a 15.6 GB machine whatever model follows it.
            frame = None
            arrays = load_grouped_arrays(
                args.dataset, args.group_column, args.target_column,
                args.discard_column,
            )
        else:
            arrays = None
            frame = load_frame(args.dataset, columns=None,
                               group_column=args.group_column,
                               target_column=args.target_column)
            if args.group_column not in frame or args.target_column not in frame:
                raise GroupedRunError("dataset is missing group or target column")
            frame = technical_sample(
                frame, args.group_column, args.target_column,
                args.max_rows_per_group_class, args.seed,
            )
        os.makedirs(args.out_dir, exist_ok=True)
        if arrays is None:
            rows_used = len(frame)
            # The frame is released here, before the first fit: `prepare_arrays`
            # has already copied everything the fold loop reads into compact
            # arrays, and on a full run keeping both costs ~3.5 GB that a Random
            # Forest needs for its trees.
            arrays = prepare_arrays(frame, args.group_column, args.target_column,
                                    args.discard_column)
            del frame
        else:
            rows_used = len(arrays["row_index"])
        predictions_path = os.path.join(args.out_dir, "grouped_predictions.csv")
        # Predictions are written straight to disk as each fold finishes
        # instead of being collected into one Python list first (on a full
        # run that list would hold one dict per dataset row). Staged under
        # `.tmp` and renamed into place only once the whole run - including
        # the JSON report - has succeeded, so a failure partway through a
        # fold still leaves no partial `grouped_predictions.csv` behind,
        # matching the previous all-or-nothing behaviour.
        predictions_tmp = predictions_path + ".tmp"
        scores_path = os.path.join(args.out_dir, SCORES_FILENAME)
        scores_tmp = scores_path + ".tmp" if args.save_scores else None
        scores_writer = None
        try:
            if args.save_scores:
                scores_writer = ScoreWriter(
                    scores_tmp, arrays["classes"], arrays["group_labels"])
            with open(predictions_tmp, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["split_id", "row_index", "split_group", "y_true", "y_pred"]
                )
                writer.writeheader()
                metrics, classes, features = run_folds(
                    arrays, split_payload["splits"], args.model, args.seed, writer,
                    balance_strategy=args.balance,
                    smote_factor=args.smote_oversample_factor,
                    smote_max_target=args.smote_max_target,
                    max_train_rows=args.max_train_rows_per_fold,
                    n_jobs=args.n_jobs, scores_writer=scores_writer,
                )
        finally:
            if scores_writer is not None:
                scores_writer.close()
        report = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "status": "technical_smoke" if args.max_rows_per_group_class else "full_grouped_run",
            "dataset": os.path.abspath(args.dataset),
            "dataset_sha256": digest,
            "splits": os.path.abspath(args.splits),
            "protocol": split_payload["protocol"],
            "model": args.model,
            "seed": args.seed,
            "group_column": args.group_column,
            "target_column": args.target_column,
            "balance": args.balance,
            "smote_oversample_factor": args.smote_oversample_factor if args.balance == "smote" else None,
            "smote_max_target": args.smote_max_target if args.balance == "smote" else None,
            "sample_cap_per_group_class": args.max_rows_per_group_class or None,
            "max_train_rows_per_fold": args.max_train_rows_per_fold or None,
            "n_jobs": args.n_jobs,
            "scores_file": SCORES_FILENAME if args.save_scores else None,
            "rows_used": rows_used,
            "classes": classes,
            "features": features,
            "fold_metrics": metrics,
        }
        with open(os.path.join(args.out_dir, "grouped_validation_report.json"),
                  "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        os.replace(predictions_tmp, predictions_path)
        if scores_tmp is not None:
            os.replace(scores_tmp, scores_path)
    except (OSError, json.JSONDecodeError, GroupedRunError, ValueError) as exc:
        for stale in (predictions_tmp, scores_tmp):
            try:
                if stale and os.path.exists(stale):
                    os.remove(stale)
            except OSError:
                pass
        print("GROUPED VALIDATION FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print("Completed %d grouped folds on %d rows (%s)." %
          (len(metrics), rows_used, report["status"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
