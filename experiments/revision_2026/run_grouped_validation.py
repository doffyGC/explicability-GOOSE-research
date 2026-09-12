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


def load_frame(path, columns=None):
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
        return pd.read_parquet(path, columns=columns)
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
    return features.astype("float32")


def classifier(name, seed):
    if name == "decision-tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=8, random_state=seed)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", random_state=seed
        )
    raise GroupedRunError("unknown model: %s" % name)


def class_counts(y, class_names):
    import numpy as np
    counts = np.bincount(y, minlength=len(class_names))
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


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


def run_folds(frame, splits, group_column, target_column, model_name, seed,
              extra_discard, predictions_writer, balance_strategy="none",
              smote_factor=20.0, smote_max_target=200_000):
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[target_column].astype(str))
    X = feature_matrix(frame, target_column, extra_discard)
    groups = frame[group_column].astype(str)
    all_labels = list(range(len(encoder.classes_)))
    metrics = []

    for fold_index, split in enumerate(splits):
        train_mask = groups.isin(split["train_groups"]).to_numpy()
        test_mask = groups.isin(split["test_groups"]).to_numpy()
        if (train_mask & test_mask).any():
            raise GroupedRunError("%s has row-level train/test overlap" % split["split_id"])
        if not train_mask.any() or not test_mask.any():
            raise GroupedRunError("%s has an empty partition" % split["split_id"])

        train_classes = set(y[train_mask])
        test_classes = set(y[test_mask])
        if not test_classes.issubset(train_classes):
            missing = encoder.inverse_transform(sorted(test_classes - train_classes)).tolist()
            raise GroupedRunError("%s test-only classes: %s" % (split["split_id"], missing))

        X_train, y_train = X.loc[train_mask], y[train_mask]
        counts_before = class_counts(y_train, encoder.classes_)
        X_train, y_train = resample_train(
            X_train, y_train, encoder.classes_, balance_strategy,
            seed + fold_index, smote_factor, smote_max_target,
        )
        counts_after = class_counts(y_train, encoder.classes_)

        model = classifier(model_name, seed + fold_index)
        model.fit(X_train, y_train)
        predicted = model.predict(X.loc[test_mask])
        report = classification_report(
            y[test_mask], predicted, labels=all_labels,
            target_names=encoder.classes_, output_dict=True, zero_division=0,
        )
        metrics.append({
            "split_id": split["split_id"],
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "accuracy": float(accuracy_score(y[test_mask], predicted)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "per_class": {
                label: {
                    key: float(report[label][key])
                    for key in ("precision", "recall", "f1-score", "support")
                }
                for label in encoder.classes_
            },
            "balance": {
                "strategy": balance_strategy,
                "train_rows_resampled": int(len(y_train)),
                "train_class_counts_before": counts_before,
                "train_class_counts_after": counts_after,
            },
        })
        test_indices = frame.index[test_mask]
        true_labels = encoder.inverse_transform(y[test_mask])
        predicted_labels = encoder.inverse_transform(np.asarray(predicted, dtype=int))
        # Written straight to disk instead of accumulated in a list: on a
        # full run every dataset row is a test row in exactly one fold, so
        # the list would otherwise hold one dict per row for the whole run.
        for row_index, group, truth, prediction in zip(
                test_indices, groups.loc[test_mask], true_labels, predicted_labels):
            predictions_writer.writerow({
                "split_id": split["split_id"], "row_index": int(row_index),
                "split_group": group, "y_true": truth, "y_pred": prediction,
            })
    return metrics, list(encoder.classes_), list(X.columns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", choices=["decision-tree", "xgboost"], default="decision-tree")
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
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
            frame = load_technical_sample(
                args.dataset, args.group_column, args.target_column,
                args.max_rows_per_group_class, args.seed,
            )
        else:
            columns = None
            if args.dataset.lower().endswith((".parquet", ".pq")):
                import pyarrow.parquet as pq
                schema_columns = pq.ParquetFile(args.dataset).schema_arrow.names
                drop_cols = (
                    IDENTIFIER_COLUMNS | BASE_DISCARD_COLUMNS | set(args.discard_column)
                ) - {args.group_column, args.target_column}
                columns = [c for c in schema_columns if c not in drop_cols]
            frame = load_frame(args.dataset, columns=columns)
            if args.group_column not in frame or args.target_column not in frame:
                raise GroupedRunError("dataset is missing group or target column")
            frame = technical_sample(
                frame, args.group_column, args.target_column,
                args.max_rows_per_group_class, args.seed,
            )
        os.makedirs(args.out_dir, exist_ok=True)
        predictions_path = os.path.join(args.out_dir, "grouped_predictions.csv")
        # Predictions are written straight to disk as each fold finishes
        # instead of being collected into one Python list first (on a full
        # run that list would hold one dict per dataset row). Staged under
        # `.tmp` and renamed into place only once the whole run - including
        # the JSON report - has succeeded, so a failure partway through a
        # fold still leaves no partial `grouped_predictions.csv` behind,
        # matching the previous all-or-nothing behaviour.
        predictions_tmp = predictions_path + ".tmp"
        with open(predictions_tmp, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["split_id", "row_index", "split_group", "y_true", "y_pred"]
            )
            writer.writeheader()
            metrics, classes, features = run_folds(
                frame, split_payload["splits"], args.group_column, args.target_column,
                args.model, args.seed, args.discard_column, writer,
                balance_strategy=args.balance,
                smote_factor=args.smote_oversample_factor,
                smote_max_target=args.smote_max_target,
            )
        report = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "status": "technical_smoke" if args.max_rows_per_group_class else "full_grouped_run",
            "dataset": os.path.abspath(args.dataset),
            "dataset_sha256": digest,
            "splits": os.path.abspath(args.splits),
            "protocol": split_payload["protocol"],
            "model": args.model,
            "seed": args.seed,
            "balance": args.balance,
            "smote_oversample_factor": args.smote_oversample_factor if args.balance == "smote" else None,
            "smote_max_target": args.smote_max_target if args.balance == "smote" else None,
            "sample_cap_per_group_class": args.max_rows_per_group_class or None,
            "rows_used": len(frame),
            "classes": classes,
            "features": features,
            "fold_metrics": metrics,
        }
        with open(os.path.join(args.out_dir, "grouped_validation_report.json"),
                  "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        os.replace(predictions_tmp, predictions_path)
    except (OSError, json.JSONDecodeError, GroupedRunError, ValueError) as exc:
        try:
            if os.path.exists(predictions_tmp):
                os.remove(predictions_tmp)
        except (OSError, NameError):
            pass
        print("GROUPED VALIDATION FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print("Completed %d grouped folds on %d rows (%s)." %
          (len(metrics), len(frame), report["status"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
