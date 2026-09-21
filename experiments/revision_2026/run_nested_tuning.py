"""Checklist D.4: tune the champion *inside* each train fold, and report what it bought.

D.3 picked a family at library defaults; D.1 named the features it runs on;
D.2 measured the rule it has to beat.  What none of them answer is whether the
champion is *fitted* - whether XGBoost at `max_depth=6`, 100 rounds and
`learning_rate=0.3` is near the best this data supports, or whether the paper
is reporting an arbitrary point in hyperparameter space as if it were a
capability claim.

The protocol is nested, and the nesting is the whole point: hyperparameters
are chosen on inner grouped folds drawn from the **outer fold's train groups
only**, the chosen configuration is refit on that outer train partition, and
it is scored once on the outer test groups it has never been compared
against.  Selecting on the outer test fold - the ordinary "tune with
cross-validation, then report the cross-validated score" mistake - would
re-introduce, at the hyperparameter level, exactly the leak this whole
revision exists to remove.

The runner does not own a fold loop.  It injects a selector into
`run_grouped_validation.run_folds` (its `model_selector` hook), so a tuned run
inherits that loop's invariants rather than a second copy of them: the group
overlap check, the empty-partition and test-only-class refusals, train-only
balancing, the untouched test distribution, and the same
`grouped_predictions.csv` / `grouped_scores.parquet` /
`grouped_validation_report.json` contract every audit already consumes.

What is selected on
-------------------
**Average precision on `ANY_ATTACK`, not macro F1 at the argmax.**
`ablations_baselines.md` SS11 is explicit about why: every argmax metric this
pipeline writes sits at whatever threshold the training prior implies, so a
grid point that shifts the score distribution without improving the ranking
reads as a large macro-F1 move and is worth nothing - and the reverse is
equally possible.  `ANY_ATTACK` is the axis `grouped_pr_curves.py` reports for
every learned run and the axis the D.2 comparison is made on, so selecting on
it keeps the selection criterion and the reported criterion the same object.

Macro F1 at the argmax is computed for every grid point anyway and recorded
next to the AP.  It costs nothing extra, and it turns "these two criteria
disagree" from an argument into a column.  `--selection-metric macro-f1` makes
the other choice, for measuring that disagreement rather than for reporting.

Where the grid comes from
-------------------------
SS9 and SS12 are equally explicit that the grid must be designed from the
champion's own error structure, **not** from a depth-and-estimators hunch: the
"capacity is the axis" conclusion belonged to the defective pool and was
withdrawn with it.  So the axes below are read off `pr_curves_d3_v2.md` and
the champion's per-class report:

  - `SAG.DB` and `SAG.PB` are strong rankers (AP 0.8839 / 0.8551) whose false
    alarms are overwhelmingly **each other** - 71.3% and 53.3% of them at the
    0.1%/1% budgets.  Two burst variants separated by a fine boundary is the
    one error in this matrix that more capacity could plausibly move, and
    `max_depth` is the axis that moves it.
  - `SAG.PBM` is the weakest ranker (AP 0.4094, recall 0.2773 at the argmax),
    and its false alarms are 57.2% `normal`.  `label_duplication_audit.md` SS7
    bounds how far any per-message model can go there, so **the expectation is
    a null**, stated before the run: if depth or fit budget recovers a
    meaningful part of `SAG.PBM`, the label-semantics explanation is weaker
    than the card claims, and that is worth knowing either way.
  - `FRG` loses 99.4% of its 0.1%-budget false alarms to `benign_degradation`,
    which is the by-construction `CONGESTION_LOSS` collision
    (`benign_controls.md` SS8).  No hyperparameter separates byte-identical
    rows.  **Expected null**, likewise stated in advance.
  - A leaf is allowed to form on a single row at the default
    `min_child_weight=1`.  With runs as the split unit, a leaf carved around
    one run's noise generalises to nothing, and the grouped protocol is what
    makes that measurable rather than invisible.

Three axes, twelve points, and the **first point is the library default**
(verified against `classifier("xgboost", ...)` bit for bit in the tests), so
"tuning bought nothing" is readable directly off the selection table instead
of being inferred.  Ties go to the earlier point, i.e. to the default: the
conservative direction, which never reports a gain that is really a tie.

Deliberately not in the grid
----------------------------
  - **Class weights / `scale_pos_weight`.**  That is rebalancing, card E owns
    it, and `--balance` is where it lives.  Smuggling it into the grid would
    re-open E's question inside D's and make the comparison against the
    untuned champion unreadable.
  - **`colsample_bytree`.**  D.1 measured that three columns (`timestampDiff`,
    `tDiff`, `timeFromLastChange`) carry the detection and 29 of 40 are free
    (SS15).  Column subsampling at 0.5 would keep the carriers out of half the
    trees, so the prediction is that it loses; it is left out to keep the grid
    small, and the prediction is recorded here rather than tested.
  - **Anything that changes the features or the folds.**  Those are D.1's and
    card B's, and a run that moved them would not be comparable to the
    champion it exists to be compared to.

The subsample, and the bias it carries
--------------------------------------
SS2 preregistered "a small grid on a documented subsample rather than an
exhaustive search on all 16.6M train rows", and `--inner-max-rows` is that
subsample.  It is drawn by `subsample_train`, so it is proportional within
every (`split_group`, `class`) stratum: **every training run is still
represented**, just thinner, and no rare class can be emptied.  Group
diversity - the thing the grouped protocol actually cares about - is
preserved exactly; what shrinks is rows per run.

The bias that leaves is real and points one way: less data favours smaller
capacity, so a grid selected on a subsample may under-shoot the depth that is
right at 9M.  That makes a null result here *partly* confounded with the
subsample, and the honest reading of "the default won" is "the default won at
this budget", not "no tuning could help".  `--inner-max-rows 0` removes the
cap for anyone with the compute to spend.

The default is 1.5M rows because the cost was measured rather than guessed
(`ablations_baselines.md` SS17): one inner split of 12 points takes 7.6 min at
800k rows on this machine, so the whole search is ~1.9 h there and ~3.5 h at
1.5M - both inside SS4's 4-8 h band, and the larger one buys a materially
weaker caveat on exactly the axis the grid is about.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

from benign_confusion_report import ATTACK_CLASSES
from run_grouped_validation import (
    SCORES_FILENAME,
    GroupedRunError,
    ScoreWriter,
    load_inputs,
    load_json,
    predict_proba_in_chunks,
    provenance_block,
    resample_train,
    run_folds,
    subsample_train,
)


class TuningError(ValueError):
    pass


# An axis is `(parameter names, tuple of value tuples)`.  Names travel in a
# tuple so that two parameters which only make sense together - more rounds at
# a smaller step - are one axis rather than a product that would also generate
# the two nonsensical corners (300 rounds at 0.3, 100 at 0.1).
GRIDS = {
    "champion-xgboost": {
        "model": "xgboost",
        "question": (
            "Is the D.3 champion already fitted, or is its library default an "
            "arbitrary point being reported as a capability claim?"
        ),
        "axes": (
            # Capacity. Aimed at the SAG.DB/SAG.PB boundary, which is where
            # the champion's false alarms actually concentrate.
            (("max_depth",), ((6,), (10,), (14,))),
            # The counterweight: with runs as the split unit, a leaf carved
            # around one run's noise generalises to nothing.
            (("min_child_weight",), ((1,), (20,))),
            # Fit budget, as one axis: three times the rounds at a third of
            # the step.
            (("n_estimators", "learning_rate"), ((100, 0.3), (300, 0.1))),
        ),
        # Recorded in the report so a reader of the result sees what was ruled
        # out and why, without going back to this docstring.
        "excluded": {
            "class weights / scale_pos_weight": (
                "rebalancing is card E's axis and lives in --balance; in the "
                "grid it would make the comparison against the untuned "
                "champion unreadable"),
            "colsample_bytree": (
                "D.1 SS15: three of 40 columns carry the detection, so column "
                "subsampling would keep the carriers out of half the trees - "
                "predicted to lose, left out to keep the grid small"),
        },
        # Stated before execution, so a null reads as a confirmation rather
        # than as a disappointment.
        "expected_nulls": {
            "SAG.PBM": (
                "bounded by per-message label semantics "
                "(label_duplication_audit.md SS7), not by capacity"),
            "FRG": (
                "byte-identical to the CONGESTION_LOSS benign control for 15 "
                "of 45 runs (benign_controls.md SS8); no hyperparameter "
                "separates identical rows"),
        },
    },
    # Two points, one axis: enough to exercise every moving part of the
    # nesting - inner split, both fits, the selection, the record - without
    # paying for twelve points. For validating wiring before committing to
    # the hours `champion-xgboost` costs, and for the test suite. A run
    # executed on this grid is not a D.4 result and its report says so via
    # `tuned`.
    "smoke-xgboost": {
        "model": "xgboost",
        "question": "Wiring check only - not a result.",
        "axes": ((("max_depth",), ((6,), (10,))),),
        "excluded": {},
        "expected_nulls": {},
    },
}

SELECTION_METRICS = ("any-attack-ap", "macro-f1")


def grid_definition(name):
    try:
        return GRIDS[name]
    except KeyError:
        raise TuningError(
            "unknown grid: %s (choose from %s)" % (name, ", ".join(sorted(GRIDS))))


def expand_grid(name):
    """The grid's points, in a fixed order whose first entry is the default.

    Order matters twice over: it is the tie-break (earlier wins, so a tie
    keeps the default and never reports a gain that is really a tie), and it
    is what makes `selected_index == 0` mean "tuning changed nothing" at a
    glance.
    """
    definition = grid_definition(name)
    axes = definition["axes"]
    points = []
    for combination in itertools.product(*[values for _, values in axes]):
        point = {}
        for (names, _), chosen in zip(axes, combination):
            point.update(dict(zip(names, chosen)))
        points.append(point)
    return points


def build_estimator(model_name, point, seed, n_jobs):
    """One grid point as a fitted-later estimator, on the D.3 family."""
    if model_name != "xgboost":
        raise TuningError(
            "grid tuning is implemented for the D.3 champion family only; "
            "got model=%r" % model_name)
    import xgboost as xgb

    return xgb.XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=seed, n_jobs=n_jobs, **point,
    )


def attack_columns(classes):
    """Posterior columns whose sum is the `ANY_ATTACK` score.

    The same definition `grouped_pr_curves.py` draws its `ANY_ATTACK` curve
    over, so the criterion selected on and the criterion reported are one
    quantity rather than two that happen to correlate.
    """
    columns = [index for index, name in enumerate(classes) if name in ATTACK_CLASSES]
    if not columns:
        raise TuningError("dataset's classes hold no attack class: %s" % list(classes))
    return columns


def score_point(proba, y_true, classes):
    """Both criteria for one fitted point on one inner fold.

    `any_attack_ap` is scikit-learn's exact average precision rather than
    `grouped_pr_curves.py`'s grid-quantised one: the quantisation exists to
    make a 265-run bootstrap cheap over a multi-gigabyte scores file, and an
    inner fold is scored once in memory, so it can afford the exact figure.
    The two agree far inside the interval either is reported with.

    `macro_f1` is the argmax reading of the same posteriors, recorded for
    every point whether or not it is selecting, because "these two criteria
    disagree" (SS11) should be a column rather than an argument.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score

    columns = attack_columns(classes)
    positives = np.isin(y_true, columns)
    score = proba[:, columns].sum(axis=1)
    if positives.any() and not positives.all():
        ap = float(average_precision_score(positives.astype("int8"), score))
    else:
        # An inner fold that holds no attack row (or nothing else) cannot rank
        # anything. Reported as NaN and ignored in the mean rather than scored
        # as zero, which would punish every point equally and silently.
        ap = float("nan")
    predicted = np.argmax(proba, axis=1)
    return {
        "any_attack_ap": ap,
        "macro_f1": float(f1_score(y_true, predicted,
                                   labels=list(range(len(classes))),
                                   average="macro", zero_division=0)),
        "positives": int(positives.sum()),
        "rows": int(len(y_true)),
    }


def inner_partitions(y, group_codes, positions, n_splits, seed):
    """Inner grouped folds over one outer fold's train rows.

    `StratifiedGroupKFold` for the same reason it is the outer protocol
    (`validation_protocol.md`): plain `GroupKFold` can starve a fold of an
    entire class when run sizes are uneven, and at 0.42-0.58% per attack
    class that is not a remote possibility.  Groups are `split_group` codes,
    so a run is intact inside the inner split exactly as it is inside the
    outer one - a hyperparameter chosen on a run's own rows would be the same
    leak one level down.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    groups = group_codes[positions]
    distinct = int(len(np.unique(groups)))
    if distinct < n_splits:
        raise TuningError(
            "inner split needs at least %d train groups, this fold offers %d"
            % (n_splits, distinct))
    labels = y[positions]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, test_idx)
            for train_idx, test_idx in splitter.split(positions, labels, groups=groups)]


def selection_binding(args, points, split, dataset_digest, fold_index):
    """Everything that could change which point a fold would select.

    The key of the selection cache below.  A search costs ~38 min per outer
    fold on the real pool, so replaying one that is still valid is worth
    recovering after an interruption - but only if "still valid" is decided
    by the whole of what went into it, not by a filename.  Anything that
    could move a score belongs here, and the conservative reading of "could"
    is the one taken:

      - the dataset hash, the fold's own train groups, the feature set and the
        discard list decide **which rows and columns** the search saw;
      - the grid, the metric, the inner split count, the inner subsample, the
        balancing and the seed decide **what was compared and how** - and the
        fold's *position* travels with the seed, because every fit in the
        search is seeded at ``seed + fold_index``, so the same fold read at a
        different index is a different search;
      - ``n_jobs`` is in here although it is a throughput knob and not a
        hyperparameter, because XGBoost's histogram reduction order depends on
        the thread count and can move the last decimals of a score
        (``run_grouped_validation.classifier``).  With observed gaps of
        ~0.001 between points, a last-decimal change can flip a selection, so
        a cached choice is reused only where it would have been computed
        identically.
    """
    return {
        "dataset_sha256": dataset_digest,
        "split_id": split["split_id"],
        "fold_index": fold_index,
        "train_groups": sorted(str(group) for group in split["train_groups"]),
        "model": args.model,
        "grid": args.grid,
        "points": points,
        "selection_metric": args.selection_metric,
        "inner_splits": args.inner_splits,
        "inner_max_rows": args.inner_max_rows,
        "balance": args.balance,
        "smote_oversample_factor": args.smote_oversample_factor,
        "smote_max_target": args.smote_max_target,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "feature_set": args.feature_set,
        "discard_columns": sorted(args.discard_column),
        "max_train_rows_per_fold": args.max_train_rows_per_fold,
        "max_rows_per_group_class": args.max_rows_per_group_class,
    }


def binding_digest(binding):
    import hashlib

    canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class GridSelector:
    """`run_folds`'s `model_selector`: choose a point, hand back the estimator.

    Called once per outer fold with that fold's TRAIN positions and nothing
    else, which is what makes the nesting structural rather than a promise -
    there is no test row in scope to select on.
    """

    def __init__(self, model_name, points, inner_splits, inner_max_rows, seed,
                 n_jobs, balance, smote_factor, smote_max_target, metric,
                 log=None, cache_dir=None, bindings=None):
        self.model_name = model_name
        self.points = points
        self.inner_splits = inner_splits
        self.inner_max_rows = inner_max_rows
        self.seed = seed
        self.n_jobs = n_jobs
        self.balance = balance
        self.smote_factor = smote_factor
        self.smote_max_target = smote_max_target
        if metric not in SELECTION_METRICS:
            raise TuningError("unknown selection metric: %s" % metric)
        self.metric = metric
        self.log = log or (lambda message: None)
        # The selection cache. A fold's search is the expensive part of this
        # runner (~38 min against ~8 min for the refit that follows it), and
        # a run interrupted midway - this one was killed once by system
        # memory pressure - otherwise repeats every completed search from
        # scratch. Cached by `binding_digest`, so a cache entry is reused
        # only where the search would have been recomputed identically; it
        # is a speed recovery, never a different answer.
        self.cache_dir = cache_dir
        self.bindings = bindings or {}

    def _cache_path(self, split_id):
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, "%s.json" % split_id)

    def load_cached(self, split_id):
        """A previous search for this fold, if it was the same search.

        Any failure to read, parse or match is treated as "no cache": a
        selection is cheap to recompute and impossible to verify after the
        fact, so a damaged or foreign entry is discarded rather than trusted.
        """
        path = self._cache_path(split_id)
        expected = self.bindings.get(split_id)
        if not path or not expected or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if payload.get("binding_sha256") != expected:
            return None
        record = payload.get("record")
        return record if isinstance(record, dict) else None

    def store_cached(self, split_id, record):
        path = self._cache_path(split_id)
        expected = self.bindings.get(split_id)
        if not path or not expected:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Staged and renamed, so a process killed mid-write cannot leave
            # a truncated entry that a later run would have to distrust.
            with open(path + ".tmp", "w", encoding="utf-8", newline='\n') as fh:
                json.dump({"binding_sha256": expected, "record": record}, fh, indent=2)
                fh.write('\n')
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    def __call__(self, fold_index, split_id, X, y, group_codes, train_positions,
                 classes):
        import gc

        import numpy as np

        cached = self.load_cached(split_id)
        if cached is not None:
            self.log("  %s selection reused from cache: %s (%s=%.4f)"
                     % (split_id, cached["selected_params"],
                        cached["selection_metric"], cached["score"]))
            record = dict(cached, from_cache=True)
            return {
                "estimator": build_estimator(
                    self.model_name, record["selected_params"],
                    self.seed + fold_index, self.n_jobs),
                "record": record,
            }

        # Every grid point is fitted at the same seed on the same inner
        # partitions, so a difference between two points is the
        # hyperparameters and not a reseeding.
        fold_seed = self.seed + fold_index
        strata = (group_codes[train_positions].astype("int32") * len(classes)
                  + y[train_positions])
        keep = subsample_train(strata, self.inner_max_rows, fold_seed)
        inner_positions = train_positions if keep is None else train_positions[keep]
        del strata, keep

        partitions = inner_partitions(
            y, group_codes, inner_positions, self.inner_splits, fold_seed)
        X_inner = X[inner_positions]
        y_inner = y[inner_positions]
        groups_inner = group_codes[inner_positions]

        rows = [{"point": point, "folds": []} for point in self.points]
        started = time.time()
        for inner_index, (train_idx, test_idx) in enumerate(partitions):
            # Balanced once per inner fold, not once per point: the seed is
            # fixed, so every point would otherwise be handed an identical
            # resample recomputed twelve times.
            X_train, y_train = resample_train(
                X_inner[train_idx], y_inner[train_idx], classes, self.balance,
                fold_seed, self.smote_factor, self.smote_max_target)
            for row, point in zip(rows, self.points):
                model = build_estimator(self.model_name, point, fold_seed, self.n_jobs)
                model.fit(X_train, y_train)
                proba = predict_proba_in_chunks(
                    model, X_inner, test_idx)
                del model
                row["folds"].append(score_point(proba, y_inner[test_idx], classes))
                del proba
                gc.collect()
            del X_train, y_train
            self.log("  %s inner fold %d/%d done (%.1f min elapsed)"
                     % (split_id, inner_index + 1, len(partitions),
                        (time.time() - started) / 60.0))
        del X_inner, y_inner
        gc.collect()

        key = "any_attack_ap" if self.metric == "any-attack-ap" else "macro_f1"
        for row in rows:
            for name in ("any_attack_ap", "macro_f1"):
                values = [fold[name] for fold in row["folds"]]
                finite = [v for v in values if v == v]
                row["mean_%s" % name] = float(np.mean(finite)) if finite else float("nan")
                row["std_%s" % name] = (float(np.std(finite, ddof=0))
                                        if len(finite) > 1 else 0.0)
            row["score"] = row["mean_%s" % key]
        # NaN != NaN: an unscorable point is dropped from the selection
        # rather than ranked against the others as if it had scored.
        scored = [index for index, row in enumerate(rows) if row["score"] == row["score"]]
        if not scored:
            raise TuningError(
                "%s: no grid point could be scored - every inner fold was "
                "unrankable on %s" % (split_id, self.metric))
        # max() keeps the first of equal values, so a tie goes to the earlier
        # point and the default (index 0) wins any tie it is part of.
        selected = max(scored, key=lambda index: rows[index]["score"])
        chosen = self.points[selected]
        self.log("  %s selected %s (%s=%.4f, default=%.4f) in %.1f min"
                 % (split_id, chosen, self.metric, rows[selected]["score"],
                    rows[0]["score"], (time.time() - started) / 60.0))
        record = {
            "selection_metric": self.metric,
            "selected_index": selected,
            "selected_params": chosen,
            "selected_is_default": selected == 0,
            "score": rows[selected]["score"],
            "default_score": rows[0]["score"],
            "gain_over_default": rows[selected]["score"] - rows[0]["score"],
            "inner": {
                "splits": self.inner_splits,
                "max_rows": self.inner_max_rows or None,
                "rows_available": int(len(train_positions)),
                "rows_used": int(len(inner_positions)),
                "groups": int(len(np.unique(groups_inner))),
                "seed": fold_seed,
                "balance": self.balance,
                "fold_rows": [int(len(test_idx)) for _, test_idx in partitions],
            },
            "seconds": time.time() - started,
            "from_cache": False,
            "grid": rows,
        }
        self.store_cached(split_id, record)
        estimator = build_estimator(self.model_name, chosen, fold_seed, self.n_jobs)
        return {"estimator": estimator, "record": record}


def plan(points, n_outer, inner_splits):
    """What the run will cost, before it is paid.

    A nested search is the one item in card D whose cost is a product rather
    than a sum (SS4: "the multiplier that blows the schedule"), so the fit
    count is printable without touching the dataset.
    """
    inner_fits = n_outer * inner_splits * len(points)
    return {
        "grid_points": len(points),
        "outer_folds": n_outer,
        "inner_splits": inner_splits,
        "inner_fits": inner_fits,
        "refits": n_outer,
        "total_fits": inner_fits + n_outer,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="xgboost",
                        help="Model family to tune. Card D.4 tunes the D.3 champion.")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="champion-xgboost",
                        help="Preregistered grid, designed from the champion's error "
                             "structure (ablations_baselines.md SS17).")
    parser.add_argument("--inner-splits", type=int, default=3,
                        help="Inner StratifiedGroupKFold folds, drawn from the outer "
                             "fold's TRAIN groups only.")
    parser.add_argument("--inner-max-rows", type=int, default=1_500_000,
                        help="Documented subsample the inner search runs on, "
                             "proportional within every (split_group, class) stratum "
                             "so every training run stays represented. 0 removes the "
                             "cap. See the module docstring for the bias it carries.")
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS,
                        default="any-attack-ap",
                        help="What the inner folds are ranked on. AP on ANY_ATTACK by "
                             "default: an argmax metric would select a threshold "
                             "artifact at 2%% prevalence (ablations_baselines.md SS11).")
    parser.add_argument("--plan-only", action="store_true",
                        help="Print the grid and the fit count, then exit. Reads the "
                             "split file only - it verifies nothing and trains nothing.")
    parser.add_argument("--selection-cache", default=None, metavar="DIR",
                        help="Directory in which each outer fold's completed search is "
                             "recorded, so a run killed midway replays it instead of "
                             "repeating ~38 min of fits. An entry is reused only when "
                             "the whole selection binding matches (see "
                             "`selection_binding`); anything else is recomputed. Off "
                             "by default: without this flag the run is bit for bit "
                             "what it was before the cache existed.")
    # Everything below is passed straight through to the shared runner, so a
    # tuned run is configured exactly like the run it is compared against.
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-group-class", type=int, default=0,
                        help="Non-zero enables a technical smoke sample.")
    parser.add_argument("--discard-column", action="append", default=[])
    parser.add_argument("--feature-set", default="all",
                        help="Checklist D.1's feature groups; 'all' is the reference "
                             "and the configuration the champion was chosen at.")
    parser.add_argument("--balance", choices=["none", "downsample", "smote"], default="none",
                        help="Applied to the inner TRAIN partitions as well as the "
                             "outer one, so the inner search is a replica of the "
                             "pipeline it is selecting for.")
    parser.add_argument("--smote-oversample-factor", type=float, default=20.0)
    parser.add_argument("--smote-max-target", type=int, default=200_000)
    parser.add_argument("--max-train-rows-per-fold", type=int, default=0,
                        help="Applied before selection, so the inner search sees "
                             "exactly the rows the final fit will see.")
    parser.add_argument("--save-scores", action="store_true",
                        help="Write grouped_scores.parquet. Required by "
                             "grouped_pr_curves.py, which is the only place a D.4 "
                             "result may be read from (ablations_baselines.md SS11).")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    predictions_tmp = scores_tmp = None
    try:
        definition = grid_definition(args.grid)
        if definition["model"] != args.model:
            raise TuningError(
                "grid %r is designed for model %r, not %r: the axes were read off "
                "that family's error structure and mean nothing on another one"
                % (args.grid, definition["model"], args.model))
        points = expand_grid(args.grid)

        if args.plan_only:
            split_payload = load_json(args.splits)
            costs = plan(points, len(split_payload["splits"]), args.inner_splits)
            print("Grid %s (%s): %s" % (args.grid, args.model, definition["question"]))
            for index, point in enumerate(points):
                print("  [%2d]%s %s" % (index, " default" if index == 0 else "        ", point))
            print("Plan: %(outer_folds)d outer folds x %(inner_splits)d inner folds x "
                  "%(grid_points)d points = %(inner_fits)d inner fits, plus "
                  "%(refits)d refits = %(total_fits)d fits." % costs)
            print("Inner subsample: %s rows per outer fold."
                  % (args.inner_max_rows or "uncapped"))
            return 0

        inputs = load_inputs(args)
        arrays = inputs["arrays"]
        split_payload = inputs["splits"]
        os.makedirs(args.out_dir, exist_ok=True)

        # One binding per outer fold, computed before the first fit. They are
        # derived from the arguments and the verified dataset digest, so a
        # cache written by a run that differed in any of them - a different
        # grid, a different subsample, a different dataset - cannot be read
        # back here.
        bindings = {
            split["split_id"]: binding_digest(
                selection_binding(args, points, split, inputs["digest"], fold_index))
            for fold_index, split in enumerate(split_payload["splits"])
        } if args.selection_cache else {}

        selector = GridSelector(
            args.model, points, args.inner_splits, args.inner_max_rows, args.seed,
            args.n_jobs, args.balance, args.smote_oversample_factor,
            args.smote_max_target, args.selection_metric,
            log=lambda message: print(message, flush=True),
            cache_dir=args.selection_cache, bindings=bindings,
        )

        predictions_path = os.path.join(args.out_dir, "grouped_predictions.csv")
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
                    fh, fieldnames=["split_id", "row_index", "split_group",
                                    "y_true", "y_pred"])
                writer.writeheader()
                metrics, classes, features = run_folds(
                    arrays, split_payload["splits"], args.model, args.seed, writer,
                    balance_strategy=args.balance,
                    smote_factor=args.smote_oversample_factor,
                    smote_max_target=args.smote_max_target,
                    max_train_rows=args.max_train_rows_per_fold,
                    n_jobs=args.n_jobs, scores_writer=scores_writer,
                    model_selector=selector,
                )
        finally:
            if scores_writer is not None:
                scores_writer.close()

        report = provenance_block(args, inputs, classes, features)
        # `model` stays the family name so every existing reader keeps working;
        # `tuned` is what stops a tuned run being mistaken for the untuned
        # champion whose report is otherwise identically shaped.
        report["tuned"] = args.grid
        report["tuning"] = {
            "grid": args.grid,
            "question": definition["question"],
            "selection_metric": args.selection_metric,
            "points": points,
            "default_point_index": 0,
            "inner_splits": args.inner_splits,
            "inner_max_rows": args.inner_max_rows or None,
            "excluded_axes": definition["excluded"],
            "expected_nulls": definition["expected_nulls"],
            "plan": plan(points, len(split_payload["splits"]), args.inner_splits),
            # Which folds this process actually searched, and which it replayed
            # from a previous one. The plan above counts the fits the protocol
            # calls for; this counts the ones that were paid here, so a report
            # never implies a search it did not run.
            "selection_cache": {
                "dir": os.path.abspath(args.selection_cache) if args.selection_cache else None,
                "folds_reused": [fold["split_id"] for fold in metrics
                                 if fold.get("selection", {}).get("from_cache")],
            },
        }
        report["fold_metrics"] = metrics
        with open(os.path.join(args.out_dir, "grouped_validation_report.json"),
                  "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        os.replace(predictions_tmp, predictions_path)
        if scores_tmp is not None:
            os.replace(scores_tmp, scores_path)
    except (OSError, json.JSONDecodeError, GroupedRunError, TuningError, ValueError) as exc:
        for stale in (predictions_tmp, scores_tmp):
            try:
                if stale and os.path.exists(stale):
                    os.remove(stale)
            except OSError:
                pass
        print("NESTED TUNING FAILED\n%s" % exc, file=sys.stderr)
        return 1
    chosen = [fold["selection"]["selected_index"] for fold in metrics]
    print("Tuned %d grouped folds on %d rows (%s). Selected points per fold: %s%s"
          % (len(metrics), report["rows_used"], report["status"], chosen,
             " - the library default won every fold." if set(chosen) == {0} else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
