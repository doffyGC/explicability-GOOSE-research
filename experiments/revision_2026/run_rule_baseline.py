"""Checklist D.2: the threshold rule the learned detector has to beat.

The reviewer's question is not "does this model score well" but "does it earn
its complexity".  A grayhole that drops messages stretches the interval to the
next one; a single threshold on that interval is the cheapest detector anybody
would deploy, and every number in card D is worth reporting only against it.

**The rule to beat was measured, not guessed.**  `ablations_baselines.md` SS14
established by elimination that the nine within-trace deltas carry essentially
all of the detection, and SS15 split that group and named the carriers: the
three *timing* deltas cost -0.0645 AP when removed and are the only sub-group
whose removal moves an operating point, while the four size/state deltas are
free.  So the preregistered scope for D.2 ("`sqNum`/`stNum` gap detector +
delay threshold", SS2) is implemented with the weight on the interval, and the
gap detector is kept and reported rather than dropped - a preregistered arm is
not removed because a later result predicts it will lose.  `sqnum-gap` is that
prediction's test.

Every rule here is one comparison against one threshold.  The threshold is the
only thing fitted, and it is fitted **on each fold's train partition alone**,
from the same persisted `splits_grouped.json` the learned runs consume - a
baseline calibrated on the data it is scored against would beat the model for
the wrong reason.

Artifact contract
-----------------
The run writes exactly what `run_grouped_validation.py` writes -
`grouped_predictions.csv`, `grouped_scores.parquet`, and a
`grouped_validation_report.json` of the same shape - so
`check_prediction_integrity.py`, `bootstrap_run_intervals.py` and
`grouped_pr_curves.py` consume it unchanged, and the baseline is audited by
the same scripts under the same invariants as the model it is compared to.

That contract is multiclass and the rule is binary, which is a real mismatch
and is resolved in the open rather than hidden:

  - `y_true` keeps the dataset's own six classes, untouched.
  - A firing row is predicted as the **designated attack class** - the most
    frequent attack class in *that fold's train partition*, recorded per fold
    in the report.  A row that does not fire is predicted `normal`.
  - The posterior block puts the rule's score on the designated class and the
    rest on `normal`, **exactly zero elsewhere**.

The zero columns are a deliberate, tested property, and they have one
consequence worth saying out loud: in `grouped_pr_curves.py`'s per-class
table, the three non-designated attack classes are scored by an all-zero
column, so their average precision comes back at the prevalence floor.  That
is an artifact of the projection, not a measurement of the rule.  Only the
`ANY_ATTACK` row, and the designated class's own row read as that one class,
carry anything.

The consequence is stated once here and repeated in the docs: **only the
`ANY_ATTACK` axis of this run is a result.**  Its macro F1 over six classes is
structurally floored - the rule cannot name a family, so three of the four
attack classes are predicted never - and reporting that number as if it were
comparable to the champion's would be a straw man.  `ANY_ATTACK` is exactly
the axis `grouped_pr_curves.py` reports for every learned run, and on it the
comparison is honest and direct.

The score
---------
`argmax(posterior)` has to reproduce the rule's own decision or the integrity
audit is measuring something else, so the score is built to cross 0.5 exactly
at the calibrated threshold: the train partition's empirical CDF, rescaled
piecewise so that `theta` maps to 0.5, then clamped by one float32 ulp on the
firing side.  The map is **strictly monotone in the rule's feature**, so it
changes no ranking and therefore no AP, precision or recall at any budget -
it only decides how the curve is sampled by `grouped_pr_curves.py`'s logit
grid, and a rank-based score spreads it across that grid instead of piling
every row onto two points.  It is fitted on train rows only, like the
threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

from benign_confusion_report import ATTACK_CLASSES
from run_grouped_validation import (
    GroupedRunError,
    ScoreWriter,
    average_block,
    class_counts,
    load_json,
    verify_artifacts,
)

HERE = os.path.dirname(os.path.abspath(__file__))

SCORES_FILENAME = "grouped_scores.parquet"
NORMAL_LABEL = "normal"

# How many distinct thresholds a fold's calibration considers when the feature
# has more distinct values than that.  Every distinct value is used when it
# fits, so an integer-valued rule like `sqnum-gap` is searched exhaustively;
# only the continuous ones are quantised, and 4,096 rank-uniform candidates
# resolve the train ECDF to ~0.02%, far finer than the difference between two
# adjacent operating points on 9M train rows.
MAX_THRESHOLD_CANDIDATES = 4096

# The rules.  Each is one feature and one direction, because the point of a
# baseline is to be the thing nobody could call complex.  `question` is what
# the run answers and is copied into the report so a result cannot be read
# without it.
RULES = {
    "interval-timestamp": {
        "column": "timestampDiff",
        "direction": "above",
        "question": "Does a threshold on the GOOSE timestamp interval detect the discards?",
    },
    "interval-t": {
        "column": "tDiff",
        "direction": "above",
        "question": "Same, on the capture-clock interval rather than the published timestamp.",
    },
    "time-since-change": {
        "column": "timeFromLastChange",
        "direction": "above",
        "question": "Is it enough to watch how long a publisher has been silent since its last state change?",
    },
    "delay": {
        "column": "delay",
        "direction": "above",
        "question": "The preregistered delay threshold (SS2): is the per-message transport delay alone enough?",
    },
    "sqnum-gap": {
        "column": "sqDiff",
        "direction": "above",
        "question": "The preregistered gap detector (SS2): does a jump in SqNum mark the discards? SS15 predicts it loses to the interval.",
    },
    "stnum-gap": {
        "column": "stDiff",
        "direction": "above",
        "question": "Same for the state counter, which is what resets at the boundaries where SAG.PBM discards.",
    },
}


class RuleBaselineError(ValueError):
    pass


def rule_definition(name):
    try:
        return RULES[name]
    except KeyError:
        raise RuleBaselineError(
            "unknown rule %r; known rules: %s" % (name, ", ".join(sorted(RULES))))


def load_rule_arrays(path, column, group_column, target_column):
    """Read only the three columns a rule needs, row group by row group.

    A rule reads one feature.  Loading the 40-column matrix
    `run_grouped_validation.py` needs would cost ~3.5 GB and most of the run's
    wall clock for columns no threshold will ever look at.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    schema = handle.schema_arrow
    for name in (column, group_column, target_column):
        if name not in schema.names:
            raise RuleBaselineError("dataset is missing column %r" % name)
    field = schema.field(column)
    if not (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)):
        raise RuleBaselineError(
            "rule feature %r is %s, which no threshold orders" % (column, field.type))

    rows = handle.metadata.num_rows
    if rows == 0:
        raise RuleBaselineError("dataset holds no rows")
    values = np.empty(rows, dtype="float64")
    group_codes = np.empty(rows, dtype="int32")
    target_codes = np.empty(rows, dtype="int32")
    group_dictionaries, target_dictionaries, spans = [], [], []
    encoded_type = pa.dictionary(pa.int32(), pa.string())

    offset = 0
    for number in range(handle.num_row_groups):
        table = handle.read_row_group(
            number, columns=[column, group_column, target_column])
        count = table.num_rows
        values[offset:offset + count] = table.column(column).to_numpy(
            zero_copy_only=False)
        for name, dictionaries, destination in (
            (group_column, group_dictionaries, group_codes),
            (target_column, target_dictionaries, target_codes),
        ):
            series = table.column(name).cast(encoded_type).to_pandas()
            local = series.cat.codes.to_numpy()
            if (local < 0).any():
                raise RuleBaselineError(
                    "column %s holds nulls, which have no class or group" % name)
            dictionaries.append([str(value) for value in series.cat.categories])
            destination[offset:offset + count] = local
        spans.append((offset, count))
        offset += count
        del table

    def globalise(dictionaries, codes):
        # sorted(), matching `load_grouped_arrays`: the class and group
        # orderings have to be the ones every other artifact in the chain
        # already uses, not merely a defensible ordering of their own.
        labels = sorted({label for dictionary in dictionaries for label in dictionary})
        index = {label: position for position, label in enumerate(labels)}
        for (start, count), dictionary in zip(spans, dictionaries):
            lookup = np.asarray([index[label] for label in dictionary], dtype="int32")
            codes[start:start + count] = lookup[codes[start:start + count]]
        return labels

    group_labels = globalise(group_dictionaries, group_codes)
    classes = globalise(target_dictionaries, target_codes)
    return {
        "values": values,
        "y": target_codes.astype("int16"),
        "group_codes": group_codes,
        "group_labels": group_labels,
        "row_index": np.arange(rows, dtype="int64"),
        "classes": classes,
        "column": column,
    }


def attack_mask(y, classes):
    """The binary target the rule is actually calibrated and judged on."""
    import numpy as np

    codes = [index for index, name in enumerate(classes) if name in ATTACK_CLASSES]
    if not codes:
        raise RuleBaselineError(
            "dataset holds no attack class, so there is nothing to detect: %s" % classes)
    return np.isin(y, np.asarray(codes, dtype=y.dtype))


def threshold_candidates(values):
    """Every distinct value, or a rank-uniform sample when there are too many.

    Returns the candidates and whether they are the exhaustive set, because a
    rule searched over every distinct value of its feature and one searched
    over a 4,096-point quantile grid are different claims and the report has
    to be able to tell them apart.
    """
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuleBaselineError("rule feature holds no finite value to threshold")
    distinct = np.unique(finite)
    if distinct.size <= MAX_THRESHOLD_CANDIDATES:
        return distinct, True
    quantiles = np.linspace(0.0, 1.0, MAX_THRESHOLD_CANDIDATES)
    return np.unique(np.quantile(finite, quantiles)), False


def calibrate(values, positives):
    """Pick the threshold maximising F1 on ANY_ATTACK over the rows given.

    F1 rather than accuracy: at ~2% prevalence, "never fire" already scores
    98% accurate, so accuracy would select a detector that detects nothing.
    F1 is the harmonic mean the rule is reported at, it needs no cost ratio
    nobody has agreed on, and the whole threshold *curve* is recovered anyway
    by `grouped_pr_curves.py` - this choice fixes the single operating point
    the multiclass report shows, not the result.

    Ties are broken towards the **higher** threshold, which is the more
    conservative detector (fewer alerts) among equals.
    """
    import numpy as np

    candidates, exhaustive = threshold_candidates(values)
    finite = np.isfinite(values)
    positive_values = np.sort(values[positives & finite])
    negative_values = np.sort(values[(~positives) & finite])
    total_positive = int(positives.sum())
    if total_positive == 0:
        raise RuleBaselineError("train partition holds no attack row to calibrate on")

    # A row fires when value > theta, so the rows at or below theta are
    # exactly `searchsorted(..., side='right')`. Non-finite rows never fire
    # and are therefore already excluded from both sorted arrays while still
    # counting in `total_positive` - a NaN attack row is a miss, not an
    # absence.
    true_positive = positive_values.size - np.searchsorted(
        positive_values, candidates, side="right")
    false_positive = negative_values.size - np.searchsorted(
        negative_values, candidates, side="right")
    false_negative = total_positive - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = np.divide(2.0 * true_positive, denominator,
                   out=np.zeros(candidates.shape, dtype="float64"),
                   where=denominator > 0)
    best = int(np.flatnonzero(f1 == f1.max())[-1])
    threshold = float(candidates[best])
    return {
        "threshold": threshold,
        "train_f1": float(f1[best]),
        "train_true_positive": int(true_positive[best]),
        "train_false_positive": int(false_positive[best]),
        "train_false_negative": int(false_negative[best]),
        "train_positive_rows": total_positive,
        "train_alert_rate": float(
            (true_positive[best] + false_positive[best]) / float(values.size)),
        "candidates_considered": int(candidates.size),
        "candidates_exhaustive": bool(exhaustive),
    }


def ecdf_score(train_values, threshold, values):
    """A strictly monotone [0, 1] score that crosses 0.5 exactly at `threshold`.

    Built from the *train* partition's ECDF so that nothing about the test
    rows enters it, and rescaled piecewise - `[0, F(theta)] -> [0, 0.5]` and
    `(F(theta), 1] -> (0.5, 1]` - so the rule's own decision is what
    `argmax` recovers.  The final clamp is one float32 ulp: the posteriors are
    persisted as float32, and a firing row whose rescaled score lands exactly
    on 0.5 after rounding would otherwise argmax to `normal` and make the
    integrity audit report a disagreement that is an artifact of this
    function.

    Monotone means AP and every recall-at-budget are untouched by it; see the
    module docstring.
    """
    import numpy as np

    finite_train = np.sort(train_values[np.isfinite(train_values)])
    if finite_train.size == 0:
        raise RuleBaselineError("train partition holds no finite value to rank")
    total = float(finite_train.size)
    at_threshold = float(np.searchsorted(finite_train, threshold, side="right")) / total

    ranks = np.searchsorted(finite_train, values, side="right") / total
    fires = np.isfinite(values) & (values > threshold)
    scores = np.empty(values.shape, dtype="float64")

    below = ~fires
    if at_threshold > 0:
        scores[below] = 0.5 * np.minimum(ranks[below] / at_threshold, 1.0)
    else:
        scores[below] = 0.0
    span = 1.0 - at_threshold
    if span > 0:
        scores[fires] = 0.5 + 0.5 * np.clip(
            (ranks[fires] - at_threshold) / span, 0.0, 1.0)
    else:
        scores[fires] = 1.0
    # Non-finite rows carry no evidence either way and must rank last.
    scores[~np.isfinite(values)] = 0.0

    scores = scores.astype("float32")
    half = np.float32(0.5)
    # Strict on *both* sides. `argmax` breaks a tie towards the lower class
    # index, and the class vocabulary is sorted, so every attack class sorts
    # before `normal` - a non-firing row scoring exactly 0.5 would argmax to
    # the designated attack class and contradict its own `y_pred`.
    scores[fires] = np.maximum(scores[fires], np.nextafter(half, np.float32(1.0)))
    scores[below] = np.minimum(scores[below], np.nextafter(half, np.float32(0.0)))
    return scores, fires


def designated_attack_class(y_train, classes):
    """The class a firing row is reported as: the train partition's commonest attack.

    Chosen per fold and from train rows only, for the same reason the
    threshold is.  It carries no claim - the rule cannot tell one family from
    another - and exists so the run can be written in the multiclass format
    every audit in this chain already reads.  The report records it per fold
    so that a reader can see whether it even stayed constant.
    """
    import numpy as np

    counts = {
        name: int((y_train == index).sum())
        for index, name in enumerate(classes) if name in ATTACK_CLASSES
    }
    if not counts or max(counts.values()) == 0:
        raise RuleBaselineError("train partition holds no attack row to designate")
    best = max(sorted(counts), key=lambda name: counts[name])
    return best, counts


def run_folds(arrays, splits, predictions_writer, scores_writer=None):
    import gc

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report

    values = arrays["values"]
    y = arrays["y"]
    group_codes = arrays["group_codes"]
    group_labels = arrays["group_labels"]
    row_index = arrays["row_index"]
    classes = np.asarray(arrays["classes"], dtype=object)
    class_list = list(arrays["classes"])
    all_labels = list(range(len(classes)))
    if NORMAL_LABEL not in class_list:
        raise RuleBaselineError(
            "dataset has no %r class for a non-firing row to be reported as"
            % NORMAL_LABEL)
    normal_code = class_list.index(NORMAL_LABEL)
    attacks = attack_mask(y, class_list)
    label_to_code = {label: code for code, label in enumerate(group_labels)}
    metrics = []

    for split in splits:
        def mask_for(partition):
            codes = [label_to_code[g] for g in split[partition] if g in label_to_code]
            return np.isin(group_codes, np.asarray(codes, dtype=group_codes.dtype))

        train_mask = mask_for("train_groups")
        test_mask = mask_for("test_groups")
        if (train_mask & test_mask).any():
            raise RuleBaselineError(
                "%s has row-level train/test overlap" % split["split_id"])
        if not train_mask.any() or not test_mask.any():
            raise RuleBaselineError("%s has an empty partition" % split["split_id"])

        train_positions = np.flatnonzero(train_mask)
        test_positions = np.flatnonzero(test_mask)
        del train_mask, test_mask
        gc.collect()

        train_values = values[train_positions]
        y_train = y[train_positions]
        calibration = calibrate(train_values, attacks[train_positions])
        designated, attack_counts = designated_attack_class(y_train, class_list)
        designated_code = class_list.index(designated)
        counts_train = class_counts(y_train, classes)
        del y_train

        test_values = values[test_positions]
        scores, fires = ecdf_score(
            train_values, calibration["threshold"], test_values)
        del train_values, test_values
        gc.collect()

        predicted = np.where(fires, designated_code, normal_code).astype("int64")
        y_test = y[test_positions]
        report = classification_report(
            y_test, predicted, labels=all_labels,
            target_names=classes, output_dict=True, zero_division=0,
        )
        attacks_test = attacks[test_positions]
        true_positive = int((fires & attacks_test).sum())
        false_positive = int((fires & ~attacks_test).sum())
        false_negative = int((~fires & attacks_test).sum())
        precision = true_positive / float(true_positive + false_positive or 1)
        recall = true_positive / float(true_positive + false_negative or 1)
        metrics.append({
            "split_id": split["split_id"],
            "train_rows": int(len(train_positions)),
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
            # The only line of this block that is a result. Everything above
            # it is the multiclass projection described in the module
            # docstring and is floored by construction.
            "any_attack": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / ((precision + recall) or 1)),
                "alert_rate": float(fires.sum() / float(fires.size)),
                "positive_rows": int(attacks_test.sum()),
            },
            "rule": dict(calibration, designated_attack_class=designated,
                         train_attack_class_counts=attack_counts),
            "subsample": {
                "max_train_rows_per_fold": None,
                "train_rows_available": int(len(train_positions)),
                "train_rows_sampled": int(len(train_positions)),
                "applied": False,
            },
            # Shaped like a learned run's so `grouped_pr_curves.py` can read
            # the prior it was "trained" at. A rule has no training prior to
            # correct, which is exactly what `balance: none` means here.
            "balance": {
                "strategy": "none",
                "train_rows_resampled": int(len(train_positions)),
                "train_class_counts_before": counts_train,
                "train_class_counts_after": counts_train,
            },
        })
        true_labels = classes[y_test]
        predicted_labels = classes[predicted]
        for position, truth, prediction in zip(test_positions, true_labels, predicted_labels):
            predictions_writer.writerow({
                "split_id": split["split_id"], "row_index": int(row_index[position]),
                "split_group": group_labels[group_codes[position]],
                "y_true": truth, "y_pred": prediction,
            })
        if scores_writer is not None:
            proba = np.zeros((len(test_positions), len(classes)), dtype="float32")
            proba[:, designated_code] = scores
            proba[:, normal_code] = np.float32(1.0) - scores
            scores_writer.write_fold(
                split["split_id"], row_index[test_positions],
                group_codes[test_positions], y_test, proba,
            )
            del proba
        del (train_positions, test_positions, y_test, predicted, scores, fires,
             true_labels, predicted_labels, attacks_test)
        gc.collect()
    return metrics, class_list


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibrate and score checklist D.2's threshold baseline on "
                    "the persisted grouped splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Rules:\n" + "\n".join(
            "  %-20s %s (%s)" % (name, RULES[name]["question"], RULES[name]["column"])
            for name in sorted(RULES)),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rule", required=True, choices=sorted(RULES))
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Recorded for provenance; a threshold search is "
                             "deterministic and does not consume it.")
    parser.add_argument("--save-scores", action="store_true",
                        help="Also write grouped_scores.parquet, which is what "
                             "grouped_pr_curves.py needs to put this baseline "
                             "on the same threshold axis as the learned runs.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    predictions_tmp = scores_tmp = None
    try:
        definition = rule_definition(args.rule)
        preparation = load_json(args.preparation_report)
        split_payload = load_json(args.splits)
        digest = verify_artifacts(args.dataset, preparation, split_payload)

        arrays = load_rule_arrays(
            args.dataset, definition["column"], args.group_column, args.target_column)
        rows_used = len(arrays["row_index"])
        os.makedirs(args.out_dir, exist_ok=True)
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
                metrics, classes = run_folds(
                    arrays, split_payload["splits"], writer,
                    scores_writer=scores_writer)
        finally:
            if scores_writer is not None:
                scores_writer.close()
        report = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "status": "full_grouped_run",
            "dataset": os.path.abspath(args.dataset),
            "dataset_sha256": digest,
            "splits": os.path.abspath(args.splits),
            "protocol": split_payload["protocol"],
            "model": "rule:%s" % args.rule,
            "rule": {
                "name": args.rule,
                "column": definition["column"],
                "direction": definition["direction"],
                "question": definition["question"],
                "decision": "fire when %s > threshold" % definition["column"],
                "calibration": "threshold maximising ANY_ATTACK F1 on the "
                               "fold's train partition only",
                "reported_axis": "ANY_ATTACK; the multiclass metrics are the "
                                 "projection described in run_rule_baseline.py "
                                 "and are floored by construction",
                "per_class_curves": "Only the designated attack class carries "
                                    "the score. The other attack columns are "
                                    "exactly 0, so a per-class curve over them "
                                    "is the prevalence floor of an all-zero "
                                    "score and is not a result - read this "
                                    "run's ANY_ATTACK row, and its designated "
                                    "class's row only as that one class.",
            },
            "feature_set": "rule",
            "feature_groups_dropped": [],
            "features_dropped": [],
            "discard_columns": [],
            "features_in_no_group": [],
            "seed": args.seed,
            "group_column": args.group_column,
            "target_column": args.target_column,
            "balance": "none",
            "smote_oversample_factor": None,
            "smote_max_target": None,
            "sample_cap_per_group_class": None,
            "max_train_rows_per_fold": None,
            "n_jobs": 1,
            "scores_file": SCORES_FILENAME if args.save_scores else None,
            "rows_used": rows_used,
            "classes": classes,
            "features": [definition["column"]],
            "fold_metrics": metrics,
        }
        with open(os.path.join(args.out_dir, "grouped_validation_report.json"),
                  "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        os.replace(predictions_tmp, predictions_path)
        if scores_tmp is not None:
            os.replace(scores_tmp, scores_path)
    except (OSError, json.JSONDecodeError, GroupedRunError, RuleBaselineError,
            ValueError) as exc:
        for stale in (predictions_tmp, scores_tmp):
            try:
                if stale and os.path.exists(stale):
                    os.remove(stale)
            except OSError:
                pass
        print("RULE BASELINE FAILED\n%s" % exc, file=sys.stderr)
        return 1
    thresholds = ", ".join("%s=%.6g" % (fold["split_id"], fold["rule"]["threshold"])
                           for fold in metrics)
    recalls = [fold["any_attack"]["recall"] for fold in metrics]
    print("Rule %s over %d folds on %d rows." % (args.rule, len(metrics), rows_used))
    print("  thresholds: %s" % thresholds)
    print("  ANY_ATTACK recall per fold: %s"
          % ", ".join("%.4f" % value for value in recalls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
