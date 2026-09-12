"""Reconcile grouped-validation predictions before any statistical test.

Checklist items E.4 and E.5.  E.5 asks, literally, to check the class sums and
the number of paired predictions *before* writing any statistical test; E.4
asks that every reported metric say whether it is macro, weighted or per
class.  This script does both over one or more finished
`run_grouped_validation.py` runs, and exits non-zero if any reconciliation
fails.

It is deliberately decoupled from the runner, the same way
`check_no_leakage.py` is decoupled from `generate_grouped_splits.py`: it
re-reads the persisted `grouped_predictions.csv` and recomputes every number
from a confusion matrix built with plain `numpy.bincount`, never calling the
`sklearn` helpers the runner itself used.  A metric bug in the runner
therefore cannot pass its own audit.

What it checks, per run:

  - **row coverage** - every `row_index` appears exactly once across folds
    (the row-level counterpart to `check_no_leakage.py`'s group-level
    guarantee), and, for a `full_grouped_run`, covers the whole dataset
    0..N-1 with no gaps;
  - **count reconciliation** - predictions per fold == that fold's
    `test_rows` == the sum of its per-class `support`; summed over folds ==
    `rows_used`;
  - **class sums** - per-class `y_true` totals from the predictions match the
    supports recorded in the report and, when the dataset is available and
    hash-verified, the dataset's own class counts;
  - **metrics** - accuracy plus macro *and* weighted averages plus per-class
    precision/recall/F1, each explicitly labelled, recomputed from the
    predictions and cross-checked against what the report recorded.

And, across runs:

  - **paired predictions** - whether two runs predicted the exact same rows
    with the same ground truth (a precondition for any paired test), and the
    2x2 agreement table (both correct / only A / only B / neither) whose
    off-diagonal cells are what a McNemar-style test would consume.

Usage
-----
    python experiments/revision_2026/check_prediction_integrity.py \
        --run results/grouped-validation-full \
        --run results/grouped-validation-full-downsample \
        --run results/grouped-validation-full-smote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

REPORT_NAME = "grouped_validation_report.json"
PREDICTIONS_NAME = "grouped_predictions.csv"
REQUIRED_COLUMNS = ["split_id", "row_index", "y_true", "y_pred"]

# Recomputed-vs-recorded metrics must agree to floating-point noise. They are
# the same definitions over the same rows, so anything above this is a real
# disagreement, not accumulation error.
METRIC_TOLERANCE = 1e-9


class PredictionIntegrityError(ValueError):
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


def _json_default(value):
    """Last-resort coercion for numpy scalars reaching `json.dump`."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("%r is not JSON serializable" % type(value).__name__)


# --------------------------------------------------------------------------
# Metrics - recomputed from a bincount confusion matrix, not from sklearn
# --------------------------------------------------------------------------

def confusion_counts(y_true, y_pred, n_classes):
    """`n_classes` x `n_classes` counts, rows = true, columns = predicted."""
    import numpy as np
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _safe_divide(numerator, denominator):
    import numpy as np
    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def metrics_from_confusion(matrix, class_names):
    """Accuracy, per-class P/R/F1, and both averaging schemes, each named.

    Mirrors `sklearn.metrics.classification_report(..., zero_division=0)`:
    macro averages over *every* class including zero-support ones, weighted
    averages the same per-class values by support.
    """
    import numpy as np

    true_positives = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    total = float(matrix.sum())

    precision = _safe_divide(true_positives, predicted)
    recall = _safe_divide(true_positives, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    weights = support if support.sum() > 0 else np.ones_like(support)
    return {
        # Overall/micro figure: the fraction of all rows predicted correctly.
        # Cast out of numpy here, not at the call sites: these values are
        # written to JSON, and a stray np.float64 only fails at dump time.
        "accuracy": float(true_positives.sum() / total) if total else 0.0,
        "rows": int(total),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1-score": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "averages": {
            # Unweighted mean over classes - rare attack classes count as
            # much as `normal`.
            "macro": {
                "precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1-score": float(f1.mean()),
            },
            # Support-weighted mean of the same per-class values - tracks
            # whatever the majority class does.
            "weighted": {
                "precision": float(np.average(precision, weights=weights)),
                "recall": float(np.average(recall, weights=weights)),
                "f1-score": float(np.average(f1, weights=weights)),
            },
        },
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_prediction_arrays(path, classes, fold_ids, chunk_rows=2_000_000):
    """Stream the predictions CSV into compact arrays sorted by `row_index`.

    Read in chunks and stored as integer codes (never as Python strings):
    a full run's predictions file is one row per dataset row, so holding the
    raw CSV columns in memory would cost more than the model training did.
    `split_group` is not read at all - `check_no_leakage.py` already owns the
    group-level invariant; this script only needs rows, folds and labels.
    """
    import numpy as np
    import pandas as pd

    class_to_code = {name: index for index, name in enumerate(classes)}
    fold_to_code = {name: index for index, name in enumerate(fold_ids)}

    row_parts, fold_parts, true_parts, pred_parts = [], [], [], []
    unknown_classes, unknown_folds = set(), set()

    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS,
                             chunksize=chunk_rows, encoding="utf-8"):
        missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
        if missing:
            raise PredictionIntegrityError(
                "predictions CSV is missing columns: %s" % sorted(missing))
        true_codes = chunk["y_true"].map(class_to_code)
        pred_codes = chunk["y_pred"].map(class_to_code)
        fold_codes = chunk["split_id"].map(fold_to_code)
        if true_codes.isna().any():
            unknown_classes |= set(chunk.loc[true_codes.isna(), "y_true"].unique())
        if pred_codes.isna().any():
            unknown_classes |= set(chunk.loc[pred_codes.isna(), "y_pred"].unique())
        if fold_codes.isna().any():
            unknown_folds |= set(chunk.loc[fold_codes.isna(), "split_id"].unique())
        if unknown_classes or unknown_folds:
            continue
        row_parts.append(chunk["row_index"].to_numpy(dtype=np.int64))
        fold_parts.append(fold_codes.to_numpy(dtype=np.int16))
        true_parts.append(true_codes.to_numpy(dtype=np.int16))
        pred_parts.append(pred_codes.to_numpy(dtype=np.int16))

    if unknown_classes:
        raise PredictionIntegrityError(
            "predictions contain labels absent from the report's class list: %s"
            % sorted(str(c) for c in unknown_classes))
    if unknown_folds:
        raise PredictionIntegrityError(
            "predictions contain split_ids absent from the report's folds: %s"
            % sorted(str(f) for f in unknown_folds))
    if not row_parts:
        raise PredictionIntegrityError("predictions CSV has no rows")

    row_index = np.concatenate(row_parts)
    order = np.argsort(row_index, kind="stable")
    return {
        "row_index": row_index[order],
        "fold": np.concatenate(fold_parts)[order],
        "y_true": np.concatenate(true_parts)[order],
        "y_pred": np.concatenate(pred_parts)[order],
    }


def dataset_class_counts(dataset_path, expected_sha256, target_column):
    """Per-class row counts read straight from the (hash-verified) dataset.

    Parquet is read one row group at a time - the same chunking the rest of
    this pipeline uses - so a 20M-row dataset costs one row group of memory,
    not the whole column.
    """
    digest = sha256_file(dataset_path)
    if digest != expected_sha256:
        raise PredictionIntegrityError(
            "dataset hash differs from the one recorded in the validation report - "
            "point --dataset at the file the run actually used.")

    counts = Counter()
    lowered = dataset_path.lower()
    if lowered.endswith((".parquet", ".pq")):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(dataset_path)
        for index in range(parquet_file.num_row_groups):
            column = parquet_file.read_row_group(index, columns=[target_column]).column(target_column)
            value_counts = pc.value_counts(column.combine_chunks())
            values = value_counts.field("values").to_pylist()
            occurrences = value_counts.field("counts").to_pylist()
            for value, occurrence in zip(values, occurrences):
                counts[value] += occurrence
    elif lowered.endswith(".csv"):
        import pandas as pd
        for chunk in pd.read_csv(dataset_path, usecols=[target_column],
                                 chunksize=1_000_000, encoding="utf-8"):
            counts.update(Counter(chunk[target_column].tolist()))
    else:
        raise PredictionIntegrityError("dataset must be .csv or .parquet")
    return counts


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------

def check_row_coverage(arrays, report_meta):
    """Every row predicted exactly once; a full run covers the whole dataset."""
    import numpy as np

    row_index = arrays["row_index"]
    duplicates = int(np.count_nonzero(np.diff(row_index) == 0))
    checks = [{
        "check": "each row_index predicted at most once",
        "expected": "0 duplicates",
        "observed": "%d duplicates" % duplicates,
        "passed": duplicates == 0,
    }]

    rows_used = report_meta.get("rows_used")
    if rows_used is not None:
        checks.append({
            "check": "prediction rows == report rows_used",
            "expected": "%d" % rows_used,
            "observed": "%d" % len(row_index),
            "passed": len(row_index) == rows_used,
        })

    if report_meta.get("status") == "full_grouped_run" and rows_used:
        contiguous = (
            duplicates == 0
            and len(row_index) == rows_used
            and int(row_index[0]) == 0
            and int(row_index[-1]) == rows_used - 1
        )
        checks.append({
            "check": "full run covers dataset rows 0..N-1 with no gaps",
            "expected": "0..%d" % (rows_used - 1),
            "observed": "%d..%d over %d rows" % (int(row_index[0]), int(row_index[-1]), len(row_index)),
            "passed": contiguous,
        })
    return checks


def check_fold_counts(arrays, report_meta, fold_ids):
    """Per-fold prediction counts against `test_rows` and per-class support."""
    import numpy as np

    fold_metrics = report_meta.get("fold_metrics", [])
    counts = np.bincount(arrays["fold"], minlength=len(fold_ids))
    checks = []
    for index, fold in enumerate(fold_metrics):
        predicted_rows = int(counts[index])
        test_rows = int(fold.get("test_rows", -1))
        support_sum = int(sum(entry.get("support", 0) for entry in fold.get("per_class", {}).values()))
        checks.append({
            "check": "%s: predictions == test_rows == sum(per-class support)" % fold_ids[index],
            "expected": "%d == %d" % (test_rows, support_sum),
            "observed": "%d predictions" % predicted_rows,
            "passed": predicted_rows == test_rows == support_sum,
        })
    return checks


def check_class_sums(arrays, report_meta, classes, dataset_counts=None):
    """`y_true` totals from predictions vs. the report, and vs. the dataset."""
    import numpy as np

    observed = np.bincount(arrays["y_true"], minlength=len(classes))
    recorded = Counter()
    for fold in report_meta.get("fold_metrics", []):
        for name, entry in fold.get("per_class", {}).items():
            recorded[name] += int(entry.get("support", 0))

    checks = []
    for index, name in enumerate(classes):
        checks.append({
            "check": "class sum (predictions vs. report support): %s" % name,
            "expected": "%d" % recorded.get(name, 0),
            "observed": "%d" % int(observed[index]),
            "passed": int(observed[index]) == recorded.get(name, 0),
        })

    if dataset_counts is not None:
        full_run = report_meta.get("status") == "full_grouped_run"
        for index, name in enumerate(classes):
            in_dataset = int(dataset_counts.get(name, 0))
            in_predictions = int(observed[index])
            checks.append({
                "check": "class sum (predictions vs. dataset): %s" % name,
                "expected": "%d in dataset" % in_dataset,
                "observed": "%d predicted" % in_predictions,
                # A capped smoke run only samples rows, so equality is not
                # expected - only that it never claims more rows than exist.
                "passed": in_predictions == in_dataset if full_run else in_predictions <= in_dataset,
            })
    return checks


def check_recorded_metrics(arrays, report_meta, classes, fold_ids, tolerance=METRIC_TOLERANCE):
    """Recomputed per-fold accuracy/macro-F1 against what the run recorded."""
    import numpy as np

    checks = []
    recomputed = []
    for index, fold in enumerate(report_meta.get("fold_metrics", [])):
        mask = arrays["fold"] == index
        matrix = confusion_counts(arrays["y_true"][mask], arrays["y_pred"][mask], len(classes))
        result = metrics_from_confusion(matrix, classes)
        recomputed.append({"split_id": fold_ids[index], **result})
        for key, recorded_value in (
            ("accuracy", fold.get("accuracy")),
            ("macro_f1", fold.get("macro_f1")),
            ("weighted_f1", fold.get("weighted_f1")),
        ):
            if recorded_value is None:
                # Runs produced before checklist E.4 added `weighted_f1` to
                # the report simply do not carry it. That is nothing to fail
                # on - recomputing it here is exactly the point.
                continue
            fresh = (result["accuracy"] if key == "accuracy"
                     else result["averages"]["macro" if key == "macro_f1" else "weighted"]["f1-score"])
            checks.append({
                "check": "%s: recorded %s matches recomputation" % (fold_ids[index], key),
                "expected": "%.12f" % recorded_value,
                "observed": "%.12f" % fresh,
                "passed": bool(abs(fresh - recorded_value) <= tolerance),
            })
    return checks, recomputed


def pooled_metrics(arrays, classes):
    matrix = confusion_counts(arrays["y_true"], arrays["y_pred"], len(classes))
    return metrics_from_confusion(matrix, classes)


# --------------------------------------------------------------------------
# Pairing across runs
# --------------------------------------------------------------------------

def pair_runs(first, second):
    """Are two runs' predictions pairable, and how do they disagree?

    Pairable means: exactly the same rows, in the same order, with the same
    ground truth.  Without that, any paired statistical test (McNemar and
    friends) is comparing different populations.
    """
    import numpy as np

    same_rows = np.array_equal(first["arrays"]["row_index"], second["arrays"]["row_index"])
    same_truth = same_rows and np.array_equal(first["arrays"]["y_true"], second["arrays"]["y_true"])
    if not (same_rows and same_truth):
        return {
            "paired": False,
            "same_rows": bool(same_rows),
            "same_y_true": bool(same_truth),
            "n_paired": 0,
        }

    first_correct = first["arrays"]["y_pred"] == first["arrays"]["y_true"]
    second_correct = second["arrays"]["y_pred"] == second["arrays"]["y_true"]
    both = int(np.count_nonzero(first_correct & second_correct))
    only_first = int(np.count_nonzero(first_correct & ~second_correct))
    only_second = int(np.count_nonzero(~first_correct & second_correct))
    neither = int(np.count_nonzero(~first_correct & ~second_correct))
    return {
        "paired": True,
        "same_rows": True,
        "same_y_true": True,
        "n_paired": int(len(first_correct)),
        "both_correct": both,
        "only_first_correct": only_first,
        "only_second_correct": only_second,
        "neither_correct": neither,
        # McNemar consumes exactly the two discordant cells.
        "discordant": only_first + only_second,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def verdict_mark(passed):
    return "pass" if passed else "**FAIL**"


def build_report(runs, pairings, generated=None):
    generated = generated or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)",
        "",
        f"- Generated: {generated}",
        "- Recomputed independently of `run_grouped_validation.py`: every number below",
        "  comes from a `numpy.bincount` confusion matrix over the persisted",
        "  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.",
        "",
        "## 1. Runs audited",
        "",
        "| run | status | model | balance | folds | rows |",
        "|---|---|---|---|---:|---:|",
    ]
    for run in runs:
        meta = run["report"]
        lines.append(
            "| `%s` | %s | %s | %s | %d | %s |" % (
                run["label"], meta.get("status", "?"), meta.get("model", "?"),
                meta.get("balance", "none"), len(run["fold_ids"]),
                f"{run['pooled']['rows']:,}",
            )
        )
    lines.append("")

    lines += [
        "## 2. Count reconciliation",
        "",
        "Checklist E.5: class sums and paired prediction counts, checked before any",
        "statistical test is written.",
        "",
    ]
    for run in runs:
        failures = [c for c in run["checks"] if not c["passed"]]
        lines += [
            "### `%s`" % run["label"],
            "",
            "%d checks, %d failed." % (len(run["checks"]), len(failures)),
            "",
        ]
        rows = failures if failures else run["checks"]
        if not failures:
            lines.append("<details><summary>All checks passed - expand for detail</summary>")
            lines.append("")
        lines += ["| check | expected | observed | result |", "|---|---|---|---|"]
        for check in rows:
            lines.append("| %s | %s | %s | %s |" % (
                check["check"], check["expected"], check["observed"], verdict_mark(check["passed"])))
        lines.append("")
        if not failures:
            lines.append("</details>")
            lines.append("")

    lines += [
        "## 3. Metrics, explicitly labelled",
        "",
        "Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages",
        "the per-class values without weights, so each of the four rare attack",
        "classes counts as much as `normal`; `weighted` averages the same per-class",
        "values by support, so it tracks the majority class. Pooled over all folds,",
        "on the original (never rebalanced) test distribution.",
        "",
        "| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        pooled = run["pooled"]
        macro = pooled["averages"]["macro"]
        weighted = pooled["averages"]["weighted"]
        lines.append("| `%s` | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            run["label"], pooled["accuracy"],
            macro["precision"], macro["recall"], macro["f1-score"],
            weighted["precision"], weighted["recall"], weighted["f1-score"],
        ))
    lines.append("")

    lines += [
        "### Per-class (pooled over folds)",
        "",
    ]
    for run in runs:
        lines += [
            "#### `%s`" % run["label"],
            "",
            "| class | precision | recall | f1 | support |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, entry in run["pooled"]["per_class"].items():
            lines.append("| %s | %.4f | %.4f | %.4f | %s |" % (
                name, entry["precision"], entry["recall"], entry["f1-score"],
                f"{entry['support']:,}"))
        lines.append("")

    lines += [
        "## 4. Paired predictions across runs",
        "",
        "Two runs are pairable only if they predicted exactly the same rows with the",
        "same ground truth. `discordant` is the number of rows where exactly one of",
        "the two got it right - the only cells a McNemar-style paired test consumes.",
        "",
        "| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pairing in pairings:
        info = pairing["result"]
        if info["paired"]:
            lines.append("| `%s` | `%s` | yes | %s | %s | %s | %s | %s | %s |" % (
                pairing["first"], pairing["second"], f"{info['n_paired']:,}",
                f"{info['both_correct']:,}", f"{info['only_first_correct']:,}",
                f"{info['only_second_correct']:,}", f"{info['neither_correct']:,}",
                f"{info['discordant']:,}"))
        else:
            lines.append("| `%s` | `%s` | **NO** (same rows: %s, same y_true: %s) | 0 | - | - | - | - | - |" % (
                pairing["first"], pairing["second"],
                info["same_rows"], info["same_y_true"]))
    if not pairings:
        lines.append("| - | - | n/a (single run audited) | - | - | - | - | - | - |")
    lines.append("")
    return lines


def write_report(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="DIR",
                        help="Directory holding %s and %s. Repeat to audit and pair "
                             "several runs." % (REPORT_NAME, PREDICTIONS_NAME))
    parser.add_argument("--dataset", default=None,
                        help="Override the dataset path recorded in each run's report. "
                             "When the dataset is readable, class sums are also "
                             "reconciled against it (hash-verified).")
    parser.add_argument("--skip-dataset-check", action="store_true",
                        help="Do not read the dataset at all; reconcile only predictions "
                             "against the run reports.")
    parser.add_argument("--out", default=None,
                        help="Output Markdown path (default: prediction_integrity.md "
                             "next to this script).")
    parser.add_argument("--json-out", default=None,
                        help="Optional machine-readable copy of the audit.")
    return parser.parse_args(argv)


def audit_run(directory, dataset_override, skip_dataset_check):
    report_path = os.path.join(directory, REPORT_NAME)
    predictions_path = os.path.join(directory, PREDICTIONS_NAME)
    for path in (report_path, predictions_path):
        if not os.path.exists(path):
            raise PredictionIntegrityError("missing %s" % path)

    report_meta = load_json(report_path)
    classes = report_meta.get("classes")
    if not classes:
        raise PredictionIntegrityError("%s has no class list" % report_path)
    fold_ids = [fold["split_id"] for fold in report_meta.get("fold_metrics", [])]
    if not fold_ids:
        raise PredictionIntegrityError("%s has no fold metrics" % report_path)

    arrays = load_prediction_arrays(predictions_path, classes, fold_ids)

    dataset_counts = None
    dataset_path = dataset_override or report_meta.get("dataset")
    if not skip_dataset_check and dataset_path and os.path.exists(dataset_path):
        dataset_counts = dataset_class_counts(
            dataset_path, report_meta.get("dataset_sha256"),
            report_meta.get("target_column", "class"),
        )

    checks = []
    checks += check_row_coverage(arrays, report_meta)
    checks += check_fold_counts(arrays, report_meta, fold_ids)
    checks += check_class_sums(arrays, report_meta, classes, dataset_counts)
    metric_checks, per_fold = check_recorded_metrics(arrays, report_meta, classes, fold_ids)
    checks += metric_checks

    return {
        "label": os.path.basename(os.path.normpath(directory)),
        "directory": directory,
        "report": report_meta,
        "classes": classes,
        "fold_ids": fold_ids,
        "arrays": arrays,
        "checks": checks,
        "per_fold": per_fold,
        "pooled": pooled_metrics(arrays, classes),
        "dataset_checked": dataset_counts is not None,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        runs = [audit_run(directory, args.dataset, args.skip_dataset_check)
                for directory in args.run]

        pairings = []
        for index in range(1, len(runs)):
            pairings.append({
                "first": runs[0]["label"],
                "second": runs[index]["label"],
                "result": pair_runs(runs[0], runs[index]),
            })

        out_path = args.out or os.path.join(HERE, "prediction_integrity.md")
        write_report(out_path, build_report(runs, pairings))

        if args.json_out:
            payload = {
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "runs": [{
                    "label": run["label"],
                    "directory": run["directory"],
                    "status": run["report"].get("status"),
                    "balance": run["report"].get("balance", "none"),
                    "dataset_checked": run["dataset_checked"],
                    "checks": run["checks"],
                    "pooled": run["pooled"],
                    "per_fold": run["per_fold"],
                } for run in runs],
                "pairings": pairings,
            }
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                # `default` is a safety net, not the plan: every value above is
                # cast at its source. Without it a single stray numpy scalar
                # would kill an audit that had already done all its work.
                json.dump(payload, fh, indent=2, default=_json_default)
                fh.write("\n")

        failed = [(run["label"], check) for run in runs
                  for check in run["checks"] if not check["passed"]]
        unpairable = [p for p in pairings if not p["result"]["paired"]]
    except (OSError, json.JSONDecodeError, PredictionIntegrityError,
            ValueError, KeyError, TypeError) as exc:
        print("PREDICTION INTEGRITY CHECK FAILED\n%s" % exc, file=sys.stderr)
        return 1

    print("Prediction integrity audit written: %s" % out_path)
    for run in runs:
        print("  %s: %d checks, %d failed%s" % (
            run["label"], len(run["checks"]),
            sum(1 for c in run["checks"] if not c["passed"]),
            "" if run["dataset_checked"] else " (dataset check skipped)"))
    for pairing in pairings:
        info = pairing["result"]
        print("  paired %s vs %s: %s%s" % (
            pairing["first"], pairing["second"],
            "yes" if info["paired"] else "NO",
            (", %d rows, %d discordant" % (info["n_paired"], info["discordant"]))
            if info["paired"] else ""))
    if failed:
        print("\n%d RECONCILIATION FAILURE(S)" % len(failed), file=sys.stderr)
        for label, check in failed[:20]:
            print("  [%s] %s: expected %s, observed %s" % (
                label, check["check"], check["expected"], check["observed"]), file=sys.stderr)
        return 1
    if unpairable:
        print("\n%d RUN PAIR(S) NOT PAIRABLE - no paired statistical test may be "
              "written over them" % len(unpairable), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
