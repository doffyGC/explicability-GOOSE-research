"""Checklist D.5: every card-D run in one table, on one pool, per class.

By the time D.4 closed, the evidence for card D was spread across seven
files - `pr_curves_d1.md`, `pr_curves_d1_split.md`, `pr_curves_d2.md`,
`pr_curves_d2_rules.md`, `pr_curves_d3_v2.md`, `pr_curves_d4.md` and four
`run_bootstrap.*` reports - none of which shows more than one card at a time.
A reviewer asking "what did the ablation cost relative to the model
comparison?" has to reconstruct the answer from three documents that were
never checked against each other. D.5 is that table, and the checking is the
point rather than the layout.

What it reads, and what it therefore costs
------------------------------------------
**Only `grouped_validation_report.json`, one per run.** No dataset, no
predictions, no refit - a full 24-run consolidation runs in under a second,
which is what `ablations_baselines.md` §4 budgeted for this item ("~0"
compute). That is possible because a fold's `per_class` block records
`recall`, `precision` and `support`, and the five folds partition the
evaluated rows exactly once. So for each class:

    TP = recall x support            FN = support - TP
    FP = TP / precision - TP         (precision > 0)

summed over folds, gives the pooled counts, and the pooled metrics follow.
This is a reconstruction, so it is verified rather than trusted, in three
independent ways:

1. `TP = recall x support` must land on an integer (it is a count). The
   observed deviation across all 24 runs is 0.0000; anything above
   `COUNT_TOLERANCE` is fatal, because a non-integral TP means the report's
   per-fold block is not what this arithmetic assumes.
2. Where `--confusion` builds the real matrix from the predictions CSV, the
   reconstructed per-class metrics are checked against
   `check_prediction_integrity.metrics_from_confusion` on that matrix, and a
   disagreement above `METRIC_TOLERANCE` is fatal.
3. `test_cross_run_report.py` pins the reconstruction against the same
   script's implementation on synthetic matrices, and against the published
   `run_bootstrap.champion_v2.md` point estimates, which were computed the
   expensive way - from 11M rows of predictions. They agree to four decimals
   on every class and on macro F1 (0.7310).

Why pooled rather than the mean over folds
------------------------------------------
`bootstrap_run_intervals.py` is explicit that the five folds are not five
independent estimates - they are one partition of the runs. Averaging their
per-class F1 weights a fold holding one `SAG.DB` run the same as a fold
holding four. Pooling counts first and computing the metric once weights
every evaluated row equally, and it is the same quantity the run-level
bootstrap reports as its point estimate. The mean over folds is not printed
anywhere here, so it cannot be mistaken for it.

What this does **not** do
-------------------------
It computes no interval of its own. A cross-run table with 24 rows and no
uncertainty invites exactly the reading `validation_protocol.md` forbids -
comparing two configurations by their marginal numbers. So every row that has
a run-level interval or a paired difference elsewhere carries a pointer to it,
and the document says in its own text that the table ranks nothing. The
paired comparisons stay in `grouped_pr_curves.py` and
`bootstrap_run_intervals.py`, which own the resampling.

Rule runs are in the table, and their per-class cells are not results
--------------------------------------------------------------------
`ablations_baselines.md` §16: a rule has one score and cannot name an attack
family, so `run_rule_baseline.py` emits a posterior carrying it on one
designated class and exactly zero on the other three. Those three classes
come back at the prevalence floor **by construction**, and a per-class table
that printed them next to a learned run's would be inviting a comparison that
means nothing. They are rendered as `n/a` with the reason attached, and a
rule's row carries only the `ANY_ATTACK` axis it is entitled to.

Usage
-----
    python experiments/revision_2026/cross_run_report.py \\
      --runs-glob 'results/*' \\
      --pr-curves experiments/revision_2026/pr_curves_d3_v2.json \\
      --pr-curves experiments/revision_2026/pr_curves_d1.json \\
      --pr-curves experiments/revision_2026/pr_curves_d4.json \\
      --confusion results/v2-xgboost-none \\
      --out experiments/revision_2026/cross_run_comparison.md
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import os
import sys
from datetime import datetime, timezone

from benign_confusion_report import (
    CANONICAL_CLASS_ORDER,
    confusion_matrix_lines,
    display_name,
)
from check_prediction_integrity import metrics_from_confusion

HERE = os.path.dirname(os.path.abspath(__file__))

REPORT_NAME = "grouped_validation_report.json"
PREDICTIONS_NAME = "grouped_predictions.csv"

# `TP = recall * support` is a count, so it must be integral. Floats in the
# report carry ~15 significant digits and supports here run to 10.5M, which
# leaves room to spare; 1e-6 is tight enough to catch a report whose per-fold
# block was produced by different arithmetic, and loose enough never to fire
# on representation error.
COUNT_TOLERANCE = 1e-6

# How far the reconstruction may sit from the same metric computed on a real
# confusion matrix before the run is refused.
METRIC_TOLERANCE = 1e-9

CHUNK_ROWS = 2_000_000


class CrossRunError(ValueError):
    pass


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Which card a run belongs to
# --------------------------------------------------------------------------

def classify_run(report):
    """Derived from the report's own contents, never from the directory name.

    A run renamed or re-run into another directory must land in the same
    section, because the section is a claim about how the run was produced.
    """
    if "rule" in report:
        return "D.2", "rule baseline"
    if "tuned" in report:
        return "D.4", "nested tuning"
    feature_set = report.get("feature_set")
    if feature_set not in (None, "all"):
        return "D.1", "feature-group ablation"
    return "D.3", "model family comparison"


def scenario_of(report):
    parts = [report.get("balance") or "none"]
    cap = report.get("max_train_rows_per_fold")
    if cap:
        parts.append("cap %s" % format(int(cap), ","))
    return ", ".join(parts)


def variant_of(report):
    """The one axis that makes this run different from the reference."""
    if "rule" in report:
        return "rule: `%s`" % report["rule"]["name"]
    if "tuned" in report:
        return "grid: `%s`" % report["tuned"]
    feature_set = report.get("feature_set")
    if feature_set not in (None, "all"):
        return "features: `%s`" % feature_set
    return "—"


# --------------------------------------------------------------------------
# Pooled metrics, reconstructed from the per-fold block
# --------------------------------------------------------------------------

def pooled_counts(report):
    """Sum TP / FP / FN / support per class over the folds.

    The folds partition the evaluated rows, so summing counts and computing
    the metric once is the same operation `bootstrap_run_intervals.py`
    performs on the real predictions - see the module docstring for why that
    is the right average and what verifies it.
    """
    classes = list(report["classes"])
    totals = {name: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support": 0.0}
              for name in classes}
    worst_residual = 0.0

    for fold in report["fold_metrics"]:
        per_class = fold.get("per_class")
        if per_class is None:
            raise CrossRunError(
                "%s has no per_class block - it predates the metric contract "
                "this table reads" % fold.get("split_id", "?"))
        for name in classes:
            entry = per_class.get(name)
            if entry is None:
                raise CrossRunError(
                    "%s does not report class %r, but the run declares it"
                    % (fold.get("split_id", "?"), name))
            support = float(entry["support"])
            recall = float(entry["recall"])
            precision = float(entry["precision"])

            true_positives = recall * support
            worst_residual = max(
                worst_residual, abs(true_positives - round(true_positives)))
            # A class with no predicted row has precision 0 by the report's
            # own zero_division convention; it also has TP 0, so there is no
            # false-positive mass to recover and 0 is exact rather than a
            # fallback.
            false_positives = (true_positives / precision - true_positives
                               if precision > 0 else 0.0)

            bucket = totals[name]
            bucket["tp"] += true_positives
            bucket["fn"] += support - true_positives
            bucket["fp"] += false_positives
            bucket["support"] += support

    return classes, totals, worst_residual


def metrics_from_counts(classes, totals):
    """Per-class precision/recall/F1 and both averaging schemes.

    Mirrors `check_prediction_integrity.metrics_from_confusion`, which owns
    the definition; `test_cross_run_report.py` pins the two against each
    other rather than leaving "mirrors" as a comment. Accuracy is *not*
    recoverable from per-class counts alone (the off-diagonal mass is not
    attributable), so it is read from the report instead of guessed at.
    """
    per_class = {}
    for name in classes:
        bucket = totals[name]
        true_positives = bucket["tp"]
        predicted = true_positives + bucket["fp"]
        support = bucket["support"]
        precision = true_positives / predicted if predicted > 0 else 0.0
        recall = true_positives / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(round(support)),
        }

    supports = [per_class[name]["support"] for name in classes]
    total_support = sum(supports)

    def average(key, weights=None):
        values = [per_class[name][key] for name in classes]
        if weights is None:
            return sum(values) / len(values) if values else 0.0
        if total_support == 0:
            return sum(values) / len(values) if values else 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_support

    return {
        "per_class": per_class,
        "rows": total_support,
        "averages": {
            "macro": {key: average(key)
                      for key in ("precision", "recall", "f1-score")},
            "weighted": {key: average(key, supports)
                         for key in ("precision", "recall", "f1-score")},
        },
    }


def pooled_accuracy(report):
    """Rows predicted correctly over rows evaluated, summed over folds.

    Read from the per-fold accuracy and test_rows rather than reconstructed,
    because accuracy needs the diagonal of the *whole* matrix and per-class
    counts do not carry it.
    """
    correct = 0.0
    rows = 0.0
    for fold in report["fold_metrics"]:
        test_rows = float(fold["test_rows"])
        correct += float(fold["accuracy"]) * test_rows
        rows += test_rows
    return (correct / rows) if rows else 0.0


# --------------------------------------------------------------------------
# Loading runs, and refusing runs that cannot share a table
# --------------------------------------------------------------------------

def designated_attack_class(report):
    """Which attack class a rule run's single score was put on.

    Recorded per fold by `run_rule_baseline.py` (it is chosen from the fold's
    own train partition), so a rule that designated different classes in
    different folds would have a per-class row that is not about one class at
    all. That is refused rather than rendered.
    """
    if "rule" not in report:
        return None
    designated = {fold["rule"]["designated_attack_class"]
                  for fold in report["fold_metrics"] if "rule" in fold}
    if not designated:
        raise CrossRunError(
            "rule run records no designated attack class in its folds")
    if len(designated) > 1:
        raise CrossRunError(
            "rule run designates different attack classes across folds (%s), "
            "so none of its per-class rows is about a single class"
            % ", ".join(sorted(designated)))
    return designated.pop()


def load_run(directory):
    report_path = os.path.join(directory, REPORT_NAME)
    if not os.path.exists(report_path):
        raise CrossRunError("%s has no %s" % (directory, REPORT_NAME))
    report = load_json(report_path)
    classes, totals, residual = pooled_counts(report)
    if residual > COUNT_TOLERANCE:
        raise CrossRunError(
            "%s: recall x support is %.6g away from an integer, so the pooled "
            "counts this table is built from cannot be recovered from its "
            "per-fold block" % (directory, residual))
    metrics = metrics_from_counts(classes, totals)
    card, card_name = classify_run(report)
    return {
        "label": os.path.basename(os.path.normpath(directory)),
        "directory": directory,
        "report": report,
        "card": card,
        "card_name": card_name,
        "classes": classes,
        "metrics": metrics,
        "accuracy": pooled_accuracy(report),
        "count_residual": residual,
        "is_rule": "rule" in report,
        "designated": designated_attack_class(report),
        "n_folds": len(report["fold_metrics"]),
    }


def check_comparable(runs, allow_smoke):
    """Refuse a table whose rows were not produced against the same evidence.

    This is the whole risk of a consolidation script: a cross-run table looks
    equally authoritative whether or not its rows share a pool, and the pool
    *was* regenerated once during this revision (`data_card.md`). A mixed
    table would be the most convincing wrong artifact in the repository.
    """
    digests = {run["report"].get("dataset_sha256") for run in runs}
    if len(digests) > 1:
        raise CrossRunError(
            "runs span %d different datasets (%s) - they cannot share a table"
            % (len(digests), ", ".join(sorted(str(d)[:12] for d in digests))))

    protocols = {run["report"].get("protocol") for run in runs}
    if len(protocols) > 1:
        raise CrossRunError(
            "runs span %d split protocols (%s)"
            % (len(protocols), ", ".join(sorted(map(str, protocols)))))

    splits = {os.path.basename(str(run["report"].get("splits"))) for run in runs}
    if len(splits) > 1:
        raise CrossRunError(
            "runs consume %d different split files (%s) - the folds must be the "
            "same folds" % (len(splits), ", ".join(sorted(splits))))

    fold_counts = {run["n_folds"] for run in runs}
    if len(fold_counts) > 1:
        raise CrossRunError("runs span different fold counts: %s"
                            % sorted(fold_counts))

    smoke = [run["label"] for run in runs
             if run["report"].get("status") != "full_grouped_run"]
    if smoke and not allow_smoke:
        raise CrossRunError(
            "not a full grouped run: %s. A technical smoke is a wiring check, "
            "not a result; pass --allow-smoke to include it and it will be "
            "marked in every row." % ", ".join(sorted(smoke)))

    row_counts = {run["metrics"]["rows"] for run in runs}
    if len(row_counts) > 1 and not smoke:
        raise CrossRunError(
            "runs evaluated different row counts (%s) - same folds should mean "
            "the same evaluated rows"
            % ", ".join(format(n, ",") for n in sorted(row_counts)))


# --------------------------------------------------------------------------
# The ranking axis, merged in from the curve reports
# --------------------------------------------------------------------------

def load_average_precisions(paths, dataset_digest):
    """`ANY_ATTACK` AP per run label, from any number of curve reports.

    The argmax metrics this table is built on sit at whatever threshold the
    training prior implies (`ablations_baselines.md` §11), so a cross-run
    table carrying only macro F1 would rank configurations on the one axis
    the card says not to rank them on. Where a curve report covers a run, its
    AP travels into the table beside the F1.
    """
    found = {}
    for path in paths:
        payload = load_json(path)
        for run in payload.get("runs", []):
            digest = run.get("dataset_sha256")
            if digest and dataset_digest and digest != dataset_digest:
                # Silently skipping would put a number from the withdrawn
                # pool next to numbers from the current one.
                raise CrossRunError(
                    "%s carries run %r from dataset %s, but this table is "
                    "built on %s" % (path, run.get("label"), str(digest)[:12],
                                     str(dataset_digest)[:12]))
            target = (run.get("targets") or {}).get("ANY_ATTACK")
            if not target:
                continue
            label = run.get("label")
            entry = {
                "average_precision": target.get("average_precision"),
                "ci": target.get("average_precision_ci"),
                "source": os.path.basename(path),
            }
            previous = found.get(label)
            if previous and previous["average_precision"] != entry["average_precision"]:
                raise CrossRunError(
                    "run %r has two different ANY_ATTACK APs: %.6f in %s and "
                    "%.6f in %s" % (label, previous["average_precision"],
                                    previous["source"],
                                    entry["average_precision"], entry["source"]))
            found.setdefault(label, entry)
    return found


# --------------------------------------------------------------------------
# Confusion matrices for the runs the paper discusses
# --------------------------------------------------------------------------

def stream_confusion(path):
    """The real matrix, read in chunks rather than loaded whole.

    `benign_confusion_report.load_predictions` reads the CSV in one go, which
    is 722 MB and ~11M rows of object dtype here; this machine has already
    lost one overnight run to memory pressure, so the matrix is accumulated
    chunk by chunk and only the counts are kept.
    """
    import pandas as pd

    counts = {}
    reader = pd.read_csv(path, usecols=["y_true", "y_pred"], encoding="utf-8",
                         chunksize=CHUNK_ROWS)
    for chunk in reader:
        grouped = chunk.groupby(["y_true", "y_pred"]).size()
        for (true, pred), n in grouped.items():
            counts.setdefault(true, {})
            counts[true][pred] = counts[true].get(pred, 0) + int(n)

    present = sorted(
        set(counts) | {p for row in counts.values() for p in row},
        key=lambda c: (CANONICAL_CLASS_ORDER.index(c)
                       if c in CANONICAL_CLASS_ORDER else 99, c),
    )
    dense = {true: {pred: counts.get(true, {}).get(pred, 0) for pred in present}
             for true in present}
    return present, dense


def verify_against_matrix(run, present, dense):
    """The reconstruction, checked against the matrix it claims to summarise.

    Only possible for runs given to `--confusion`, which is why those runs
    are worth naming: each one turns the pooled arithmetic from an argument
    into a check.
    """
    import numpy as np

    matrix = np.array([[dense[true][pred] for pred in present]
                       for true in present], dtype=np.int64)
    truth = metrics_from_confusion(matrix, present)
    failures = []
    for name in present:
        reconstructed = run["metrics"]["per_class"].get(name)
        if reconstructed is None:
            failures.append("%s: class absent from the reconstruction" % name)
            continue
        for key in ("precision", "recall", "f1-score"):
            delta = abs(reconstructed[key] - truth["per_class"][name][key])
            if delta > METRIC_TOLERANCE:
                failures.append("%s %s: %.12f vs %.12f"
                                % (name, key, reconstructed[key],
                                   truth["per_class"][name][key]))
    macro_delta = abs(run["metrics"]["averages"]["macro"]["f1-score"]
                      - truth["averages"]["macro"]["f1-score"])
    if macro_delta > METRIC_TOLERANCE:
        failures.append("macro f1: %.12f vs %.12f"
                        % (run["metrics"]["averages"]["macro"]["f1-score"],
                           truth["averages"]["macro"]["f1-score"]))
    if failures:
        raise CrossRunError(
            "%s: pooled metrics reconstructed from the per-fold report "
            "disagree with the confusion matrix built from its predictions:\n  %s"
            % (run["label"], "\n  ".join(failures)))
    return truth


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

RULE_CELL = "n/a"


def fmt(value, places=4):
    return "—" if value is None else ("%.*f" % (places, value))


def fmt_ap(entry):
    if entry is None:
        return "—"
    ci = entry.get("ci")
    if not ci:
        return fmt(entry["average_precision"])
    return "%s [%s, %s]" % (fmt(entry["average_precision"]),
                            fmt(ci.get("lower")), fmt(ci.get("upper")))


def build_report(runs, average_precisions, confusions, generated=None):
    generated = generated or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")
    reference = runs[0]["report"]
    classes = runs[0]["classes"]
    attack_like = [c for c in classes if c not in ("normal", "benign_degradation")]
    ordered = sorted(runs, key=lambda r: (r["card"], r["label"]))

    lines = [
        "# Cross-run comparison (checklist D.5)",
        "",
        "- Generated: %s" % generated,
        "- Runs: **%d**, all on dataset `%s…`, protocol `%s`, %d folds from "
        "`%s`." % (len(runs), str(reference.get("dataset_sha256"))[:12],
                   reference.get("protocol"), runs[0]["n_folds"],
                   os.path.basename(str(reference.get("splits")))),
        "- Rows evaluated per run: **%s**" % format(runs[0]["metrics"]["rows"], ","),
        "",
        "Every metric here is **pooled over the folds** - counts summed first, "
        "metric computed once - not averaged across them. The folds partition "
        "the runs rather than replicate them, so a fold mean would weight a "
        "fold holding one `SAG.DB` run the same as one holding four "
        "(`bootstrap_run_intervals.py`). These are the same point estimates "
        "that script reports.",
        "",
        "**This table ranks nothing.** It carries no interval, and two rows "
        "differing by less than their unshown uncertainty are not two results. "
        "Every comparison this repository makes is paired over the same runs - "
        "`grouped_pr_curves.py` for the ranking axis, "
        "`bootstrap_run_intervals.py` for macro F1 - and those are the only "
        "places a difference between two rows may be read from.",
        "",
        "## Headline",
        "",
        "AP is on `ANY_ATTACK` and is the axis card D compares on (§11); macro "
        "F1 and accuracy sit at whatever threshold each run's training prior "
        "implies and are context, not verdict.",
        "",
        "| Card | Run | Model | Scenario | Variant | AP `ANY_ATTACK` [95% CI] | macro F1 | weighted F1 | accuracy |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for run in ordered:
        report = run["report"]
        averages = run["metrics"]["averages"]
        lines.append("| %s | `%s` | `%s` | %s | %s | %s | %s | %s | %s |" % (
            run["card"], run["label"], report.get("model"), scenario_of(report),
            variant_of(report), fmt_ap(average_precisions.get(run["label"])),
            fmt(averages["macro"]["f1-score"]),
            fmt(averages["weighted"]["f1-score"]), fmt(run["accuracy"])))

    lines += [
        "",
        "Cards: **D.1** feature-group ablation · **D.2** rule baseline · "
        "**D.3** model family comparison · **D.4** nested tuning. Each run's "
        "card is derived from its own report - the ablated feature set, the "
        "`rule` block, the `tuned` block - never from its directory name.",
        "",
        "## Per class",
        "",
        "Per-class values first, macro as the headline average, weighted and "
        "accuracy as context (`ablations_baselines.md` §5).",
        "",
    ]

    for name in classes:
        support = runs[0]["metrics"]["per_class"][name]["support"]
        lines += [
            "### `%s`" % display_name(name),
            "",
            "%s rows across all folds." % format(support, ","),
            "",
            "| Run | precision | recall | F1 |",
            "|---|---:|---:|---:|",
        ]
        for run in ordered:
            entry = run["metrics"]["per_class"][name]
            if run["is_rule"] and name in attack_like:
                if run["designated"] != name:
                    lines.append("| `%s` | %s | %s | %s |"
                                 % (run["label"], RULE_CELL, RULE_CELL, RULE_CELL))
                    continue
            lines.append("| `%s` | %s | %s | %s |" % (
                run["label"], fmt(entry["precision"]), fmt(entry["recall"]),
                fmt(entry["f1-score"])))
        lines.append("")

    if any(run["is_rule"] for run in runs):
        lines += [
            "`n/a` marks a cell that is **not a result**. A single-threshold "
            "rule has one score and cannot name an attack family, so "
            "`run_rule_baseline.py` puts that score on one designated class "
            "and exactly zero on the other three; those three would come back "
            "at the prevalence floor by construction and mean nothing next to "
            "a learned run's (`ablations_baselines.md` §16). Only a rule's "
            "`ANY_ATTACK` column is a result.",
            "",
        ]

    if confusions:
        lines += [
            "## Confusion matrices",
            "",
            "Built from each run's `%s`, over the full six-class vocabulary, "
            "so ideal `normal` and `benign_degradation` are never folded into "
            "one bucket (checklist C.3). Every matrix here also **verifies the "
            "table above**: the pooled metrics reconstructed from the per-fold "
            "report are checked against "
            "`check_prediction_integrity.metrics_from_confusion` on this "
            "matrix, and a disagreement beyond %g is fatal." % (
                PREDICTIONS_NAME, METRIC_TOLERANCE),
            "",
        ]
        for label, present, dense in confusions:
            lines += ["### `%s`" % label, ""]
            lines += confusion_matrix_lines(present, dense)
            lines.append("")

    lines += [
        "## Where the uncertainty lives",
        "",
        "| Question | The artifact that answers it |",
        "|---|---|",
        "| Does configuration A rank attacks better than B? | `grouped_pr_curves.py`, paired AP and recall at four alert budgets |",
        "| How much does a per-class number depend on which runs were generated? | `bootstrap_run_intervals.py`, resampling runs |",
        "| Do a run's predictions reconcile with its own report and posteriors? | `check_prediction_integrity.py` |",
        "| When a benign-degradation row is misclassified, what does it become? | `benign_confusion_report.py` |",
        "",
    ]
    return lines


def write_report(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines).rstrip("\n") + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[],
                        help="A grouped-validation run directory. Repeatable.")
    parser.add_argument("--runs-glob", action="append", default=[],
                        help="Glob of run directories, e.g. 'results/*'. "
                             "Directories without a report are skipped.")
    parser.add_argument("--pr-curves", action="append", default=[],
                        help="A grouped_pr_curves JSON whose ANY_ATTACK AP is "
                             "merged into the headline table. Repeatable; one "
                             "built on another dataset is fatal, not skipped.")
    parser.add_argument("--confusion", action="append", default=[],
                        help="Run directory to emit a full confusion matrix "
                             "for, and to verify the pooled reconstruction "
                             "against. Repeatable. Reads that run's "
                             "predictions CSV, so it is the only part of this "
                             "script that costs anything.")
    parser.add_argument("--allow-smoke", action="store_true",
                        help="Include technical-smoke runs, marked as such. "
                             "They are refused by default: a smoke is a wiring "
                             "check, not a result.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args(argv)


def resolve_directories(args):
    directories = list(args.run)
    for pattern in args.runs_glob:
        for path in sorted(globlib.glob(pattern)):
            if os.path.isdir(path) and os.path.exists(os.path.join(path, REPORT_NAME)):
                directories.append(path)
    seen = set()
    ordered = []
    for directory in directories:
        key = os.path.abspath(directory)
        if key not in seen:
            seen.add(key)
            ordered.append(directory)
    return ordered


def main(argv=None):
    args = parse_args(argv)
    try:
        directories = resolve_directories(args)
        if not directories:
            raise CrossRunError("no run directories given (--run / --runs-glob)")

        runs = [load_run(directory) for directory in directories]
        check_comparable(runs, args.allow_smoke)

        digest = runs[0]["report"].get("dataset_sha256")
        average_precisions = load_average_precisions(args.pr_curves, digest)

        by_label = {run["label"]: run for run in runs}
        confusions = []
        for directory in args.confusion:
            label = os.path.basename(os.path.normpath(directory))
            run = by_label.get(label)
            if run is None:
                raise CrossRunError(
                    "--confusion %s is not one of the runs in the table" % directory)
            predictions = os.path.join(run["directory"], PREDICTIONS_NAME)
            if not os.path.exists(predictions):
                raise CrossRunError("%s has no %s" % (directory, PREDICTIONS_NAME))
            present, dense = stream_confusion(predictions)
            verify_against_matrix(run, present, dense)
            confusions.append((label, present, dense))

        out_path = args.out or os.path.join(HERE, "cross_run_comparison.md")
        write_report(out_path, build_report(runs, average_precisions, confusions))

        if args.json_out:
            payload = {
                "generated": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"),
                "dataset_sha256": digest,
                "protocol": runs[0]["report"].get("protocol"),
                "splits": runs[0]["report"].get("splits"),
                "rows_evaluated": runs[0]["metrics"]["rows"],
                "verified_against_confusion_matrix": [c[0] for c in confusions],
                "runs": [{
                    "label": run["label"],
                    "directory": os.path.abspath(run["directory"]),
                    "card": run["card"],
                    "card_name": run["card_name"],
                    "model": run["report"].get("model"),
                    "balance": run["report"].get("balance"),
                    "feature_set": run["report"].get("feature_set"),
                    "tuned": run["report"].get("tuned"),
                    "rule": (run["report"].get("rule") or {}).get("name"),
                    "designated_attack_class": run["designated"],
                    "status": run["report"].get("status"),
                    "is_rule": run["is_rule"],
                    "accuracy": run["accuracy"],
                    "count_residual": run["count_residual"],
                    "average_precision": average_precisions.get(run["label"]),
                    "pooled": run["metrics"],
                } for run in sorted(runs, key=lambda r: (r["card"], r["label"]))],
            }
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
    except (OSError, json.JSONDecodeError, CrossRunError, ValueError,
            KeyError, TypeError) as exc:
        print("CROSS-RUN REPORT FAILED\n%s" % exc, file=sys.stderr)
        return 1

    print("Cross-run comparison written: %s" % out_path)
    print("  %d runs on dataset %s…, %s rows each"
          % (len(runs), str(digest)[:12], format(runs[0]["metrics"]["rows"], ",")))
    cards = {}
    for run in runs:
        cards.setdefault(run["card"], []).append(run["label"])
    for card in sorted(cards):
        print("  %s: %d run(s)" % (card, len(cards[card])))
    missing = [run["label"] for run in runs
               if run["label"] not in average_precisions]
    if missing:
        print("  no ANY_ATTACK AP for: %s" % ", ".join(sorted(missing)))
    if confusions:
        print("  reconstruction verified against the full matrix for: %s"
              % ", ".join(c[0] for c in confusions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
