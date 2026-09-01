"""Confusion analysis for card C: benign-degradation vs. attack vs. normal.

Checklist item C.4. `run_grouped_validation.py` reports standard multiclass
metrics but deliberately discards `impairment_mode` from the feature matrix
(it would leak the label) and never asks the one question card C exists to
answer: when a `benign_degradation` row is misclassified, does it get flipped
into an *attack* class it is specifically paired against (see
`benign_controls.md` SS3's "falsifies" column), or just confused with
`normal`? This script reads a grouped-validation run's predictions, rejoins
`impairment_mode` from the (hash-verified) prepared dataset, and reports:

  - a class x class confusion matrix over the full 6-class vocabulary
    (`normal`, `benign_degradation`, and the four SAG.*/FRG classes) - "6x6"
    rather than the baseline pipeline's 5, so Normal (ideal) and Benign
    degradation are never folded into one bucket (checklist C.3). Classes
    absent from the run (e.g. a benign-only smoke dataset) are reported as
    absent rather than silently zero-filled into a matrix that looks complete.
  - a per-mode breakdown of what each of the 7 impairment mechanisms actually
    gets predicted as;
  - false-positive rate (misclassified as an *attack* class - the specific
    harm card C targets) and alert burden (misclassified as anything other
    than `normal`, `benign_degradation` included) per mode, contrasted with
    ideal `normal` traffic and with the `normal` baseline messages captured
    inside a benign-impairment run (same run, undegraded messages -
    `impairment_mode` is run-level, so these are not the same population as
    ideal normal traffic from an attack run - see benign_controls.md SS6
    constraint 8's note that impairment_mode does not distinguish them at the
    row level).

This never re-derives predictions or retrains anything; it is a pure
read-and-summarise pass over an existing grouped-validation run, bound to
that run's dataset by SHA-256 the same way the rest of the section-B/C chain
is.

Usage
-----
    python experiments/revision_2026/benign_confusion_report.py \
        --report results/grouped-smoke/grouped_validation_report.json \
        --predictions results/grouped-smoke/grouped_predictions.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from add_experiment_metadata import BENIGN_DEGRADATION_LABEL, NORMAL_LABEL, VARIANT_OF_CLASS

HERE = os.path.dirname(os.path.abspath(__file__))

# The four raw ERENO class labels that are attacks - every VARIANT_OF_CLASS
# key except the two non-attack labels.
ATTACK_CLASSES = sorted(
    k for k in VARIANT_OF_CLASS if k not in (NORMAL_LABEL, BENIGN_DEGRADATION_LABEL)
)

# Canonical row/column order for the confusion matrix: normal and
# benign_degradation first (the axis checklist item C.3 cares about), then
# the four attack classes.
CANONICAL_CLASS_ORDER = [NORMAL_LABEL, BENIGN_DEGRADATION_LABEL] + ATTACK_CLASSES

NONE_MODE = "NONE"


class ConfusionReportError(ValueError):
    pass


def display_name(raw_class):
    """Short label for a raw `class` value.

    Deliberately not VARIANT_OF_CLASS itself: that dict maps both `normal`
    and `benign_degradation` to "none" (the *attack_variant* reading - see
    add_experiment_metadata.py), which would collapse the two classes this
    report exists to keep apart.
    """
    if raw_class in (NORMAL_LABEL, BENIGN_DEGRADATION_LABEL):
        return raw_class
    return VARIANT_OF_CLASS.get(raw_class, raw_class)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_predictions(path):
    import pandas as pd
    frame = pd.read_csv(path, encoding="utf-8")
    missing = {"row_index", "split_id", "split_group", "y_true", "y_pred"} - set(frame.columns)
    if missing:
        raise ConfusionReportError(f"predictions CSV is missing columns: {sorted(missing)}")
    return frame


def load_impairment_modes(dataset_path, expected_sha256, row_indices):
    """Rejoin `impairment_mode` for exactly the rows a predictions file cites.

    `row_index` is the positional label in the dataset `run_grouped_validation.py`
    was pointed at (that script never resets the index, and
    `prepare_grouped_dataset.py` writes its output with a clean 0..N-1 range,
    so position is a stable join key). Hash-verified against the run's own
    report so a stale or hand-edited dataset cannot silently be substituted.
    """
    import pandas as pd

    digest = sha256_file(dataset_path)
    if digest != expected_sha256:
        raise ConfusionReportError(
            "dataset hash differs from the one recorded in the validation report - "
            "re-run the grouped-validation step or point --dataset at the right file."
        )

    if dataset_path.lower().endswith(".parquet") or dataset_path.lower().endswith(".pq"):
        column = pd.read_parquet(dataset_path, columns=["impairment_mode"])["impairment_mode"]
    elif dataset_path.lower().endswith(".csv"):
        column = pd.read_csv(dataset_path, usecols=["impairment_mode"], encoding="utf-8")["impairment_mode"]
    else:
        raise ConfusionReportError("dataset must be .csv or .parquet")

    max_index = int(row_indices.max())
    if max_index >= len(column):
        raise ConfusionReportError(
            f"predictions reference row_index={max_index} but the dataset has only "
            f"{len(column)} rows - wrong dataset for these predictions?"
        )
    return column.reindex(row_indices).reset_index(drop=True)


# --------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------

def build_confusion_matrix(predictions):
    """Counts of (true class, predicted class), restricted to classes present."""
    present = sorted(
        set(predictions["y_true"]) | set(predictions["y_pred"]),
        key=lambda c: CANONICAL_CLASS_ORDER.index(c) if c in CANONICAL_CLASS_ORDER else 99,
    )
    counts = {true: {pred: 0 for pred in present} for true in present}
    for true, pred in zip(predictions["y_true"], predictions["y_pred"]):
        counts[true][pred] += 1
    return present, counts


def confusion_matrix_lines(present, counts):
    header = "| true \\ predicted | " + " | ".join(display_name(c) for c in present) + " | total |"
    sep = "|---" * (len(present) + 2) + "|"
    lines = [header, sep]
    for true in present:
        row = counts[true]
        total = sum(row.values())
        cells = " | ".join(f"{row[pred]:,}" for pred in present)
        lines.append(f"| {display_name(true)} | {cells} | {total:,} |")
    return lines


# --------------------------------------------------------------------------
# Per-mode breakdown and FPR / alert burden
# --------------------------------------------------------------------------

def outcome_counts(y_pred_series, present_classes):
    """How a slice of predictions distributes across every present class."""
    counts = {c: 0 for c in present_classes}
    for pred in y_pred_series:
        counts[pred] = counts.get(pred, 0) + 1
    return counts


def alert_stats(y_pred_series):
    n = len(y_pred_series)
    if n == 0:
        return {"n": 0, "attack_fpr": None, "alert_rate": None}
    n_attack = sum(1 for p in y_pred_series if p in ATTACK_CLASSES)
    n_alert = sum(1 for p in y_pred_series if p != NORMAL_LABEL)
    return {
        "n": n,
        # Misclassified specifically as an attack: the exact harm each
        # mechanism's "falsifies" column (benign_controls.md SS3) targets.
        "attack_fpr": n_attack / n,
        # Misclassified as anything other than `normal`, benign_degradation
        # included: the broader "would this traffic look unusual at all"
        # question, useful even when the model gets the class right.
        "alert_rate": n_alert / n,
    }


def per_mode_breakdown(predictions, present_classes):
    benign = predictions[predictions["y_true"] == BENIGN_DEGRADATION_LABEL]
    modes = sorted(m for m in benign["impairment_mode"].unique() if m != NONE_MODE)
    rows = []
    for mode in modes:
        subset = benign[benign["impairment_mode"] == mode]
        rows.append({
            "mode": mode,
            "outcomes": outcome_counts(subset["y_pred"], present_classes),
            **alert_stats(subset["y_pred"]),
        })
    return rows


def normal_baselines(predictions):
    normal = predictions[predictions["y_true"] == NORMAL_LABEL]
    ideal = normal[normal["impairment_mode"] == NONE_MODE]
    inside_benign_run = normal[normal["impairment_mode"] != NONE_MODE]
    return {
        "ideal_normal": {"label": "normal (ideal, impairment_mode=NONE)", **alert_stats(ideal["y_pred"])},
        "baseline_in_benign_run": {
            "label": "normal (baseline messages inside a benign-impairment run)",
            **alert_stats(inside_benign_run["y_pred"]),
        },
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fmt_rate(value):
    return "n/a" if value is None else f"{value:.2%}"


def build_report(report_meta, present, counts, per_mode, baselines, dataset_path):
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    six_class = list(CANONICAL_CLASS_ORDER)
    missing_classes = [c for c in six_class if c not in present]

    lines = [
        "# Benign-degradation confusion report - Gray-GOOSE (card C)",
        "",
        f"- Generated: {generated}",
        f"- Source run: `{report_meta.get('dataset')}`",
        f"- Run status: `{report_meta.get('status')}`"
        + (" - **technical smoke, not a scientific result**" if report_meta.get("status") == "technical_smoke" else ""),
        f"- Protocol: `{report_meta.get('protocol')}`",
        f"- Model: `{report_meta.get('model')}`",
        f"- Dataset re-verified against: `{dataset_path}`",
        "",
        "## 1. Confusion matrix",
        "",
    ]
    if missing_classes:
        lines.append(
            f"> {len(missing_classes)}/6 classes absent from this run's true+predicted labels: "
            + ", ".join(display_name(c) for c in missing_classes)
            + f". Matrix below is {len(present)}x{len(present)}, not the full 6x6 - "
              "this is expected for a benign-only or attack-only pool and not an error."
        )
        lines.append("")
    lines += confusion_matrix_lines(present, counts)
    lines.append("")

    lines += [
        "## 2. Per-mode outcome breakdown",
        "",
        "What each impairment mode's `benign_degradation` rows actually get predicted",
        "as, across every class present in this run.",
        "",
    ]
    if not per_mode:
        lines.append("> No `benign_degradation` rows in this run.")
        lines.append("")
    else:
        header = "| mode | n | " + " | ".join(display_name(c) for c in present) + " |"
        lines.append(header)
        lines.append("|---" * (len(present) + 2) + "|")
        for row in per_mode:
            cells = " | ".join(
                f"{row['outcomes'].get(c, 0):,} ({row['outcomes'].get(c, 0) / row['n']:.1%})"
                for c in present
            )
            lines.append(f"| {row['mode']} | {row['n']:,} | {cells} |")
        lines.append("")

    lines += [
        "## 3. False-positive rate and alert burden by mode",
        "",
        "`attack_fpr`: fraction of a slice's rows predicted as one of the four attack",
        "classes - the specific confusion each mechanism is paired to falsify",
        "(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other",
        "than `normal` (`benign_degradation` included) - the broader \"does this traffic",
        "look unusual at all\" question.",
        "",
        "| slice | n | attack_fpr | alert_rate |",
        "|---|---:|---:|---:|",
    ]
    for key in ("ideal_normal", "baseline_in_benign_run"):
        b = baselines[key]
        lines.append(f"| {b['label']} | {b['n']:,} | {fmt_rate(b['attack_fpr'])} | {fmt_rate(b['alert_rate'])} |")
    for row in per_mode:
        lines.append(
            f"| benign_degradation: {row['mode']} | {row['n']:,} | "
            f"{fmt_rate(row['attack_fpr'])} | {fmt_rate(row['alert_rate'])} |"
        )
    lines.append("")

    if per_mode and any(c in present for c in ATTACK_CLASSES):
        worst = max(per_mode, key=lambda r: r["attack_fpr"] or 0.0)
        lines += [
            f"> Highest attack_fpr: **{worst['mode']}** ({fmt_rate(worst['attack_fpr'])}). A high",
            "> attack_fpr on a mode means the model is learning \"gap/anomaly in traffic\" as a",
            "> proxy for \"attack\" rather than the attack's actual signature - exactly the",
            "> failure mode this card exists to surface (see CLAUDE.md, \"Known baseline issues",
            "> driving the revision\").",
            "",
        ]
    elif per_mode:
        lines += [
            "> No attack classes present in this run (benign-only pool), so attack_fpr is",
            "> trivially 0.00% everywhere above - it is not yet evidence of anything. This",
            "> comparison only becomes meaningful once attack and benign runs are pooled",
            "> together (checklist C5).",
            "",
        ]

    return lines


def write_report(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", required=True,
                   help="grouped_validation_report.json from run_grouped_validation.py.")
    p.add_argument("--predictions", required=True,
                   help="grouped_predictions.csv from run_grouped_validation.py.")
    p.add_argument("--dataset", default=None,
                   help="Override the dataset path recorded in --report (default: use it as-is).")
    p.add_argument("--out", default=None,
                   help="Output Markdown path (default: benign_confusion.md next to this script).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report_meta = load_json(args.report)
        predictions = load_predictions(args.predictions)

        dataset_path = args.dataset or report_meta.get("dataset")
        if not dataset_path:
            raise ConfusionReportError("no dataset path in --report and none given via --dataset")

        predictions["impairment_mode"] = load_impairment_modes(
            dataset_path, report_meta.get("dataset_sha256"), predictions["row_index"]
        )

        present, counts = build_confusion_matrix(predictions)
        per_mode = per_mode_breakdown(predictions, present)
        baselines = normal_baselines(predictions)

        out_path = args.out or os.path.join(HERE, "benign_confusion.md")
        lines = build_report(report_meta, present, counts, per_mode, baselines, dataset_path)
        write_report(out_path, lines)
    except (OSError, json.JSONDecodeError, ConfusionReportError, ValueError) as exc:
        print("BENIGN CONFUSION REPORT FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print(f"Confusion report written: {out_path}")
    print(f"  classes present: {len(present)}/6")
    print(f"  impairment modes with benign_degradation rows: {len(per_mode)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
