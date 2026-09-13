"""Where does a run's attack signal actually come from?

Checklist ref.: D.1 in miniature, and the control `label_duplication_audit.md`
needed to keep its own conclusion honest.

Given a finished `--save-scores` run, this scores three things on the same
average-precision axis, against the same positives:

  - the **model**, from its persisted posteriors;
  - each **single feature**, used directly as a score (best orientation, since
    a raw feature may rank either way);
  - any **trivial rule** passed as `--rule`, e.g. `sqDiff != 0`.

Why it exists
-------------
`label_duplication_audit.md` established that every attack row has a
content-identical `normal` row, so no content feature can discriminate and all
of it must come from the eight delta columns. The obvious next inference -
that the model is really just reading "am I the duplicate copy" - is the kind
of claim that has to be measured rather than reasoned to, and on the
265-run pool it turns out to be **false**: `sqDiff != 0` scores exactly
chance while the model scores 7.8x it.

That matters in both directions. It keeps the audit from overstating what the
defect implies, and it says the model's lift is a multivariate pattern that no
single-feature marginal explains - which is precisely why D.1 needs to ablate
feature *groups* and not rank features one at a time.

Usage
-----
    python experiments/revision_2026/feature_signal_probe.py \\
      --run results/d5-xgboost-none \\
      --dataset data/runs/gray-GOOSE-runs-prepared.parquet \\
      --rule "sqDiff != 0" \\
      --out experiments/revision_2026/feature_signal.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from benign_confusion_report import ATTACK_CLASSES
from grouped_pr_curves import (
    average_precision,
    suffix_counts,
    threshold_edges,
)

HERE = os.path.dirname(os.path.abspath(__file__))


class ProbeError(ValueError):
    pass


def normalised(raw):
    """Map a raw feature onto (0, 1) monotonically.

    Only the ranking matters to average precision, so this is a presentation
    detail - but it has to be robust to the heavy tails these columns have,
    hence percentile clipping rather than min/max.
    """
    import numpy as np

    values = np.asarray(raw, dtype="float64")
    low, high = np.nanpercentile(values, 0.01), np.nanpercentile(values, 99.99)
    scaled = (values - low) / (high - low + 1e-12)
    return np.clip(scaled, 1e-6, 1 - 1e-6).astype("float32")


def average_precision_of(score, is_attack, edges, both_orientations=True):
    """AP for a score, trying both orientations when the score is a raw feature."""
    import numpy as np

    positives = int(is_attack.sum())
    n_bins = len(edges) + 1
    best = float("-inf")
    for candidate in ((score, 1.0 - score) if both_orientations else (score,)):
        bins = np.searchsorted(edges, candidate, side="left")
        counts = np.stack([np.bincount(bins[~is_attack], minlength=n_bins),
                           np.bincount(bins[is_attack], minlength=n_bins)])
        cumulative = suffix_counts(counts)
        best = max(best, average_precision(cumulative[1].astype(float),
                                           cumulative[0].astype(float), positives))
    return best


def load_model_scores(directory, rows):
    """Per-row attack score = the posterior mass on the attack classes."""
    import numpy as np
    import pyarrow.parquet as pq

    path = os.path.join(directory, "grouped_scores.parquet")
    if not os.path.exists(path):
        raise ProbeError("%s has no grouped_scores.parquet (re-run with --save-scores)"
                         % directory)
    handle = pq.ParquetFile(path)
    columns = ["p_%s" % name for name in ATTACK_CLASSES
               if "p_%s" % name in handle.schema_arrow.names]
    if not columns:
        raise ProbeError("no attack posterior columns in %s" % path)
    score = np.zeros(rows, dtype="float32")
    seen = np.zeros(rows, dtype=bool)
    for number in range(handle.num_row_groups):
        frame = handle.read_row_group(number, columns=["row_index"] + columns).to_pandas()
        index = frame["row_index"].to_numpy()
        score[index] = frame[columns].to_numpy(dtype="float32").sum(axis=1)
        seen[index] = True
        del frame
    if not seen.all():
        raise ProbeError("scores cover %d of %d dataset rows" % (int(seen.sum()), rows))
    return score


def load_columns(dataset, columns, label_column="class"):
    """Feature columns plus the attack mask, in dataset row order."""
    import numpy as np
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(dataset)
    rows = handle.metadata.num_rows
    held = {name: np.empty(rows, dtype="float32") for name in columns}
    is_attack = np.empty(rows, dtype=bool)
    offset = 0
    for number in range(handle.num_row_groups):
        frame = handle.read_row_group(
            number, columns=list(columns) + [label_column]).to_pandas()
        count = len(frame)
        for name in columns:
            held[name][offset:offset + count] = frame[name].to_numpy()
        labels = frame[label_column].astype(str).to_numpy()
        is_attack[offset:offset + count] = np.isin(labels, ATTACK_CLASSES)
        offset += count
        del frame
    if offset != rows:
        raise ProbeError("read %d rows, expected %d" % (offset, rows))
    return held, is_attack


def evaluate_rule(expression, held):
    """A trivial baseline written as a Python expression over the columns.

    Evaluated with no builtins and only the loaded columns in scope: this is a
    convenience for writing `sqDiff != 0` on a command line, not an extension
    point, and it should not become one.
    """
    import numpy as np

    try:
        value = eval(expression, {"__builtins__": {}, "np": np}, dict(held))
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as-is
        raise ProbeError("could not evaluate rule %r: %s" % (expression, exc))
    return np.asarray(value, dtype="float32") * 0.9 + 0.05


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="A run directory with persisted scores.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature", action="append", default=None,
                        help="Feature to probe (default: the run's delta columns).")
    parser.add_argument("--rule", action="append", default=[],
                        help="A trivial baseline, e.g. \"sqDiff != 0\". Repeatable.")
    parser.add_argument("--out", default=os.path.join(HERE, "feature_signal.md"))
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = json.load(open(os.path.join(args.run, "grouped_validation_report.json"),
                                encoding="utf-8"))
        available = list(report.get("features", []))
        features = args.feature or [f for f in available
                                    if f.endswith("Diff") or f in
                                    ("delay", "timeFromLastChange")]
        missing = [f for f in features if f not in available]
        if missing:
            raise ProbeError("not model features in this run: %s" % missing)

        edges = threshold_edges()
        held, is_attack = load_columns(args.dataset, features)
        rows = len(is_attack)
        prevalence = float(is_attack.mean())
        model = load_model_scores(args.run, rows)

        results = [("[model] %s/%s" % (report.get("model"), report.get("balance")),
                    average_precision_of(model, is_attack, edges, both_orientations=False))]
        for expression in args.rule:
            results.append(("[rule] %s" % expression,
                            average_precision_of(evaluate_rule(expression, held),
                                                 is_attack, edges,
                                                 both_orientations=False)))
        for name in features:
            results.append((name, average_precision_of(normalised(held[name]),
                                                       is_attack, edges)))

        lines = [
            "# Where the attack signal is",
            "",
            "- Generated: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "- Run: `%s` | dataset rows: %s | attack prevalence: %.4f%%"
            % (os.path.basename(os.path.normpath(args.run)), f"{rows:,}",
               100 * prevalence),
            "",
            "Average precision on one axis. **Chance is the prevalence, %.4f** - a"
            % prevalence,
            "single feature scoring near it carries no marginal signal at all.",
            "",
            "| Score | AP | x chance |",
            "|---|---:|---:|",
        ]
        for name, value in sorted(results, key=lambda item: -item[1]):
            lines.append("| `%s` | %.4f | %.1fx |" % (name, value, value / prevalence))
        lines.append("")
        text = "\n".join(lines) + "\n"
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        if args.report:
            with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"run": args.run, "rows": rows, "prevalence": prevalence,
                           "average_precision": dict(results)}, fh, indent=2)
                fh.write("\n")
    except (OSError, json.JSONDecodeError, ProbeError, ValueError, KeyError) as exc:
        print("FEATURE SIGNAL PROBE FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
