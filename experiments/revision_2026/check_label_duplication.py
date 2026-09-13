"""Is the label a function of the features? On the regenerated pool, no.

Checklist ref.: A.5/B.5 (dataset integrity), and the precondition every
card-D and card-E number silently assumed.

The invariant
-------------
A supervised dataset must not contain two rows with identical model features
and different labels. When it does, no classifier can separate them and every
reported precision is capped by an artifact of the data rather than by the
phenomenon being studied. Nothing in the existing chain checks this:
`check_no_leakage.py` owns train/test group overlap, `merge_runs.py` owns
cross-run payload sharing, and `check_prediction_integrity.py` audits
predictions after the fact. All three pass on a dataset that violates this.

What this script found on `data/runs/` (2026-09-13)
---------------------------------------------------
**Every attack row in the regenerated 265-run pool has a row labelled
`normal`, in the same run, whose 32 content features are bit-identical.**
100.0%, all four variants, in the raw ERENO CSVs before any pipeline step
touches them.

The mechanism: the regenerated ERENO emits most GOOSE messages **twice**.
Across all 65 raw columns the two copies differ only in the five
inter-message delta columns - `sqDiff`, `timestampDiff`, `tDiff`, `stDiff`,
`cbStatusDiff`, which are relative to the predecessor and so are necessarily
different on a duplicate - and in `class`, on exactly the attacked messages.
`ethSrc`, `ethDst`, `batch_index` and every SV value match, so the pair is not
two subscribers and not two SV cycles correlated to one message: it is the
same message, written twice, with the attack label applied to one copy.

The legacy dataset behind the submitted paper does **not** have this:
1,006,981 distinct message keys over 1,006,989 rows, and 15 attack rows
(0.004%) sharing a key with a `normal` row. The defect was introduced by the
regeneration, not inherited.

Why it matters more than it looks
---------------------------------
The content features cannot separate an attack row from its twin, so all
discriminative power is pushed onto the delta columns - and on the twin those
deltas differ *because the row is a duplicate*, not because of the attack. A
model that scores well by reading `sqDiff == 0` has learned "am I the second
copy", which is a property of the writer, not of a grayhole.

Usage
-----
    python experiments/revision_2026/check_label_duplication.py \\
      --runs-dir data/runs \\
      --report experiments/revision_2026/label_duplication.json

    python experiments/revision_2026/check_label_duplication.py \\
      --dataset data/runs/gray-GOOSE-runs-prepared.parquet

Exit code 1 when any attack row has a content twin, so this can gate a
regeneration the way `check_no_leakage.py` gates training.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

# Relative to the predecessor, so a duplicate row necessarily differs here.
# Excluded from the "content" comparison for exactly that reason: including
# them would hide the duplication behind the arithmetic it causes.
DELTA_COLUMNS = {
    "stDiff", "sqDiff", "gooseLengthDiff", "cbStatusDiff", "apduSizeDiff",
    "frameLengthDiff", "timestampDiff", "tDiff",
}

# Provenance, not observation: present in the raw CSVs but never model input.
PROVENANCE_COLUMNS = {
    "class", "run_id", "trace_id", "batch_index", "scenario_id", "seed",
    "attack_variant", "loss_rate", "burst_size", "traffic_rate",
    "substation_config", "impairment_mode", "impairment_rate",
    "impairment_intensity_ms", "split_group", "event_id", "message_index",
}

DEFAULT_KEY = ("StNum", "SqNum", "GooseTimestamp", "t")
NON_ATTACK = ("normal", "benign_degradation")


class DuplicationError(ValueError):
    pass


def row_keys(values):
    """Exact byte identity per row - not a hash, so there is no collision caveat."""
    import numpy as np

    block = np.ascontiguousarray(values)
    return block.view(np.dtype((np.void, block.dtype.itemsize * block.shape[1]))).ravel()


def feature_block(frame, columns):
    """One exact numeric block for columns of mixed dtype.

    Strings (MAC addresses, `gocbRef`, `goID`) are factorised rather than
    dropped: a twin must match on *everything* a row says about itself, and
    silently ignoring the identity fields would overstate the twin rate.
    Factorisation is run-local, which is the only scope twins are computed in.
    `float64`, not the model's `float32`: a coarser cast could merge two
    genuinely different values into one twin, so this errs toward finding
    fewer twins than the model would actually see.
    """
    import numpy as np
    import pandas as pd

    parts = []
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    if numeric:
        parts.append(frame[numeric].to_numpy(dtype="float64"))
    for column in columns:
        if column in numeric:
            continue
        codes = pd.factorize(frame[column].astype(str))[0]
        parts.append(codes.astype("float64").reshape(-1, 1))
    if not parts:
        raise DuplicationError("no comparable columns")
    return np.hstack(parts)


def twin_counts(values, is_attack, labels=None):
    """Rows whose exact feature vector also occurs under a different label.

    Returns two counts. `attack` answers the card-D/E question - attack rows
    with a non-attack twin - and `conflicting` answers the general one, any
    row whose content is repeated under a different label at all. The second
    matters for card C: a `benign_degradation` row with a `normal` twin is
    the same defect and would otherwise go unreported, because
    `benign_degradation` is not an attack.
    """
    import numpy as np

    if not len(values):
        return 0, 0
    keys = row_keys(values)
    _, inverse = np.unique(keys, return_inverse=True)
    size = int(inverse.max()) + 1
    attack = np.bincount(inverse[is_attack], minlength=size)
    other = np.bincount(inverse[~is_attack], minlength=size)
    shared = (attack > 0) & (other > 0)
    attack_twins = int(attack[shared].sum())

    conflicting = 0
    if labels is not None:
        codes = np.unique(labels, return_inverse=True)[1]
        pairs = np.unique(np.stack([inverse, codes], axis=1), axis=0)
        labels_per_group = np.bincount(pairs[:, 0], minlength=size)
        conflicting = int(np.isin(inverse, np.flatnonzero(labels_per_group > 1)).sum())
    return attack_twins, conflicting


def varying_columns(frame, key_columns):
    """Which columns ever differ between two rows sharing a message key.

    This is the diagnostic that names the cause rather than just detecting the
    symptom: a duplicate whose only differences are the delta columns and
    `class` is a writer emitting one message twice, not two distinct messages.
    """
    duplicated = frame[frame.duplicated(subset=list(key_columns), keep=False)]
    if duplicated.empty:
        return {}, 0
    marker = duplicated.groupby(list(key_columns), sort=False).ngroup()
    counts = {}
    for column in frame.columns:
        if column in key_columns:
            continue
        varies = duplicated.groupby(marker)[column].nunique(dropna=False)
        hits = int((varies > 1).sum())
        if hits:
            counts[column] = hits
    return counts, int(marker.nunique())


def audit_frame(frame, label_column, key_columns, content_columns, feature_columns):
    import numpy as np

    labels = frame[label_column].astype(str).to_numpy()
    is_attack = ~np.isin(labels, NON_ATTACK)
    present = [c for c in key_columns if c in frame.columns]
    result = {
        "rows": int(len(frame)),
        "attack_rows": int(is_attack.sum()),
        "distinct_message_keys": None,
        "rows_on_duplicated_keys": None,
        "keys_with_conflicting_labels": None,
        "attack_rows_sharing_a_key": None,
    }
    result["content_twins"], result["content_conflicting_rows"] = twin_counts(
        feature_block(frame, content_columns), is_attack, labels)
    result["full_twins"], result["full_conflicting_rows"] = twin_counts(
        feature_block(frame, feature_columns), is_attack, labels)
    if len(present) == len(key_columns):
        grouped = frame.groupby(list(key_columns), sort=False)
        distinct = grouped.ngroups
        result["distinct_message_keys"] = int(distinct)
        result["rows_on_duplicated_keys"] = int(
            frame.duplicated(subset=list(key_columns), keep=False).sum())
        classes_per_key = grouped[label_column].nunique()
        result["keys_with_conflicting_labels"] = int((classes_per_key > 1).sum())
        conflicted = classes_per_key[classes_per_key > 1].index
        if len(conflicted):
            marker = frame.set_index(list(key_columns)).index.isin(conflicted)
            result["attack_rows_sharing_a_key"] = int((marker & is_attack).sum())
        else:
            result["attack_rows_sharing_a_key"] = 0
    return result


def column_sets(columns, label_column):
    feature_columns = [c for c in columns
                       if c not in PROVENANCE_COLUMNS and c != label_column]
    content_columns = [c for c in feature_columns if c not in DELTA_COLUMNS]
    if not content_columns:
        raise DuplicationError("no content columns left to compare")
    return feature_columns, content_columns


def audit_runs_dir(directory, label_column, key_columns, limit=0):
    """One raw per-run CSV at a time - the state before any pipeline step."""
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        raise DuplicationError("no per-run CSVs under %s" % directory)
    if limit:
        paths = paths[:limit]
    per_run, diagnostics = {}, {}
    for path in paths:
        frame = pd.read_csv(path)
        if label_column not in frame.columns:
            raise DuplicationError("%s has no %s column" % (path, label_column))
        features, content = column_sets(frame.columns, label_column)
        name = os.path.basename(path)[: -len(".csv")]
        per_run[name] = audit_frame(frame, label_column, key_columns, content, features)
        if not diagnostics and per_run[name]["rows_on_duplicated_keys"]:
            varying, keys = varying_columns(frame, key_columns)
            diagnostics = {"run": name, "duplicated_keys": keys,
                           "columns_that_differ_between_copies": varying}
        del frame
    return per_run, diagnostics


def audit_dataset(path, label_column, key_columns, group_column="split_group"):
    """The pooled Parquet, one row group at a time."""
    import pandas as pd
    import pyarrow.parquet as pq

    if path.lower().endswith((".parquet", ".pq")):
        handle = pq.ParquetFile(path)
        names = handle.schema_arrow.names
        if label_column not in names:
            raise DuplicationError("%s has no %s column" % (path, label_column))
        features, content = column_sets(names, label_column)
        wanted = sorted(set(features) | set(content) | {label_column}
                        | {c for c in key_columns if c in names}
                        | ({group_column} if group_column in names else set()))
        per_run = {}
        for number in range(handle.num_row_groups):
            frame = handle.read_row_group(number, columns=wanted).to_pandas()
            name = (str(frame[group_column].iloc[0]) if group_column in frame.columns
                    else "row-group-%d" % number)
            per_run[name] = audit_frame(frame, label_column, key_columns,
                                        content, features)
            del frame
        return per_run, {}
    frame = pd.read_csv(path)
    features, content = column_sets(frame.columns, label_column)
    groups = ({name: block for name, block in frame.groupby(group_column, sort=False)}
              if group_column in frame.columns else {"whole-dataset": frame})
    return ({str(name): audit_frame(block, label_column, key_columns, content, features)
             for name, block in groups.items()}, {})


def summarise(per_run):
    totals = {"runs": len(per_run), "rows": 0, "attack_rows": 0,
              "content_twins": 0, "full_twins": 0,
              "content_conflicting_rows": 0, "full_conflicting_rows": 0,
              "attack_rows_sharing_a_key": 0, "rows_on_duplicated_keys": 0}
    for entry in per_run.values():
        for key in list(totals):
            if key == "runs":
                continue
            value = entry.get(key)
            if value:
                totals[key] += value
    attack = max(totals["attack_rows"], 1)
    totals["content_twin_rate"] = totals["content_twins"] / attack
    totals["full_twin_rate"] = totals["full_twins"] / attack
    totals["duplicate_row_rate"] = (
        totals["rows_on_duplicated_keys"] / max(totals["rows"], 1))
    totals["content_conflict_rate"] = (
        totals["content_conflicting_rows"] / max(totals["rows"], 1))
    return totals


def build_lines(totals, diagnostics, source):
    lines = [
        "# Label/feature duplication audit",
        "",
        "- Generated: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "- Source: `%s`" % source,
        "",
        "| Quantity | Value |",
        "|---|---:|",
        "| runs audited | %s |" % f"{totals['runs']:,}",
        "| rows | %s |" % f"{totals['rows']:,}",
        "| rows sharing a message key with another row | %s (%.1f%%) |"
        % (f"{totals['rows_on_duplicated_keys']:,}", 100 * totals["duplicate_row_rate"]),
        "| attack rows | %s |" % f"{totals['attack_rows']:,}",
        "| **attack rows with a content-identical non-attack row** | **%s (%.2f%%)** |"
        % (f"{totals['content_twins']:,}", 100 * totals["content_twin_rate"]),
        "| attack rows identical on *every* model feature (irreducible) | %s (%.2f%%) |"
        % (f"{totals['full_twins']:,}", 100 * totals["full_twin_rate"]),
        "| **rows whose content is repeated under a different label** (any label) | **%s (%.2f%%)** |"
        % (f"{totals['content_conflicting_rows']:,}", 100 * totals["content_conflict_rate"]),
        "| same, identical on every model feature | %s |"
        % f"{totals['full_conflicting_rows']:,}",
        "",
        "`content` excludes the eight inter-message delta columns, which differ on a",
        "duplicate by construction; `full` includes them, so a full twin cannot be",
        "separated by any classifier reading this feature matrix.",
        "",
    ]
    if diagnostics:
        lines += [
            "## What differs between two copies of the same message",
            "",
            "Diagnostic from `%s` (%s duplicated keys). A pair differing *only* in the"
            % (diagnostics["run"], f"{diagnostics['duplicated_keys']:,}"),
            "delta columns and `class` is one message written twice, not two messages.",
            "",
            "| Column | keys where the copies differ |",
            "|---|---:|",
        ]
        for column, hits in sorted(diagnostics["columns_that_differ_between_copies"].items(),
                                   key=lambda item: -item[1]):
            lines.append("| `%s` | %s (%.1f%%) |" % (
                column, f"{hits:,}", 100 * hits / max(diagnostics["duplicated_keys"], 1)))
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="A pooled Parquet or CSV dataset.")
    source.add_argument("--runs-dir", help="A directory of raw per-run ERENO CSVs.")
    parser.add_argument("--label-column", default="class")
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--key-column", action="append", default=None,
                        help="Message identity columns (default: %s)."
                             % ", ".join(DEFAULT_KEY))
    parser.add_argument("--limit-runs", type=int, default=0,
                        help="Audit only the first N per-run CSVs (a smoke check).")
    parser.add_argument("--out", default=None, help="Markdown report path.")
    parser.add_argument("--report", default=None, help="JSON report path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    key_columns = tuple(args.key_column or DEFAULT_KEY)
    try:
        if args.runs_dir:
            per_run, diagnostics = audit_runs_dir(
                args.runs_dir, args.label_column, key_columns, args.limit_runs)
            source = args.runs_dir
        else:
            per_run, diagnostics = audit_dataset(
                args.dataset, args.label_column, key_columns, args.group_column)
            source = args.dataset
        totals = summarise(per_run)
        text = build_lines(totals, diagnostics, source)
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        if args.report:
            with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"source": source, "totals": totals,
                           "diagnostics": diagnostics, "per_run": per_run},
                          fh, indent=2)
                fh.write("\n")
    except (OSError, DuplicationError, ValueError, KeyError) as exc:
        print("LABEL DUPLICATION AUDIT FAILED\n%s" % exc, file=sys.stderr)
        return 2
    print(text)
    if totals["content_twins"] or totals["content_conflicting_rows"]:
        print("FAIL: %.2f%% of attack rows have a content-identical non-attack row; "
              "%s rows in total carry a label their content twin contradicts."
              % (100 * totals["content_twin_rate"],
                 f"{totals['content_conflicting_rows']:,}"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
