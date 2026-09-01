"""Create and persist leakage-free grouped validation folds.

Checklist B.1/B.3/B.4.  The canonical protocol is StratifiedGroupKFold: it
keeps runs intact while distributing message labels across folds. GroupKFold,
LeaveOneGroupOut, and an explicit leave-one-event-type-out diagnostic are also
available. Splits are saved as JSON and long-form CSV, then passed through
``check_no_leakage.py`` before success is reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from check_no_leakage import ValidationError, validate_splits


FORBIDDEN_GROUPS = {"T-UNRESOLVED"}


class SplitPlanningError(ValueError):
    pass


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_event_type(value):
    """Map a scenario/run identifier to an event-type family.

    Four attack mechanism families (DB/FRG/PB/PBM) plus, for card-C benign
    runs (`SC-BENIGN_<MODE>-...`), one family per impairment mechanism. The
    benign families give LOETO an event-type axis orthogonal to `class` -
    every benign run carries both `normal` and `benign_degradation` rows, so
    holding out one mechanism's runs no longer holds out a class the way
    holding out an attack variant does (see validation_protocol.md, "Leave-
    one-event-type-out status").
    """
    text = str(value).upper()
    mappings = [
        ("DETERMINISTIC_BURST", "SAG.DB"),
        ("FULLY_RANDOMIZED", "FRG"),
        ("RANDOMIC_BURST", "SAG.PB"),
        ("RANDOMIC_MESSAGE", "SAG.PBM"),
        ("BENIGN_CONGESTION_LOSS", "BENIGN.CONGESTION_LOSS"),
        ("BENIGN_QUEUE_OVERLOAD_BURST", "BENIGN.QUEUE_OVERLOAD_BURST"),
        ("BENIGN_JITTER", "BENIGN.JITTER"),
        ("BENIGN_DELAY", "BENIGN.DELAY"),
        ("BENIGN_LINK_FLAP", "BENIGN.LINK_FLAP"),
        ("BENIGN_DUPLICATION", "BENIGN.DUPLICATION"),
        ("BENIGN_REORDERING", "BENIGN.REORDERING"),
    ]
    for marker, label in mappings:
        if marker in text:
            return label
    for label in ("SAG.PBM", "SAG.PB", "SAG.DB", "FRG"):
        if label in text:
            return label
    raise SplitPlanningError("cannot infer event type from %r" % value)


def load_split_columns(path, group_column, target_column, scenario_column):
    import pandas as pd
    columns = [group_column, target_column]
    if scenario_column not in columns:
        columns.append(scenario_column)
    if path.lower().endswith(".csv"):
        frame = pd.read_csv(path, usecols=columns, encoding="utf-8")
    elif path.lower().endswith((".parquet", ".pq")):
        frame = pd.read_parquet(path, columns=columns)
    else:
        raise SplitPlanningError("dataset must be .csv or .parquet")
    if frame[columns].isna().any().any():
        raise SplitPlanningError("split metadata columns contain null values")
    for column in columns:
        frame[column] = frame[column].astype(str)
    return frame


def group_metadata(frame, group_column, target_column, scenario_column):
    groups = sorted(frame[group_column].unique())
    forbidden = sorted(set(groups) & FORBIDDEN_GROUPS)
    if forbidden:
        raise SplitPlanningError("forbidden non-independent groups present: %s" % forbidden)

    labels_by_group = frame.groupby(group_column, observed=True)[target_column].agg(
        lambda values: sorted(set(values))
    ).to_dict()
    scenario_counts = frame.groupby(group_column, observed=True)[scenario_column].nunique()
    bad = sorted(scenario_counts[scenario_counts != 1].index.astype(str))
    if bad:
        raise SplitPlanningError("groups map to multiple scenarios: %s" % bad)
    scenario_by_group = frame.groupby(group_column, observed=True)[scenario_column].first().to_dict()
    event_type_by_group = {group: infer_event_type(scenario_by_group[group]) for group in groups}
    rows_by_group = frame.groupby(group_column, observed=True).size().astype(int).to_dict()
    return groups, labels_by_group, event_type_by_group, rows_by_group


def _record(split_id, train_groups, test_groups, labels_by_group, rows_by_group,
            held_out_event_type=None):
    train_groups = sorted(str(v) for v in train_groups)
    test_groups = sorted(str(v) for v in test_groups)
    train_labels = sorted({label for group in train_groups for label in labels_by_group[group]})
    test_labels = sorted({label for group in test_groups for label in labels_by_group[group]})
    return {
        "split_id": split_id,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "train_rows": sum(rows_by_group[g] for g in train_groups),
        "test_rows": sum(rows_by_group[g] for g in test_groups),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "test_only_labels": sorted(set(test_labels) - set(train_labels)),
        **({"held_out_event_type": held_out_event_type} if held_out_event_type else {}),
    }


def make_grouped_splits(frame, protocol, n_splits, seed, group_column="split_group",
                        target_column="class", scenario_column="scenario_id"):
    """Build fold records using scikit-learn group splitters."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold

    groups, labels_by_group, event_type_by_group, rows_by_group = group_metadata(
        frame, group_column, target_column, scenario_column
    )
    if len(groups) < 2:
        raise SplitPlanningError("at least two independent groups are required")

    records = []
    if protocol in ("stratified-group-kfold", "group-kfold"):
        if not 2 <= n_splits <= len(groups):
            raise SplitPlanningError("n_splits must be between 2 and the number of groups")
        if protocol == "stratified-group-kfold":
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
        else:
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for index, (train_idx, test_idx) in enumerate(
                splitter.split(frame, frame[target_column], groups=frame[group_column])):
            train_groups = set(frame.iloc[train_idx][group_column])
            test_groups = set(frame.iloc[test_idx][group_column])
            records.append(_record("fold-%02d" % index, train_groups, test_groups,
                                   labels_by_group, rows_by_group))
    elif protocol == "leave-one-group-out":
        splitter = LeaveOneGroupOut()
        for index, (train_idx, test_idx) in enumerate(
                splitter.split(frame, frame[target_column], groups=frame[group_column])):
            train_groups = set(frame.iloc[train_idx][group_column])
            test_groups = set(frame.iloc[test_idx][group_column])
            records.append(_record("logo-%02d" % index, train_groups, test_groups,
                                   labels_by_group, rows_by_group))
    elif protocol == "leave-one-event-type-out":
        event_types = sorted(set(event_type_by_group.values()))
        if len(event_types) < 3:
            raise SplitPlanningError(
                "leave-one-event-type-out requires at least three event types"
            )
        all_groups = set(groups)
        for event_type in event_types:
            test_groups = {g for g in groups if event_type_by_group[g] == event_type}
            records.append(_record(
                "loeto-%s" % re.sub(r"[^A-Za-z0-9]+", "-", event_type).strip("-").lower(),
                all_groups - test_groups, test_groups, labels_by_group, rows_by_group,
                held_out_event_type=event_type,
            ))
    else:
        raise SplitPlanningError("unknown protocol: %s" % protocol)
    return records, groups, event_type_by_group


def validate_class_coverage(records, allow_unseen_test_classes=False):
    failures = {
        record["split_id"]: record["test_only_labels"]
        for record in records if record["test_only_labels"]
    }
    if failures and not allow_unseen_test_classes:
        details = "; ".join("%s=%s" % item for item in failures.items())
        raise SplitPlanningError(
            "test labels absent from training (%s). Generate multiple independent "
            "runs per label or use --allow-unseen-test-classes only for an explicit "
            "open-set/OOD diagnostic." % details
        )
    return failures


def write_split_files(json_path, csv_path, payload):
    for path in (json_path, csv_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split_id", "partition", "split_group"])
        for split in payload["splits"]:
            for partition in ("train", "test"):
                for group in split[partition + "_groups"]:
                    writer.writerow([split["split_id"], partition, group])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Prepared annotated CSV/Parquet")
    parser.add_argument("--protocol", choices=["stratified-group-kfold", "group-kfold",
                                                "leave-one-group-out",
                                                "leave-one-event-type-out"],
                        default="stratified-group-kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--scenario-column", default="scenario_id")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--allow-unseen-test-classes", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        frame = load_split_columns(
            args.dataset, args.group_column, args.target_column, args.scenario_column
        )
        records, groups, event_types = make_grouped_splits(
            frame, args.protocol, args.n_splits, args.seed,
            args.group_column, args.target_column, args.scenario_column,
        )
        unseen = validate_class_coverage(records, args.allow_unseen_test_classes)
        # Reuse the independent checker delivered in A.4. This is intentionally
        # not just an assertion inside the split-generation code.
        validate_splits(records, set(groups), require_complete=True, require_unique_test=True)
        payload = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "dataset": os.path.abspath(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "dataset_rows": len(frame),
            "protocol": args.protocol,
            "n_splits": len(records),
            "seed": args.seed,
            "group_column": args.group_column,
            "target_column": args.target_column,
            "groups": len(groups),
            "event_types": dict(sorted(event_types.items())),
            "open_set_diagnostic": bool(unseen),
            "splits": records,
        }
        write_split_files(args.out_json, args.out_csv, payload)
    except (OSError, ValidationError, SplitPlanningError, ValueError) as exc:
        print("GROUPED SPLIT GENERATION FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print("Saved %d leakage-free %s splits across %d groups." %
          (len(records), args.protocol, len(groups)))
    if unseen:
        print("WARNING: open-set diagnostic; some test labels are absent from training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
