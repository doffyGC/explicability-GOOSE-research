"""Validate grouped train/test splits before any model is trained.

Checklist item A.4.  The unit of independence is ``split_group`` (normally a
run), never an individual message.  This command fails with exit status 1 if a
group crosses the train/test boundary, if a split references an unknown group,
or if the fold collection is incomplete/inconsistent.

Accepted split formats
----------------------
JSON (recommended)::

    {
      "group_column": "split_group",
      "splits": [
        {"split_id": "fold-0", "train_groups": ["R02", "R03"],
         "test_groups": ["R01"]}
      ]
    }

CSV (long form)::

    split_id,partition,split_group
    fold-0,train,R02
    fold-0,test,R01

Usage::

    python experiments/revision_2026/check_no_leakage.py \
        --dataset data/gray-GOOSE-runs-metadata.parquet \
        --splits experiments/revision_2026/splits.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter


class ValidationError(ValueError):
    """A split file is malformed or violates grouped-validation integrity."""


def _normalise_group(value):
    if value is None:
        raise ValidationError("group identifiers cannot be null")
    value = str(value).strip()
    if not value:
        raise ValidationError("group identifiers cannot be empty")
    return value


def _normalise_split(raw, position):
    if not isinstance(raw, dict):
        raise ValidationError("split %d must be a JSON object" % position)
    split_id = str(raw.get("split_id", "fold-%d" % position)).strip()
    if not split_id:
        raise ValidationError("split %d has an empty split_id" % position)
    missing = {"train_groups", "test_groups"} - set(raw)
    if missing:
        raise ValidationError("%s is missing %s" % (split_id, sorted(missing)))
    if not isinstance(raw["train_groups"], list) or not isinstance(raw["test_groups"], list):
        raise ValidationError("%s train_groups/test_groups must be JSON lists" % split_id)
    train = [_normalise_group(v) for v in raw["train_groups"]]
    test = [_normalise_group(v) for v in raw["test_groups"]]
    return {"split_id": split_id, "train_groups": train, "test_groups": test}


def load_splits(path):
    """Read the canonical JSON format or long-form CSV."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        raw_splits = payload.get("splits") if isinstance(payload, dict) else payload
        if not isinstance(raw_splits, list) or not raw_splits:
            raise ValidationError("JSON must contain a non-empty `splits` list")
        splits = [_normalise_split(raw, i) for i, raw in enumerate(raw_splits)]
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"split_id", "partition", "split_group"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValidationError("CSV columns must include %s" % sorted(required))
            grouped = {}
            order = []
            for line, row in enumerate(reader, 2):
                split_id = str(row["split_id"]).strip()
                partition = str(row["partition"]).strip().lower()
                if not split_id:
                    raise ValidationError("CSV line %d has an empty split_id" % line)
                if partition not in ("train", "test"):
                    raise ValidationError("CSV line %d partition must be train or test" % line)
                if split_id not in grouped:
                    grouped[split_id] = {"split_id": split_id, "train_groups": [], "test_groups": []}
                    order.append(split_id)
                grouped[split_id][partition + "_groups"].append(_normalise_group(row["split_group"]))
        splits = [grouped[key] for key in order]
        if not splits:
            raise ValidationError("CSV contains no split rows")
    else:
        raise ValidationError("split format must be .json or .csv")

    ids = [s["split_id"] for s in splits]
    duplicates = sorted(k for k, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise ValidationError("duplicate split_id values: %s" % duplicates)
    return splits


def load_dataset_groups(path, group_column):
    """Load only the group column so large datasets do not fill memory."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        groups = set()
        rows = 0
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or group_column not in reader.fieldnames:
                raise ValidationError("dataset has no `%s` column" % group_column)
            for line, row in enumerate(reader, 2):
                try:
                    groups.add(_normalise_group(row[group_column]))
                except ValidationError as exc:
                    raise ValidationError("dataset line %d: %s" % (line, exc)) from exc
                rows += 1
    elif suffix in (".parquet", ".pq"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValidationError("pandas and pyarrow are required to read Parquet") from exc
        frame = pd.read_parquet(path, columns=[group_column])
        if group_column not in frame:
            raise ValidationError("dataset has no `%s` column" % group_column)
        if frame[group_column].isna().any():
            raise ValidationError("dataset `%s` contains null values" % group_column)
        groups = {_normalise_group(v) for v in frame[group_column].unique()}
        rows = len(frame)
    else:
        raise ValidationError("dataset format must be .csv or .parquet")
    if not groups:
        raise ValidationError("dataset contains no groups")
    return groups, rows


def validate_splits(splits, dataset_groups, require_complete=True,
                    require_unique_test=True):
    """Return a structured report; raise ValidationError on any violation."""
    errors = []
    test_counts = Counter()

    for split in splits:
        split_id = split["split_id"]
        train_list = split["train_groups"]
        test_list = split["test_groups"]
        train = set(train_list)
        test = set(test_list)

        duplicate_train = sorted(k for k, n in Counter(train_list).items() if n > 1)
        duplicate_test = sorted(k for k, n in Counter(test_list).items() if n > 1)
        if duplicate_train:
            errors.append("%s repeats train groups: %s" % (split_id, duplicate_train))
        if duplicate_test:
            errors.append("%s repeats test groups: %s" % (split_id, duplicate_test))
        if not train:
            errors.append("%s has no train groups" % split_id)
        if not test:
            errors.append("%s has no test groups" % split_id)

        overlap = sorted(train & test)
        if overlap:
            errors.append("%s LEAKAGE: groups in both train and test: %s" % (split_id, overlap))

        unknown = sorted((train | test) - dataset_groups)
        if unknown:
            errors.append("%s references unknown groups: %s" % (split_id, unknown))

        if require_complete:
            omitted = sorted(dataset_groups - (train | test))
            if omitted:
                errors.append("%s omits dataset groups: %s" % (split_id, omitted))

        test_counts.update(test)

    if require_unique_test:
        repeated = sorted(group for group, count in test_counts.items() if count > 1)
        missing = sorted(dataset_groups - set(test_counts))
        if repeated:
            errors.append("test groups repeated across folds: %s" % repeated)
        if missing:
            errors.append("groups never used for testing: %s" % missing)

    if errors:
        raise ValidationError("\n  - ".join(["split validation failed:"] + errors))

    return {
        "splits": len(splits),
        "dataset_groups": len(dataset_groups),
        "test_assignments": sum(test_counts.values()),
        "every_group_tested_once": all(test_counts[g] == 1 for g in dataset_groups),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Annotated .csv/.parquet dataset.")
    parser.add_argument("--splits", required=True, help="Split definition (.json or long-form .csv).")
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--allow-unused-groups", action="store_true",
                        help="Allow groups absent from both train and test within a split (holdout/subset designs).")
    parser.add_argument("--allow-repeated-test-groups", action="store_true",
                        help="Allow repeated or not-yet-tested groups across folds (repeated/partial CV).")
    parser.add_argument("--forbid-group", action="append", default=[],
                        help="Additional group identifier that must never enter a split (repeatable).")
    parser.add_argument("--report", help="Optional JSON report written only after successful validation.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        groups, rows = load_dataset_groups(args.dataset, args.group_column)
        # The legacy annotation uses this marker for a pool of messages from
        # several real traces. It is not an independent group and assigning it
        # to either side would create hidden leakage despite disjoint strings.
        forbidden = {"T-UNRESOLVED"} | {_normalise_group(v) for v in args.forbid_group}
        present_forbidden = sorted(groups & forbidden)
        if present_forbidden:
            raise ValidationError(
                "dataset contains forbidden non-independent groups: %s" % present_forbidden
            )
        splits = load_splits(args.splits)
        report = validate_splits(
            splits, groups,
            require_complete=not args.allow_unused_groups,
            require_unique_test=not args.allow_repeated_test_groups,
        )
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        print("LEAKAGE CHECK FAILED\n%s" % exc, file=sys.stderr)
        return 1

    report.update({
        "status": "pass",
        "dataset": os.path.abspath(args.dataset),
        "dataset_rows": rows,
        "group_column": args.group_column,
        "splits_file": os.path.abspath(args.splits),
    })
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    print("NO GROUP LEAKAGE: %d split(s), %d dataset group(s), %d rows." %
          (report["splits"], report["dataset_groups"], rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
