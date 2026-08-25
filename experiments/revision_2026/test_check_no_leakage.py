"""Unit tests for check_no_leakage.py (no generated dataset required)."""

import csv
import json
import os
import tempfile
import unittest

from check_no_leakage import ValidationError, load_splits, main, validate_splits


GROUPS = {"R01", "R02", "R03"}
VALID = [
    {"split_id": "fold-0", "train_groups": ["R02", "R03"], "test_groups": ["R01"]},
    {"split_id": "fold-1", "train_groups": ["R01", "R03"], "test_groups": ["R02"]},
    {"split_id": "fold-2", "train_groups": ["R01", "R02"], "test_groups": ["R03"]},
]


class LeakageValidationTests(unittest.TestCase):
    def test_grouped_folds_pass(self):
        report = validate_splits(VALID, GROUPS)
        self.assertTrue(report["every_group_tested_once"])

    def test_train_test_overlap_fails(self):
        bad = [{"split_id": "fold-0", "train_groups": ["R01", "R02"],
                "test_groups": ["R01", "R03"]}]
        with self.assertRaisesRegex(ValidationError, "LEAKAGE.*R01"):
            validate_splits(bad, GROUPS, require_unique_test=False)

    def test_unknown_group_fails(self):
        bad = [{"split_id": "fold-0", "train_groups": ["R01", "R02"],
                "test_groups": ["DOES-NOT-EXIST"]}]
        with self.assertRaisesRegex(ValidationError, "unknown groups"):
            validate_splits(bad, GROUPS, require_complete=False, require_unique_test=False)

    def test_omitted_group_fails(self):
        bad = [{"split_id": "fold-0", "train_groups": ["R01"], "test_groups": ["R02"]}]
        with self.assertRaisesRegex(ValidationError, "omits dataset groups.*R03"):
            validate_splits(bad, GROUPS, require_unique_test=False)

    def test_repeated_test_group_across_folds_fails(self):
        bad = VALID + [{"split_id": "fold-3", "train_groups": ["R02", "R03"],
                        "test_groups": ["R01"]}]
        with self.assertRaisesRegex(ValidationError, "repeated across folds.*R01"):
            validate_splits(bad, GROUPS)

    def test_duplicate_within_partition_fails(self):
        bad = [{"split_id": "fold-0", "train_groups": ["R02", "R02", "R03"],
                "test_groups": ["R01"]}]
        with self.assertRaisesRegex(ValidationError, "repeats train groups"):
            validate_splits(bad, GROUPS, require_unique_test=False)


class LeakageCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dataset = os.path.join(self.temp.name, "dataset.csv")
        with open(self.dataset, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["feature", "split_group"])
            writer.writerows([[1, "R01"], [2, "R01"], [3, "R02"], [4, "R03"]])

    def tearDown(self):
        self.temp.cleanup()

    def _write_json(self, name, splits):
        path = os.path.join(self.temp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"splits": splits}, fh)
        return path

    def test_json_cli_passes_and_writes_report(self):
        splits = self._write_json("valid.json", VALID)
        report = os.path.join(self.temp.name, "report.json")
        self.assertEqual(main(["--dataset", self.dataset, "--splits", splits,
                               "--report", report]), 0)
        with open(report, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["dataset_rows"], 4)

    def test_json_cli_returns_one_on_leakage(self):
        bad = [{"split_id": "fold-0", "train_groups": ["R01", "R02"],
                "test_groups": ["R01", "R03"]}]
        splits = self._write_json("bad.json", bad)
        self.assertEqual(main(["--dataset", self.dataset, "--splits", splits,
                               "--allow-repeated-test-groups"]), 1)

    def test_long_csv_format_loads(self):
        path = os.path.join(self.temp.name, "splits.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["split_id", "partition", "split_group"])
            writer.writerows([["fold-0", "train", "R02"],
                              ["fold-0", "train", "R03"],
                              ["fold-0", "test", "R01"]])
        loaded = load_splits(path)
        self.assertEqual(loaded[0]["test_groups"], ["R01"])

    def test_legacy_unresolved_pool_is_rejected(self):
        dataset = os.path.join(self.temp.name, "unresolved.csv")
        with open(dataset, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["split_group"])
            writer.writerows([["R01"], ["T-UNRESOLVED"]])
        splits = self._write_json(
            "unresolved.json",
            [{"split_id": "fold-0", "train_groups": ["R01"],
              "test_groups": ["T-UNRESOLVED"]}],
        )
        self.assertEqual(main(["--dataset", dataset, "--splits", splits,
                               "--allow-repeated-test-groups"]), 1)


if __name__ == "__main__":
    unittest.main()
