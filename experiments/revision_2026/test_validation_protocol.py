"""Dependency-light integrity tests for the section B protocol."""

import hashlib
import os
import tempfile
import unittest

import pandas as pd

from generate_grouped_splits import (
    SplitPlanningError,
    _record,
    infer_event_type,
    make_grouped_splits,
    validate_class_coverage,
)
from prepare_grouped_dataset import PreparationError, recompute_trace_deltas
from run_grouped_validation import GroupedRunError, verify_artifacts


class EventTypeTests(unittest.TestCase):
    def test_native_scenario_names(self):
        expected = {
            "SC-DETERMINISTIC_BURST-l100-b5": "SAG.DB",
            "SC-FULLY_RANDOMIZED-l15-b1": "FRG",
            "SC-RANDOMIC_BURST-l15-b5": "SAG.PB",
            "SC-RANDOMIC_MESSAGE-l15-b5": "SAG.PBM",
        }
        self.assertEqual({key: infer_event_type(key) for key in expected}, expected)

    def test_unknown_event_type_fails(self):
        with self.assertRaises(SplitPlanningError):
            infer_event_type("normal-traffic-only")


class ClassCoverageTests(unittest.TestCase):
    def test_closed_set_fold_passes(self):
        labels = {"R1": ["normal", "A"], "R2": ["normal", "A"]}
        rows = {"R1": 10, "R2": 12}
        fold = _record("fold-0", ["R1"], ["R2"], labels, rows)
        self.assertEqual(fold["test_only_labels"], [])
        self.assertEqual(validate_class_coverage([fold]), {})

    def test_unseen_test_class_blocks_standard_evaluation(self):
        labels = {"R1": ["normal", "A"], "R2": ["normal", "B"]}
        rows = {"R1": 10, "R2": 12}
        fold = _record("fold-0", ["R1"], ["R2"], labels, rows)
        with self.assertRaisesRegex(SplitPlanningError, "test labels absent"):
            validate_class_coverage([fold])
        diagnostic = validate_class_coverage([fold], allow_unseen_test_classes=True)
        self.assertEqual(diagnostic, {"fold-0": ["B"]})


class ArtifactBindingTests(unittest.TestCase):
    def test_dataset_is_bound_to_preparation_and_splits_by_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = os.path.join(temp, "prepared.csv")
            with open(dataset, "wb") as fh:
                fh.write(b"split_group,class\nR1,normal\n")
            with open(dataset, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            preparation = {"status": "pass", "output": os.path.abspath(dataset),
                           "output_sha256": digest}
            splits = {"dataset_sha256": digest, "open_set_diagnostic": False}
            self.assertEqual(verify_artifacts(dataset, preparation, splits), digest)

            with open(dataset, "ab") as fh:
                fh.write(b"R2,normal\n")
            with self.assertRaisesRegex(GroupedRunError, "hash differs"):
                verify_artifacts(dataset, preparation, splits)


class DeltaRecomputationTests(unittest.TestCase):
    @staticmethod
    def frame():
        # Deliberately interleaved; source order inside each trace is valid.
        return pd.DataFrame([
            {"trace_id": "A", "GooseTimestamp": 1.0, "StNum": 1, "SqNum": 4,
             "gooseLen": 100, "APDUSize": 50, "frameLen": 120,
             "cbStatus": 0, "t": 1.0, "batch_index": 1},
            {"trace_id": "B", "GooseTimestamp": 3.0, "StNum": 9, "SqNum": 1,
             "gooseLen": 190, "APDUSize": 80, "frameLen": 220,
             "cbStatus": 0, "t": 3.0, "batch_index": 1},
            {"trace_id": "A", "GooseTimestamp": 2.0, "StNum": 2, "SqNum": 0,
             "gooseLen": 110, "APDUSize": 60, "frameLen": 130,
             "cbStatus": 1, "t": 1.5, "batch_index": 1},
            {"trace_id": "B", "GooseTimestamp": 5.0, "StNum": 10, "SqNum": 2,
             "gooseLen": 200, "APDUSize": 90, "frameLen": 230,
             "cbStatus": 0, "t": 4.0, "batch_index": 1},
        ])

    def test_deltas_never_cross_trace_boundaries(self):
        result, audit = recompute_trace_deltas(self.frame())
        self.assertEqual(audit["boundary_rows"], 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["trace_id"].tolist(), ["A", "B"])
        self.assertEqual(result["stDiff"].tolist(), [1.0, 1.0])
        self.assertEqual(result["sqDiff"].tolist(), [-4.0, 1.0])
        self.assertEqual(result["timestampDiff"].tolist(), [1.0, 2.0])
        self.assertEqual(result["cbStatusDiff"].tolist(), [1.0, 0.0])
        self.assertEqual(result["timeFromLastChange"].tolist(), [0.5, 1.0])

    def test_unresolved_trace_is_rejected(self):
        frame = self.frame()
        frame.loc[0, "trace_id"] = "T-UNRESOLVED"
        with self.assertRaisesRegex(PreparationError, "non-independent"):
            recompute_trace_deltas(frame)


class GroupedSplitterTests(unittest.TestCase):
    def test_stratified_group_kfold_keeps_runs_intact(self):
        rows = []
        scenarios = {
            "DB1": "SC-DETERMINISTIC_BURST-l100-b5",
            "DB2": "SC-DETERMINISTIC_BURST-l100-b5",
            "FR1": "SC-FULLY_RANDOMIZED-l15-b1",
            "FR2": "SC-FULLY_RANDOMIZED-l15-b1",
            "PB1": "SC-RANDOMIC_BURST-l15-b5",
            "PB2": "SC-RANDOMIC_BURST-l15-b5",
        }
        attack = {"DB": "DB_ATTACK", "FR": "FR_ATTACK", "PB": "PB_ATTACK"}
        for group, scenario in scenarios.items():
            family = group[:2]
            rows.extend([
                {"split_group": group, "class": "normal", "scenario_id": scenario},
                {"split_group": group, "class": attack[family], "scenario_id": scenario},
            ] * 5)
        frame = pd.DataFrame(rows)
        records, groups, _ = make_grouped_splits(
            frame, "stratified-group-kfold", 2, 42
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(set(groups), set(scenarios))
        for record in records:
            self.assertFalse(set(record["train_groups"]) & set(record["test_groups"]))
            self.assertEqual(record["test_only_labels"], [])

    def test_open_set_splits_are_rejected_by_standard_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = os.path.join(temp, "prepared.csv")
            with open(dataset, "wb") as fh:
                fh.write(b"x")
            digest = hashlib.sha256(b"x").hexdigest()
            preparation = {"status": "pass", "output": os.path.abspath(dataset),
                           "output_sha256": digest}
            splits = {"dataset_sha256": digest, "open_set_diagnostic": True}
            with self.assertRaisesRegex(GroupedRunError, "open-set"):
                verify_artifacts(dataset, preparation, splits)


if __name__ == "__main__":
    unittest.main()
