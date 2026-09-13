"""Tests for the label/feature duplication audit.

What these defend: the audit has to separate three things that look alike in
a table but mean different things.

  - A message written twice with the *same* label is redundancy. Wasteful,
    not a correctness problem, and must not be reported as a conflict.
  - A message written twice with *different* labels is the defect: the label
    stops being a function of the features, and the content comparison has to
    see it even though the delta columns differ on the copy by construction.
  - Two rows identical on *every* model feature including the deltas are
    irreducible - no classifier can separate them - and are counted apart from
    the merely content-identical ones.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import check_label_duplication as audit
from check_label_duplication import (
    DuplicationError,
    audit_frame,
    column_sets,
    feature_block,
    summarise,
    twin_counts,
    varying_columns,
)

KEY = ("StNum", "SqNum", "GooseTimestamp", "t")
ATTACK = "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE"


def message(st, sq, timestamp, label, sq_diff=1.0, isb=0.5, eth="01:0c:cd:01:2f:78"):
    """One raw-ERENO-shaped row: content columns, a delta column, provenance."""
    return {
        "StNum": st, "SqNum": sq, "GooseTimestamp": timestamp, "t": 0.0166,
        "isbA": isb, "frameLen": 200, "ethSrc": eth,
        "sqDiff": sq_diff, "stDiff": 0.0, "tDiff": 0.0, "timestampDiff": 1.0,
        "gooseLengthDiff": 0.0, "cbStatusDiff": 0.0, "apduSizeDiff": 0.0,
        "frameLengthDiff": 0.0,
        "class": label, "run_id": "R1", "batch_index": 0,
    }


def frame_of(rows):
    return pd.DataFrame(rows)


def run_audit(frame):
    features, content = column_sets(frame.columns, "class")
    return audit_frame(frame, "class", KEY, content, features)


class CleanDataTests(unittest.TestCase):
    def test_distinct_messages_produce_no_findings(self):
        result = run_audit(frame_of([
            message(1, 1, 1.0, "normal"),
            message(1, 2, 2.0, "normal"),
            message(1, 3, 3.0, ATTACK),
        ]))
        self.assertEqual(result["content_twins"], 0)
        self.assertEqual(result["content_conflicting_rows"], 0)
        self.assertEqual(result["keys_with_conflicting_labels"], 0)
        self.assertEqual(result["attack_rows"], 1)

    def test_a_message_repeated_under_the_same_label_is_not_a_conflict(self):
        """Redundancy, not a defect - and the audit must not conflate them."""
        result = run_audit(frame_of([
            message(1, 1, 1.0, "normal", sq_diff=1.0),
            message(1, 1, 1.0, "normal", sq_diff=0.0),
            message(1, 2, 2.0, ATTACK),
        ]))
        self.assertEqual(result["rows_on_duplicated_keys"], 2)
        self.assertEqual(result["keys_with_conflicting_labels"], 0)
        self.assertEqual(result["content_twins"], 0)
        self.assertEqual(result["content_conflicting_rows"], 0)


class ConflictDetectionTests(unittest.TestCase):
    def test_the_real_defect_is_caught_through_the_delta_columns(self):
        """The copies differ in `sqDiff` by construction; the content comparison
        has to look past that or the defect is invisible."""
        result = run_audit(frame_of([
            message(1, 1, 1.0, ATTACK, sq_diff=1.0),
            message(1, 1, 1.0, "normal", sq_diff=0.0),
            message(1, 2, 2.0, "normal"),
        ]))
        self.assertEqual(result["attack_rows"], 1)
        self.assertEqual(result["content_twins"], 1)
        self.assertEqual(result["content_conflicting_rows"], 2)
        self.assertEqual(result["keys_with_conflicting_labels"], 1)
        self.assertEqual(result["attack_rows_sharing_a_key"], 1)
        # The deltas do separate them, so this pair is not irreducible.
        self.assertEqual(result["full_twins"], 0)

    def test_identical_deltas_make_the_pair_irreducible(self):
        result = run_audit(frame_of([
            message(1, 1, 1.0, ATTACK, sq_diff=1.0),
            message(1, 1, 1.0, "normal", sq_diff=1.0),
        ]))
        self.assertEqual(result["content_twins"], 1)
        self.assertEqual(result["full_twins"], 1)

    def test_a_benign_degradation_conflict_is_reported_even_though_it_is_not_an_attack(self):
        """Card C's version of the same defect, which the attack-only count misses."""
        result = run_audit(frame_of([
            message(1, 1, 1.0, "benign_degradation", sq_diff=1.0),
            message(1, 1, 1.0, "normal", sq_diff=0.0),
        ]))
        self.assertEqual(result["attack_rows"], 0)
        self.assertEqual(result["content_twins"], 0)
        self.assertEqual(result["content_conflicting_rows"], 2)

    def test_string_columns_participate_in_identity(self):
        """Two rows differing only in a MAC address are not twins."""
        result = run_audit(frame_of([
            message(1, 1, 1.0, ATTACK, eth="01:0c:cd:01:2f:78"),
            message(1, 1, 1.0, "normal", eth="aa:bb:cc:dd:ee:ff"),
        ]))
        self.assertEqual(result["content_twins"], 0)


class ColumnSetTests(unittest.TestCase):
    def test_provenance_and_label_are_never_compared(self):
        features, content = column_sets(
            ["isbA", "sqDiff", "class", "run_id", "seed", "split_group"], "class")
        self.assertEqual(features, ["isbA", "sqDiff"])
        self.assertEqual(content, ["isbA"])

    def test_a_frame_with_nothing_to_compare_is_an_error(self):
        with self.assertRaises(DuplicationError):
            column_sets(["class", "run_id", "sqDiff"], "class")

    def test_feature_block_mixes_dtypes_without_losing_identity(self):
        frame = frame_of([message(1, 1, 1.0, "normal", eth="a"),
                          message(1, 1, 1.0, "normal", eth="b")])
        block = feature_block(frame, ["isbA", "ethSrc"])
        self.assertEqual(block.shape, (2, 2))
        self.assertNotEqual(block[0, 1], block[1, 1])


class DiagnosticTests(unittest.TestCase):
    def test_varying_columns_names_exactly_what_differs(self):
        frame = frame_of([
            message(1, 1, 1.0, ATTACK, sq_diff=1.0),
            message(1, 1, 1.0, "normal", sq_diff=0.0),
            message(1, 2, 2.0, "normal"),
        ])
        varying, keys = varying_columns(frame, KEY)
        self.assertEqual(keys, 1)
        self.assertEqual(set(varying), {"sqDiff", "class"})
        self.assertEqual(varying["class"], 1)

    def test_no_duplicates_means_no_diagnostic(self):
        frame = frame_of([message(1, 1, 1.0, "normal"), message(1, 2, 2.0, "normal")])
        varying, keys = varying_columns(frame, KEY)
        self.assertEqual((varying, keys), ({}, 0))


class TwinCountTests(unittest.TestCase):
    def test_conflicting_count_needs_labels_and_is_zero_without_them(self):
        values = np.array([[1.0], [1.0]])
        is_attack = np.array([True, False])
        attack, conflicting = twin_counts(values, is_attack)
        self.assertEqual((attack, conflicting), (1, 0))
        attack, conflicting = twin_counts(
            values, is_attack, np.array([ATTACK, "normal"]))
        self.assertEqual((attack, conflicting), (1, 2))

    def test_empty_input_is_not_an_error(self):
        self.assertEqual(twin_counts(np.empty((0, 3)), np.array([], dtype=bool)), (0, 0))


class SummaryTests(unittest.TestCase):
    def test_rates_are_taken_over_the_right_denominators(self):
        totals = summarise({
            "a": {"rows": 100, "attack_rows": 10, "content_twins": 8, "full_twins": 1,
                  "content_conflicting_rows": 16, "full_conflicting_rows": 2,
                  "attack_rows_sharing_a_key": 8, "rows_on_duplicated_keys": 50},
            "b": {"rows": 100, "attack_rows": 10, "content_twins": 2, "full_twins": 0,
                  "content_conflicting_rows": 4, "full_conflicting_rows": 0,
                  "attack_rows_sharing_a_key": 2, "rows_on_duplicated_keys": 50},
        })
        self.assertEqual(totals["runs"], 2)
        self.assertAlmostEqual(totals["content_twin_rate"], 10 / 20)
        self.assertAlmostEqual(totals["duplicate_row_rate"], 100 / 200)
        self.assertAlmostEqual(totals["content_conflict_rate"], 20 / 200)


class CliTests(unittest.TestCase):
    def _write_run(self, directory, name, rows):
        path = os.path.join(directory, name + ".csv")
        frame_of(rows).to_csv(path, index=False)
        return path

    def test_a_clean_runs_directory_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_run(tmp, "clean", [message(1, 1, 1.0, "normal"),
                                           message(1, 2, 2.0, ATTACK)])
            code = audit.main(["--runs-dir", tmp])
        self.assertEqual(code, 0)

    def test_a_conflicting_runs_directory_exits_one_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_run(tmp, "broken", [
                message(1, 1, 1.0, ATTACK, sq_diff=1.0),
                message(1, 1, 1.0, "normal", sq_diff=0.0),
            ])
            report = os.path.join(tmp, "out.json")
            code = audit.main(["--runs-dir", tmp, "--report", report,
                               "--out", os.path.join(tmp, "out.md")])
            with open(report, encoding="utf-8") as fh:
                payload = json.load(fh)
        self.assertEqual(code, 1)
        self.assertEqual(payload["totals"]["content_twins"], 1)
        self.assertEqual(payload["diagnostics"]["columns_that_differ_between_copies"],
                         {"sqDiff": 1, "class": 1})

    def test_an_empty_directory_is_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(audit.main(["--runs-dir", tmp]), 2)


if __name__ == "__main__":
    unittest.main()
