"""Pipeline-plumbing tests for card C (benign-degradation controls).

Checklist ref.: C3/C4 in benign_controls.md SS6/SS7. C3's tests cover the four
constraints it is responsible for:

  - `benign_degradation` is a known class (VARIANT_OF_CLASS), so
    `add_experiment_metadata.py` no longer raises on it.
  - `SC-BENIGN_*` scenario ids resolve to their own `infer_event_type`
    families instead of raising SplitPlanningError.
  - `impairment_mode`/`impairment_rate`/`impairment_intensity_ms` never reach
    the model feature matrix (IDENTIFIER_COLUMNS) and are excluded from
    `merge_runs.py`'s payload fingerprint the same way loss_rate/burst_size
    are.
  - Because every benign run carries both `normal` and `benign_degradation`,
    leave-one-event-type-out over the benign families is a closed-set fold
    (unlike LOETO over attack variants, which is open-set - see
    validation_protocol.md, "Leave-one-event-type-out status").

C4's tests cover `benign_confusion_report.py`: the 6x6 confusion matrix
gracefully reduces when classes are absent, the per-mode breakdown and
attack_fpr/alert_rate are computed correctly, ideal-normal vs.
baseline-normal-inside-a-benign-run are kept separate, and the dataset
rejoin is hash-bound.
"""

import hashlib
import os
import tempfile
import unittest

import pandas as pd

from add_experiment_metadata import BENIGN_DEGRADATION_LABEL, VARIANT_OF_CLASS, load_raw
from benign_confusion_report import (
    ATTACK_CLASSES,
    ConfusionReportError,
    alert_stats,
    build_confusion_matrix,
    display_name,
    load_impairment_modes,
    normal_baselines,
    per_mode_breakdown,
)
from generate_grouped_splits import (
    SplitPlanningError,
    infer_event_type,
    make_grouped_splits,
    validate_class_coverage,
)
from merge_runs import payload_fingerprint
from run_grouped_validation import IDENTIFIER_COLUMNS, feature_matrix


class ClassVocabularyTests(unittest.TestCase):
    def test_benign_degradation_is_a_known_class(self):
        self.assertIn(BENIGN_DEGRADATION_LABEL, VARIANT_OF_CLASS)
        # Not an attack: attack_variant must read "none" for it, exactly like
        # `normal` - impairment_mode carries the mechanism instead.
        self.assertEqual(VARIANT_OF_CLASS[BENIGN_DEGRADATION_LABEL], "none")

    def test_load_raw_accepts_benign_degradation_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "mini.csv")
            pd.DataFrame([
                {"StNum": 1, "SqNum": 0, "t": 1.0, "Time": 0.0, "class": "normal"},
                {"StNum": 2, "SqNum": 0, "t": 2.0, "Time": 1.0, "class": BENIGN_DEGRADATION_LABEL},
            ]).to_csv(path, index=False)
            # Would raise ValueError("Unknown class labels ...") before C3.
            df = load_raw(path)
            self.assertEqual(len(df), 2)


class EventTypeTests(unittest.TestCase):
    def test_benign_scenario_ids_get_their_own_family(self):
        expected = {
            "SC-BENIGN_CONGESTION_LOSS-l5-b1": "BENIGN.CONGESTION_LOSS",
            "SC-BENIGN_QUEUE_OVERLOAD_BURST-l15-b3": "BENIGN.QUEUE_OVERLOAD_BURST",
            "SC-BENIGN_JITTER-l15-b5": "BENIGN.JITTER",
            "SC-BENIGN_DELAY-l15-b5": "BENIGN.DELAY",
            "SC-BENIGN_LINK_FLAP-l100-b3": "BENIGN.LINK_FLAP",
            "SC-BENIGN_DUPLICATION-l5-b1": "BENIGN.DUPLICATION",
            "SC-BENIGN_REORDERING-l5-b1": "BENIGN.REORDERING",
        }
        self.assertEqual({key: infer_event_type(key) for key in expected}, expected)

    def test_benign_families_do_not_collide_with_attack_families(self):
        attack = {infer_event_type(s) for s in [
            "SC-DETERMINISTIC_BURST-l100-b5", "SC-FULLY_RANDOMIZED-l15-b1",
            "SC-RANDOMIC_BURST-l15-b5", "SC-RANDOMIC_MESSAGE-l15-b5",
        ]}
        benign = {infer_event_type(s) for s in [
            "SC-BENIGN_CONGESTION_LOSS-l5-b1", "SC-BENIGN_QUEUE_OVERLOAD_BURST-l15-b3",
            "SC-BENIGN_JITTER-l15-b5", "SC-BENIGN_DELAY-l15-b5",
            "SC-BENIGN_LINK_FLAP-l100-b3", "SC-BENIGN_DUPLICATION-l5-b1",
            "SC-BENIGN_REORDERING-l5-b1",
        ]}
        self.assertEqual(len(benign), 7)
        self.assertTrue(attack.isdisjoint(benign))

    def test_unknown_scenario_still_fails(self):
        with self.assertRaises(SplitPlanningError):
            infer_event_type("normal-traffic-only")


class IdentifierLeakageTests(unittest.TestCase):
    def test_impairment_columns_are_identifiers_not_features(self):
        for column in ("impairment_mode", "impairment_rate", "impairment_intensity_ms"):
            self.assertIn(column, IDENTIFIER_COLUMNS)

    def test_impairment_mode_never_reaches_the_feature_matrix(self):
        frame = pd.DataFrame([
            {"class": "normal", "impairment_mode": "NONE", "impairment_rate": 0.0,
             "impairment_intensity_ms": 0.0, "run_id": "R1", "gooseLen": 100},
            {"class": BENIGN_DEGRADATION_LABEL, "impairment_mode": "JITTER",
             "impairment_rate": 0.0, "impairment_intensity_ms": 5.0,
             "run_id": "R1", "gooseLen": 105},
        ])
        # impairment_mode is a string column: if it survived the discard, this
        # would raise "non-numeric feature columns remain" instead of just
        # dropping the leak silently - either way the leak would be caught,
        # but this confirms the intended (clean-drop) path.
        features = feature_matrix(frame, target_column="class", extra_discard=[])
        self.assertNotIn("impairment_mode", features.columns)
        self.assertNotIn("impairment_rate", features.columns)
        self.assertNotIn("impairment_intensity_ms", features.columns)
        self.assertIn("gooseLen", features.columns)


class PayloadFingerprintTests(unittest.TestCase):
    def test_impairment_columns_excluded_from_payload_hash(self):
        # Two runs whose actual messages are identical but whose benign
        # provenance differs. If impairment_mode/rate/intensity leaked into
        # the fingerprinted payload, they would trivially differ and mask the
        # exact bug this check exists to catch (see merge_runs.py docstring).
        base = {
            "run_id": "BENIGN_JITTER-j5-s1", "trace_id": "T1", "batch_index": 1,
            "scenario_id": "SC-BENIGN_JITTER-j5", "seed": 1, "attack_variant": "NONE",
            "loss_rate": 0.0, "burst_size": 1, "traffic_rate": 1.0,
            "substation_config": "SUB-A",
        }
        df_a = pd.DataFrame([{**base, "impairment_mode": "JITTER",
                               "impairment_rate": 0.0, "impairment_intensity_ms": 5.0,
                               "gooseLen": 100}])
        df_b = pd.DataFrame([{**base, "impairment_mode": "DELAY",
                               "impairment_rate": 0.0, "impairment_intensity_ms": 25.0,
                               "gooseLen": 100}])
        self.assertEqual(payload_fingerprint(df_a), payload_fingerprint(df_b))


class LoetoClosedSetTests(unittest.TestCase):
    @staticmethod
    def benign_frame():
        modes = ["CONGESTION_LOSS", "QUEUE_OVERLOAD_BURST", "JITTER", "DELAY"]
        rows = []
        for i, mode in enumerate(modes):
            group = "BENIGN_%s-s1" % mode
            scenario = "SC-BENIGN_%s-l15-b5" % mode
            rows.extend([
                {"split_group": group, "class": "normal", "scenario_id": scenario},
                {"split_group": group, "class": BENIGN_DEGRADATION_LABEL, "scenario_id": scenario},
            ] * 5)
        return pd.DataFrame(rows)

    def test_benign_loeto_is_closed_set(self):
        records, groups, event_types = make_grouped_splits(
            self.benign_frame(), "leave-one-event-type-out", n_splits=None, seed=None
        )
        self.assertEqual(len(records), 4)  # one fold per impairment mode
        self.assertEqual(len(set(event_types.values())), 4)
        for record in records:
            # Every fold holds out one mechanism, but both `normal` and
            # `benign_degradation` remain in training via the other three -
            # unlike an attack-variant LOETO fold, this is not open-set.
            self.assertEqual(record["test_only_labels"], [])
        # Confirms the same thing from the caller's side: no
        # --allow-unseen-test-classes escape hatch is needed here.
        self.assertEqual(validate_class_coverage(records), {})

    def test_attack_variant_loeto_would_have_been_open_set(self):
        # Contrast case, mirroring validation_protocol.md's documented
        # attack-side finding: holding out one attack variant's runs removes
        # that class from training entirely, because (today) event type and
        # attack class are the same axis for attack runs.
        rows = []
        variants = {
            "DB": "SC-DETERMINISTIC_BURST-l100-b5",
            "FR": "SC-FULLY_RANDOMIZED-l15-b1",
            "PB": "SC-RANDOMIC_BURST-l15-b5",
        }
        for i, (family, scenario) in enumerate(variants.items()):
            group = "%s1" % family
            rows.extend([
                {"split_group": group, "class": "normal", "scenario_id": scenario},
                {"split_group": group, "class": family + "_ATTACK", "scenario_id": scenario},
            ] * 5)
        records, _, _ = make_grouped_splits(
            pd.DataFrame(rows), "leave-one-event-type-out", n_splits=None, seed=None
        )
        self.assertTrue(any(record["test_only_labels"] for record in records))
        with self.assertRaisesRegex(SplitPlanningError, "test labels absent"):
            validate_class_coverage(records)


DB = "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE"
FR = "FULLY_RANDOMIZED_ORIENTEDGRAYHOLE"


class ConfusionMatrixTests(unittest.TestCase):
    def test_present_classes_ordered_and_counted(self):
        frame = pd.DataFrame({
            "y_true": [BENIGN_DEGRADATION_LABEL, BENIGN_DEGRADATION_LABEL, "normal", DB, DB],
            "y_pred": [BENIGN_DEGRADATION_LABEL, DB, "normal", DB, "normal"],
        })
        present, counts = build_confusion_matrix(frame)
        # normal, benign_degradation first (checklist C.3 axis), then attacks.
        self.assertEqual(present, ["normal", BENIGN_DEGRADATION_LABEL, DB])
        self.assertEqual(counts[BENIGN_DEGRADATION_LABEL][BENIGN_DEGRADATION_LABEL], 1)
        self.assertEqual(counts[BENIGN_DEGRADATION_LABEL][DB], 1)
        self.assertEqual(counts[DB]["normal"], 1)
        self.assertEqual(counts[DB][DB], 1)

    def test_partial_dataset_does_not_hallucinate_absent_classes(self):
        # A benign-only pool (no attack rows at all, like the real C3 smoke).
        frame = pd.DataFrame({
            "y_true": ["normal", BENIGN_DEGRADATION_LABEL],
            "y_pred": ["normal", BENIGN_DEGRADATION_LABEL],
        })
        present, _ = build_confusion_matrix(frame)
        self.assertEqual(present, ["normal", BENIGN_DEGRADATION_LABEL])
        self.assertNotIn(DB, present)


class DisplayNameTests(unittest.TestCase):
    def test_normal_and_benign_keep_their_own_names(self):
        # Not VARIANT_OF_CLASS's "none" reading - that would collapse both
        # into one label and defeat the point of this report.
        self.assertEqual(display_name("normal"), "normal")
        self.assertEqual(display_name(BENIGN_DEGRADATION_LABEL), BENIGN_DEGRADATION_LABEL)

    def test_attack_classes_get_paper_short_names(self):
        self.assertEqual(display_name(DB), "SAG.DB")
        self.assertIn(DB, ATTACK_CLASSES)
        self.assertNotIn("normal", ATTACK_CLASSES)
        self.assertNotIn(BENIGN_DEGRADATION_LABEL, ATTACK_CLASSES)


class AlertStatsTests(unittest.TestCase):
    def test_attack_fpr_and_alert_rate(self):
        preds = pd.Series([BENIGN_DEGRADATION_LABEL, DB, "normal", "normal"])
        stats = alert_stats(preds)
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["attack_fpr"], 0.25)  # only the DB prediction
        self.assertAlmostEqual(stats["alert_rate"], 0.5)  # DB + benign_degradation

    def test_empty_slice_is_reported_as_na_not_zero(self):
        stats = alert_stats(pd.Series([], dtype=object))
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["attack_fpr"])
        self.assertIsNone(stats["alert_rate"])


class PerModeAndBaselineTests(unittest.TestCase):
    @staticmethod
    def frame():
        rows = []
        # Ideal normal traffic (impairment_mode=NONE): 7 correct, 1 false
        # attack alarm on genuinely clean traffic.
        rows += [{"y_true": "normal", "y_pred": "normal", "impairment_mode": "NONE"}] * 7
        rows += [{"y_true": "normal", "y_pred": DB, "impairment_mode": "NONE"}]
        # Baseline normal messages captured inside a JITTER run: still
        # `normal`, but impairment_mode is run-level and non-NONE.
        rows += [{"y_true": "normal", "y_pred": "normal", "impairment_mode": "JITTER"}] * 5
        rows += [{"y_true": "normal", "y_pred": BENIGN_DEGRADATION_LABEL, "impairment_mode": "JITTER"}]
        # CONGESTION_LOSS benign_degradation rows: 6 correct, 3 confused with
        # normal, 1 flipped into the attack it is paired against.
        rows += [{"y_true": BENIGN_DEGRADATION_LABEL, "y_pred": BENIGN_DEGRADATION_LABEL,
                  "impairment_mode": "CONGESTION_LOSS"}] * 6
        rows += [{"y_true": BENIGN_DEGRADATION_LABEL, "y_pred": "normal",
                  "impairment_mode": "CONGESTION_LOSS"}] * 3
        rows += [{"y_true": BENIGN_DEGRADATION_LABEL, "y_pred": DB, "impairment_mode": "CONGESTION_LOSS"}]
        # JITTER benign_degradation rows: all correctly identified.
        rows += [{"y_true": BENIGN_DEGRADATION_LABEL, "y_pred": BENIGN_DEGRADATION_LABEL,
                  "impairment_mode": "JITTER"}] * 5
        return pd.DataFrame(rows)

    def test_per_mode_attack_fpr_and_alert_rate(self):
        frame = self.frame()
        present = ["normal", BENIGN_DEGRADATION_LABEL, DB]
        rows = {r["mode"]: r for r in per_mode_breakdown(frame, present)}

        self.assertEqual(set(rows), {"CONGESTION_LOSS", "JITTER"})

        congestion = rows["CONGESTION_LOSS"]
        self.assertEqual(congestion["n"], 10)
        self.assertAlmostEqual(congestion["attack_fpr"], 0.1)
        self.assertAlmostEqual(congestion["alert_rate"], 0.7)  # 6 correct + 1 attack, 3 missed as normal
        self.assertEqual(congestion["outcomes"][BENIGN_DEGRADATION_LABEL], 6)
        self.assertEqual(congestion["outcomes"]["normal"], 3)
        self.assertEqual(congestion["outcomes"][DB], 1)

        jitter = rows["JITTER"]
        self.assertEqual(jitter["n"], 5)
        self.assertAlmostEqual(jitter["attack_fpr"], 0.0)
        self.assertAlmostEqual(jitter["alert_rate"], 1.0)

    def test_ideal_normal_and_in_run_baseline_are_kept_separate(self):
        baselines = normal_baselines(self.frame())
        ideal = baselines["ideal_normal"]
        self.assertEqual(ideal["n"], 8)
        self.assertAlmostEqual(ideal["attack_fpr"], 0.125)
        self.assertAlmostEqual(ideal["alert_rate"], 0.125)

        in_run = baselines["baseline_in_benign_run"]
        self.assertEqual(in_run["n"], 6)
        self.assertAlmostEqual(in_run["attack_fpr"], 0.0)
        self.assertAlmostEqual(in_run["alert_rate"], 1 / 6)


class ImpairmentRejoinTests(unittest.TestCase):
    def test_rejoin_is_hash_bound_and_positional(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "prepared.csv")
            pd.DataFrame({
                "impairment_mode": ["NONE", "JITTER", "NONE", "DELAY", "JITTER"],
            }).to_csv(path, index=False)
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()

            result = load_impairment_modes(path, digest, pd.Series([4, 0, 2]))
            self.assertEqual(list(result), ["JITTER", "NONE", "NONE"])
            self.assertEqual(list(result.index), [0, 1, 2])  # ready to assign back positionally

            with self.assertRaisesRegex(ConfusionReportError, "hash differs"):
                load_impairment_modes(path, "0" * 64, pd.Series([0]))

    def test_out_of_range_row_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "prepared.csv")
            pd.DataFrame({"impairment_mode": ["NONE", "JITTER"]}).to_csv(path, index=False)
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            with self.assertRaisesRegex(ConfusionReportError, "wrong dataset"):
                load_impairment_modes(path, digest, pd.Series([5]))


if __name__ == "__main__":
    unittest.main()
