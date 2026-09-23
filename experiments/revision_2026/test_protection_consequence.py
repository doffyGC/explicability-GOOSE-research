"""Tests for the transfer-trip consequence model.

What these defend: the dataset only holds delivered frames, so every number
this script reports rests on reconstructing the events an attack hid. The
tests pin that reconstruction on synthetic traces shaped like ERENO's - a
recovery-type initial state, then fault/recovery pairs 100 ms apart, each
announced by retransmissions at +0, +100 and +200 ms - and check each way an
event can be lost:

  - nothing lost: no missed trip, zero latency, relative energy exactly 1;
  - the whole fault state lost: a missed trip, timed from the recovery;
  - only the fault's first frame lost: delivered late, not missed, but
    breaking both transfer-time limits;
  - fault and recovery both lost: unobservable, excluded rather than guessed;
  - no fault frame anywhere in the trace: the parity still comes out right.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import protection_consequence as pc

T_RELAY, T_BREAKER, T_BACKUP = 0.020, 0.050, 0.400
IDEAL = T_RELAY + T_BREAKER


def trace(run_id, events=3, first_st=1, drop=(), trace_id=None, burst=3, loss=100.0):
    """Frames of one trace. `drop` holds (StNum, SqNum) pairs to remove."""
    rows = []
    t0 = 0.01659
    trace_id = trace_id or run_id
    # initial recovery-type state: heartbeats only
    for sq in range(1, 4):
        rows.append((first_st, 0, t0, t0 + sq - 1.0, sq))
    st = first_st
    base = 10.0
    for k in range(events):
        fault_t = base + 10.0 * k + 0.50631
        rec_t = fault_t + pc.RECOVERY_OFFSET_S
        st += 1
        for sq, dt in enumerate((0.0, 0.1, 0.2), start=1):
            rows.append((st, 1, fault_t, fault_t + dt, sq))
        st += 1
        for sq, dt in enumerate((0.0, 0.1, 0.2, 0.833, 1.833), start=1):
            rows.append((st, 0, rec_t, rec_t + dt, sq))
    frame = pd.DataFrame(rows, columns=["StNum", "cbStatus", "t", "GooseTimestamp", "SqNum"])
    keep = ~frame.apply(lambda r: (r["StNum"], r["SqNum"]) in set(drop), axis=1)
    frame = frame[keep].copy()
    frame["run_id"] = run_id
    frame["trace_id"] = trace_id
    frame["burst_size"] = burst
    frame["loss_rate"] = loss
    return frame


def scored_events(frame):
    events = pc.fault_events(pc.state_table(frame))
    return pc.score_events(events, T_RELAY, T_BREAKER, T_BACKUP)


class EventReconstructionTests(unittest.TestCase):

    def test_ideal_trace_has_no_missed_trip_and_unit_energy(self):
        ev = scored_events(trace("DETERMINISTIC_BURST-l100-b3-s1"))
        # the last state is excluded, so the final fault (StNum 6, recovery 7)
        # counts and events are StNum 2, 4, 6
        self.assertEqual(list(ev["StNum"]), [2, 4, 6])
        self.assertFalse(ev["missed"].any())
        np.testing.assert_allclose(ev["latency"], 0.0, atol=1e-12)
        np.testing.assert_allclose(ev["relative_energy"], 1.0)
        self.assertFalse(ev["violates_3ms"].any())

    def test_whole_fault_state_dropped_is_a_missed_trip_timed_from_recovery(self):
        frame = trace("X-s1", drop=[(4, 1), (4, 2), (4, 3)])
        ev = scored_events(frame).set_index("StNum")
        self.assertTrue(ev.loc[4, "missed"])
        self.assertTrue(ev.loc[4, "observable"])
        self.assertAlmostEqual(ev.loc[4, "t_f"], 20.50631, places=9)
        self.assertAlmostEqual(ev.loc[4, "relative_energy"], (T_BACKUP + T_BREAKER) / IDEAL)
        self.assertTrue(ev.loc[4, "violates_3ms"] and ev.loc[4, "violates_10ms"])

    def test_first_frame_dropped_is_late_not_missed(self):
        ev = scored_events(trace("X-s1", drop=[(2, 1)])).set_index("StNum")
        self.assertFalse(ev.loc[2, "missed"])
        self.assertAlmostEqual(ev.loc[2, "latency"], 0.1, places=9)
        self.assertTrue(ev.loc[2, "violates_3ms"] and ev.loc[2, "violates_10ms"])
        self.assertAlmostEqual(ev.loc[2, "relative_energy"], (IDEAL + 0.1) / IDEAL)

    def test_fault_and_recovery_both_lost_is_unobservable_and_excluded(self):
        drop = [(4, s) for s in range(1, 4)] + [(5, s) for s in range(1, 6)]
        scored = scored_events(trace("X-s1", drop=drop))
        ev = scored.set_index("StNum")
        self.assertFalse(ev.loc[4, "observable"])
        self.assertFalse(ev.loc[4, "missed"])
        runs = pc.per_run(scored)
        self.assertEqual(int(runs.loc["X-s1", "events"]), 2)
        self.assertEqual(int(runs.loc["X-s1", "unobservable"]), 1)

    def test_parity_without_any_surviving_fault_frame(self):
        drop = [(st, s) for st in (2, 4, 6) for s in range(1, 4)]
        ev = scored_events(trace("DB-s1", drop=drop))
        self.assertEqual(list(ev["StNum"]), [2, 4, 6])
        self.assertTrue(ev["missed"].all())

    def test_parity_follows_an_offset_initial_state(self):
        ev = scored_events(trace("X-s1", first_st=2))
        self.assertEqual(list(ev["StNum"]), [3, 5, 7])
        self.assertFalse(ev["missed"].any())

    def test_family_comes_from_the_run_id(self):
        self.assertEqual(pc.family_of("DETERMINISTIC_BURST-l100-b3-s20260101"),
                         "DETERMINISTIC_BURST")
        self.assertEqual(pc.family_of("BENIGN_LINK_FLAP-b5-s20260101"), "BENIGN_LINK_FLAP")

    def test_missing_column_is_refused(self):
        with self.assertRaises(pc.ConsequenceError):
            pc.state_table(trace("X-s1").drop(columns=["t"]))


class AggregationTests(unittest.TestCase):

    def frames(self):
        return pd.concat([
            trace("DETERMINISTIC_BURST-l100-b3-s1",
                  drop=[(st, s) for st in (2, 4, 6) for s in range(1, 4)]),
            trace("DETERMINISTIC_BURST-l100-b3-s2",
                  drop=[(st, s) for st in (2, 4, 6) for s in range(1, 4)]),
            trace("FULLY_RANDOMIZED-l5-s1", burst=1, loss=5.0),
            trace("FULLY_RANDOMIZED-l5-s2", burst=1, loss=5.0, drop=[(2, 1)]),
        ], ignore_index=True)

    def test_family_rates_are_pooled_over_events(self):
        result = pc.analyse(self.frames(), iterations=200, seed=1, sensitivity=(0.25,))
        fam = {r["family"]: r for r in result["by_family"]}
        self.assertEqual(fam["DETERMINISTIC_BURST"]["missed_rate"], 1.0)
        self.assertAlmostEqual(fam["DETERMINISTIC_BURST"]["relative_energy"],
                               (T_BACKUP + T_BREAKER) / IDEAL)
        self.assertEqual(fam["FULLY_RANDOMIZED"]["missed_rate"], 0.0)
        self.assertAlmostEqual(fam["FULLY_RANDOMIZED"]["violates_10ms"], 1 / 6)
        # identical runs give a zero-width interval
        lo, hi = fam["DETERMINISTIC_BURST"]["missed_ci"]
        self.assertEqual((lo, hi), (1.0, 1.0))
        sens = {r["family"]: r for r in result["sensitivity"]["0.25"]}
        self.assertAlmostEqual(sens["DETERMINISTIC_BURST"]["relative_energy"],
                               (0.25 + T_BREAKER) / IDEAL)

    def test_bootstrap_is_seeded(self):
        a = pc.analyse(self.frames(), iterations=200, seed=7, sensitivity=())
        b = pc.analyse(self.frames(), iterations=200, seed=7, sensitivity=())
        self.assertEqual(a["by_family"], b["by_family"])

    def test_cli_writes_reports_bound_to_the_dataset_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "pool.csv")
            self.frames().to_csv(data, index=False)
            md, js = os.path.join(tmp, "r.md"), os.path.join(tmp, "r.json")
            code = pc.main(["--dataset", data, "--iterations", "50",
                            "--out", md, "--json-out", js])
            self.assertEqual(code, 0)
            with open(js, encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertEqual(report["dataset_sha256"], pc.sha256_file(data))
            with open(md, encoding="utf-8") as fh:
                self.assertIn("`DETERMINISTIC_BURST`", fh.read())


if __name__ == "__main__":
    unittest.main()
