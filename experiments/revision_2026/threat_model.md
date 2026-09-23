# Threat model: what each SAG variant assumes (card G, items 1-2)

Started 2026-09-23. Answers Reviewer 3, comment 4 (a table linking each variant
to the observation and control capabilities it needs, and how VLANs, redundant
paths, IEC 62351-6 and routine monitoring change it), and prepares Reviewer 1,
comments 1-2, and the editor's IEC 62351-6 / defence-in-depth objection.

Every cell below is marked with where it comes from:

- **[code]** — read from the ERENO checkout next to this repo (`../ereno`,
  HEAD `37f5737`, 2026-09-01), with the file named. The run sidecars
  (`*.run.json`) do not record the ERENO commit, so this is the code
  currently checked out, not verified provenance of the pool. The data
  cross-checks it: in `DETERMINISTIC_BURST-l100-b3-s20260101` no fault state
  survives and every recovery state starts at `SqNum` 4, exactly what §1
  predicts for a 3-frame burst.
- **[data]** — measured on the corrected pool
  (`gray-GOOSE-runs-prepared.parquet`, SHA-256 `3109e4d4…`, 265 runs).
- **[assumption]** — ours. Nothing in the generator or the data constrains it,
  so the paper has to state it as an assumption, not a finding.

---

## 1. What the generator actually simulates

These are the facts every row of the table depends on. None of them is new
code; they had just never been written down next to the attack model.

**The protected event [code, `general/ProtectionIED.java`,
`benign/uc00/creator/GooseCreator.java`].** One event is a *fault* followed by
its *recovery* 100 ms later (`reportEventAt(last + 0.5)`, then `+ 0.6`). Each
is a state change: `StNum` increments, `SqNum` restarts at 1, and `cbStatus`
toggles — `1` for the fault state (the trip indication), `0` for the
recovery. Between events the publisher sends a heartbeat every `maxTime` =
1 s.

**The retransmission burst [code, `ProtectionIED.exponentialBackoff`].** With
`minTime` = 100 ms and a multiplier of 6.33, a state change is announced by
**four frames at +0, +100, +200 and +833 ms**. The fault's fourth frame is
removed before the recovery is scheduled, so **a fault state is at most three
frames long** and lives ~100 ms. `burst_size` ∈ {3, 5, 10} therefore means:
3 drops exactly one fault state's announcement; 5 and 10 run past it into the
recovery and the following heartbeats.

**The two bursts overlap on the wire [data].** The fault is retransmitted at
+100 and +200 ms and the recovery starts at +100 ms, so the two states
interleave with identical timestamps (e.g. `BENIGN_CONGESTION_LOSS-l15-s20260101`,
t = 15.60631 carries `StNum` 2 *and* 3).

**The attacker is offline, not inline [code,
`attacks/uc09/creator/OrientedGrayHoleCreator.java`].** The grayhole iterates
the publisher's already-generated message *list* — generation order, fault
burst then recovery burst — with full look-ahead, and drops by **frame count**
from the first frame of a new `StNum`. It never sees the wire order above.
Consequences:

- It needs no timing knowledge and pays no processing delay: the model
  assumes both are free.
- "Drop the next *N* frames" in the list is not the same set of frames an
  inline attacker counting frames on the wire would drop, because on the wire
  the recovery's frames are interleaved with the fault's.
- A burst that runs past the next state change (`burst_size` 5, 10) skips that
  change's trigger evaluation (`i += toDiscardPackets - 1`).

**The label [code].** Dropped frames are gone from the dataset; the label goes
on the **next delivered frame** (`label_duplication_audit.md` §7 has why this
bounds recall).

## 2. The table the reviewer asked for

Parameters in the pool: `loss_rate` ∈ {5, 15, 30}%, `burst_size` ∈ {3, 5, 10},
five seeds, 45 runs per class [data, `run_matrix_plan.json`,
`data_card.md` §5]. Detection column: the published detector
(`f10-xgboost-no-abs-no-counters`, `ablations_baselines.md` §20),
`pr_curves_f10_d2.md`.

| | `SAG.DB` (`DETERMINISTIC_BURST`) | `SAG.PB` (`RANDOMIC_BURST`) | `SAG.PBM` (`RANDOMIC_MESSAGE`) | `FRG` (`FULLY_RANDOMIZED`), reference |
|---|---|---|---|---|
| **Trigger** [code] | every `StNum` change | every `StNum` change, with probability `loss_rate` | every `StNum` change | none — every frame |
| **What is dropped** [code] | the next `burst_size` frames, always | the next `burst_size` frames, all or nothing | each of the next `burst_size` frames independently, with probability `loss_rate` | each frame independently, with probability `loss_rate` |
| **What must be observed** [code] | `StNum` of every frame, compared with the previous one | same | same | nothing about content |
| **Decision latency the model assumes** [code] | zero — the frame that reveals the change is itself dropped, so the decision is on the frame in hand | same | same | zero, but no parsing is needed |
| **Knowledge of retransmission timing** [code] | none: counts frames. Whether a burst covers the announcement depends on `burst_size` against the 3-4-frame profile above | same | same | none |
| **Duration** [code] | the whole run: every state change, no on/off | same | same | the whole run |
| **Control over drop rate** [code] | none (effective loss 100%) | per state change | per frame, inside the window | per frame |
| **Network position** [assumption] | inline on the only path from publisher to subscriber, forwarding everything it does not drop — a compromised switch, a bump-in-the-wire, or a compromised IED on the path | same | same | same, but a benign faulty element in that position is indistinguishable (`benign_controls.md` §8) |
| **Privilege** [assumption] | control of a forwarding element's data plane; no key material (it never forges or modifies a frame) | same | same | same |
| **Line-rate forwarding** [assumption] | must parse to `StNum` inside the APDU for every frame while forwarding the rest without added jitter; the pool models no forwarding delay at all | same | same | no parsing needed |
| **Detection, AP** [data] | 0.8444 [0.7817, 0.8882] | 0.8285 [0.7930, 0.8583] | **0.3716** [0.3217, 0.4230] | 0.5914 [0.4927, 0.6887] |

## 3. What changes the feasibility

This column is argument, not measurement, except where marked. It is the
material for the paper's defence-in-depth discussion (Reviewer 1, comment 2).

| Control | Effect on the SAG variants | Evidence |
|---|---|---|
| **VLAN separation** | Does not stop an attacker who is already on the path: it narrows *which* positions qualify (a trunk or access port carrying the GOOSE VLAN), it does not change what the attack does once there. | [assumption] |
| **Redundant paths (IEC 62439-3 PRP/HSR)** | The strongest structural mitigation: a frame dropped on one path arrives on the other, so a single grayhole suppresses nothing. The attacker needs a position on both paths (PRP) or two positions in the ring (HSR). | [assumption]; the pool models one path only |
| **IEC 62351-6 authentication** | Does **not** prevent the attack. A grayhole drops genuine frames and never forges one, so a message authentication code verifies on every frame that arrives. `StNum` stays in clear, so the trigger is unaffected. This is the direct answer to Reviewer 1, comment 1: authentication closes injection, masquerade and replay (ERENO uc01-uc07), not selective suppression. | [code] — no SAG path modifies a frame |
| **Payload confidentiality, where deployed** | Hides `StNum`, which removes the trigger as implemented. It does not hide the retransmission timing: a state change is announced by frames 100 ms apart where the heartbeat is 1 s apart (§1), so the trigger can plausibly be recovered from timing alone. | [code] for the timing profile; the timing-triggered attack is **[hypothesis]**, not simulated |
| **Routine monitoring at the subscriber** | A `StNum` continuity check detects the variants that lose whole states. The single-threshold `stnum-gap` rule, calibrated on train runs only, reaches AP 0.2192 on `ANY_ATTACK` (11.2x chance) and beats the model at a 0.1% alert budget. It is blind where no whole state is lost — `SAG.PBM` and `FRG`. | [data], `ablations_baselines.md` §16, §20 |

## 4. A message-level consequence proxy (exploratory, not preregistered)

Reviewer 3, comment 5, asks for evidence that the attacks harm a protection
function. This section is **not** that evidence. It is the cheapest
preliminary quantity the pool already supports, in the spirit of card G's
fallback item ("simple model as preliminary evidence, not proof").

**The quantity.** For each run, the fraction of fault states — `StNum` values
whose frames carry `cbStatus = 1`, the trip indication — of which **at least one
frame reaches the subscriber**. Denominator: `(max StNum − min StNum) // 2`
per run, the number of fault/recovery pairs after the initial state. Mean over
runs in each cell [data]:

| Family | burst | loss | trip indications delivered |
|---|---:|---:|---:|
| `SAG.DB` | 3 | 100% | **0.000** |
| `SAG.DB` | 5 | 100% | **0.000** |
| `SAG.DB` | 10 | 100% | 0.087 |
| `SAG.PB` | 3 / 5 / 10 | 5% | 0.949 / 0.949 / 0.940 |
| `SAG.PB` | 3 / 5 / 10 | 15% | 0.854 / 0.853 / 0.830 |
| `SAG.PB` | 3 / 5 / 10 | 30% | 0.711 / 0.711 / 0.677 |
| `SAG.PBM` | any | 5-30% | 0.972-1.000 |
| `FRG` | 1 | 5 / 15 / 30% | 1.000 / 0.997 / 0.971 |
| `BENIGN_QUEUE_OVERLOAD_BURST` | 3 / 5 / 10 | 15% | 0.861 / 0.696 / **0.471** |
| `BENIGN_LINK_FLAP` | 3 / 5 / 10 | 100% | 0.952 / 0.857 / 0.603 |
| `BENIGN_CONGESTION_LOSS` | 1 | 5 / 15 / 30% | 1.000 / 0.998 / 0.969 |
| `BENIGN_DELAY`, `_JITTER`, `_DUPLICATION`, `_REORDERING` | — | — | 1.000-1.001 |

The delay, jitter, duplication and reordering controls come back at 1.000,
which is the check that the quantity counts what it claims to.

**What it says.**

1. **`SAG.DB` is the only variant that suppresses trip indications by
   construction** — none reach the subscriber at `burst_size` 3 or 5. At 10 a
   few survive (0.087) because the burst overruns the recovery's trigger and
   the next fault falls outside the window (§1).
2. **`SAG.PB` suppresses at its trigger rate**: delivery ≈ 1 − `loss_rate`, and
   burst size barely matters because three frames already cover a fault
   state.
3. **`SAG.PBM` almost never suppresses one** (≥ 0.972): dropping each frame
   independently rarely removes all three frames of a fault state. It is the
   variant the detector finds hardest (AP 0.3716) *and* the one this proxy
   says is least harmful. That is the reading the paper must not invert.
   **Refined by `protection_consequence.md` §7:** it holds for *missed* trips
   and fails for *timing*. `SAG.PBM` drops the fault's first frame in ≈
   `loss_rate` of events, so the trip arrives on the +100 ms retransmission and
   breaks the 10 ms transfer-time limit as often as `SAG.PB` does — at 170 ms
   of clearing rather than the backup's 450 ms.
4. **Benign degradation is not harmless by this measure.** A queue overload
   dropping 10-frame bursts at 15% loses more trip indications (delivery 0.471)
   than `SAG.PB` at 30% (0.677). So "state-aware attacks are more harmful than
   random loss" does not hold as a statement about *loss*. What distinguishes
   `SAG.DB` is that it is *targeted and total*, not that it loses more frames.

**What it does not say.** Whether any of this delays or prevents a trip. A
subscriber that misses the fault state but receives the recovery never sees the
breaker open; whether that matters depends on the protection scheme
(breaker-failure, interlocking, transfer trip), on `timeAllowedToLive`
(11 s here) and on what the relay does when a state is skipped. That is the
content of card G item 3, and until it exists every "more harmful" claim is a
hypothesis (Reviewer 3, comment 5).

**How it was computed** (to be promoted to a script with a test if the paper
uses it — see §6):

```python
import pyarrow.parquet as pq
df = pq.read_table("data/runs/gray-GOOSE-runs-prepared.parquet",
                   columns=["run_id", "burst_size", "loss_rate", "StNum", "cbStatus"]).to_pandas()
df["fam"] = df.run_id.str.replace(r"-.*", "", regex=True)
st = df.groupby(["run_id", "StNum"]).cbStatus.max().reset_index()
meta = df.groupby("run_id").agg(fam=("fam", "first"), b=("burst_size", "first"),
                                l=("loss_rate", "first"), mn=("StNum", "min"), mx=("StNum", "max"))
meta["ratio"] = (st[st.cbStatus == 1].groupby("run_id").size()
                 .reindex(meta.index).fillna(0) / ((meta.mx - meta.mn) // 2))
print(meta.groupby(["fam", "b", "l"]).ratio.mean().round(3))
```

The family comes from the `run_id` prefix, not from `attack_variant`: the
latter is per row and reads `none` on every unlabeled row of an attack run.

## 5. What this changes in the paper

- The threat model is **an inline forwarding element with data-plane control
  and no keys**. State it once, per variant, with the table in §2.
- **IEC 62351-6 authentication does not address this attack class**, and the
  paper can say why from the code: nothing is forged. That turns the editor's
  objection into a positioning argument — the detector covers the gap
  authentication leaves — provided it is framed as one layer of defence in
  depth, next to PRP/HSR (which does address it) and `StNum` continuity
  monitoring (which catches part of it).
- The generator assumes **zero decision latency and free line-rate parsing**,
  and applies the attack to generation order rather than wire order. Those are
  limitations of the simulation, not properties of the attacker, and belong in
  the threats-to-validity paragraph.
- "`SAG.PBM` is the most advantageous strategy" is not supportable. It is the
  hardest to *detect* and, by §4, the least likely to suppress a trip
  indication.

## 6. Open

| Item | Card G | Status |
|---|---|---|
| This table | 1 | **drafted** (this file) |
| VLAN / PRP-HSR / 62351-6 / monitoring discussion | 2 | **drafted** as §3; needs citations for PRP/HSR and 62351-6 |
| Protection-function validation | 3 | preliminary model done: `protection_consequence.md` §7 (transfer trip, subscriber + clearing time). Not HIL, not a real relay |
| HIL/cosim vs. simple model decision (60 days) | 4 | **decided 2026-09-23**: simple two-layer model for this revision |
| Hypothesis wording for "more harmful" claims | 5 | §5 lists the claims to soften |
| Promote §4 to a tested script | — | superseded: `protection_consequence.py` counts the same quantity per event, with tests |
