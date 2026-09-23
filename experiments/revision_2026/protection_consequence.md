# Protection consequence: GOOSE loss read as a transfer trip (card G, items 3-4)

Preregistered 2026-09-23, before `protection_consequence.py` computed a single
number on the pool. Answers Reviewer 3, comment 5 — that packet classification
cannot establish protection risk, and that at least one protection function
should show the operational consequence (missed or delayed trips, transfer
trip latency, clearing time) — at the level card G item 4 allows: **a simple
model as preliminary evidence, not proof**. HIL and co-simulation are out of
scope for this revision (decision 2026-09-23).

**What is not blind.** `threat_model.md` §4 already reported, per cell, the
fraction of fault states with at least one delivered frame. Layer 1's
missed-trip rate is that same quantity counted per event rather than per run,
so its direction is known in advance. Latency, the timing-requirement rates
and all of layer 2 have not been computed.

## 1. The function being modelled

The scenario's GOOSE is an interlocking dataset (`goID=IntLockA`) whose
`cbStatus` carries a breaker position. We read it instead as a **direct
transfer trip** (decision 2026-09-23): the publisher is the local relay, the
fault state (`cbStatus = 1`) is the trip command, and the subscriber is the
remote breaker's relay, which trips on the **first delivered frame** carrying
it. This is the reading under which the reviewer's measures (missed trip,
transfer trip latency, clearing time) are defined. It is a reinterpretation of
the generator's semantics and the paper must say so.

Assumptions of the subscriber, all ours:

1. It acts only on a received `cbStatus = 1` frame. A `StNum` discontinuity
   (a skipped state) is not treated as a trip — it cannot know what the
   skipped state said.
2. It needs one frame; later retransmissions of the same state change nothing.
3. No local backup at the subscriber: if the trip never arrives, the fault is
   cleared by **remote backup protection** (zone-2 time-delayed tripping).

## 2. The unit, and ground truth

The unit is one **fault event**: one fault state and its recovery (§1 of
`threat_model.md`). The dataset only holds delivered frames, so the events that
were never seen must be reconstructed:

- **Fault parity.** Per run, fault states are the `StNum` values of one parity
  — the parity of the states whose frames carry `cbStatus = 1`. A run in which
  no fault frame survives takes the parity implied by its initial `StNum`
  (faults are the states after an odd number of changes).
- **Events.** Every `StNum` of fault parity strictly after the run's first
  state and strictly before its last one.
- **Event time `t_f`.** The `t` field of any delivered frame of the fault
  state; if none survives, the `t` field of the recovery state minus 0.1 s
  (the generator schedules recovery exactly 100 ms after the fault). If neither
  survives, the event is **unobservable**, reported as a count, and excluded
  from every rate. That exclusion is conservative: it removes events the
  attack hid best.

## 3. Layer 1: what the subscriber sees

Per event:

- **missed trip** — no delivered frame of the fault state.
- **transfer latency** `L` — earliest delivered `GooseTimestamp` of the fault
  state minus `t_f`, for delivered events.
- **timing violations** — `L > 3 ms` and `L > 10 ms`: the transfer-time
  requirements for trip messages (IEC 61850-5, message type 1A; 3 ms for
  performance classes P2/P3, 10 ms for P1 — to be checked against the edition
  cited in the paper). A missed trip counts as a violation of both.

## 4. Layer 2: what the power system sees

Clearing time per event:

- delivered: `T_clear = T_relay + L + T_breaker`
- missed: `T_clear = T_backup + T_breaker`

| Parameter | Value | Source |
|---|---|---|
| `T_relay` | 20 ms | [assumption] ~1.2 cycles at 60 Hz, a typical numerical relay operate time |
| `T_breaker` | 50 ms | [assumption] 3-cycle interrupting time at 60 Hz, a common rating for transmission breakers |
| `T_backup` | **400 ms**, sensitivity 250 and 600 ms | [assumption] zone-2 time delay; the range covers common settings |

Reported: mean and 95th-percentile `T_clear`, and **relative fault energy**
`T_clear / (T_relay + T_breaker)`. For a sustained fault current `I`, `I²t`
scales with clearing time, so the ratio is the energy the fault delivers
relative to an ideally delivered trip, independent of `I`. The same number
is the **voltage sag duration**, since the sag lasts until the fault clears.

**Why no absolute current, voltage or `I²t`.** ERENO ships 132 fault
waveforms per fault-resistance set (`electrical-sources/res{10,50,100}`), but
their units, network and fault types are undocumented in the repository; some
cases show no fault at all. Reporting amperes or kilovolts from them would put
numbers in the paper whose provenance we cannot state. They stay out until the
provenance is known (§8).

**What layer 2 adds, stated honestly.** It is a deterministic transform of
layer 1 through three assumed constants. It adds no information about the
attack. What it adds is units: it turns "a fraction of frames lost" into
"milliseconds the fault stays on the system", which is the question a
protection engineer asks.

## 5. Uncertainty

Runs are the independent unit. Per cell and per family, a percentile bootstrap
over runs (1,000 replicates, seed 42) on the missed-trip rate and the mean
relative energy, matching `bootstrap_run_intervals.py`'s unit.

## 6. What would change the reading, stated in advance

1. **Expected.** `SAG.DB` at burst 3/5 misses essentially every trip, so its
   mean clearing time sits at `T_backup + T_breaker` (~6.4x relative energy at
   400 ms). `SAG.PB` scales with `loss_rate`. `SAG.PBM` and `FRG` rarely miss a
   trip.
2. **Could surprise.** Latency among *delivered* trips. A fault state whose
   first frame is dropped is delivered on a retransmission ≥ 100 ms later,
   which violates both transfer-time requirements even though the trip is not
   "missed". `SAG.PBM` could therefore be harmless by the missed-trip measure
   and harmful by the latency one. That would change §4 of `threat_model.md`.
3. **The benign comparison.** If a benign control (queue overload, link flap)
   produces clearing times comparable to a SAG variant, the paper cannot claim
   state-aware attacks are *more harmful than loss*; only that they are
   *targeted*. `threat_model.md` §4 already points this way.

## 7. Result (2026-09-23)

`protection_consequence.py` on the corrected pool (SHA-256 `3109e4d4…`,
265 runs), 24 s. Full tables: `protection_consequence_result.md` / `.json`.
Tests: `test_protection_consequence.py`, 11. Constants as preregistered:
`T_relay` 20 ms, `T_breaker` 50 ms, `T_backup` 400 ms, so an ideally delivered
trip clears in 70 ms and a missed one in 450 ms (relative energy 6.43).

**Compare cells, not families.** Runs are sized by attack-message count, so
events per run range from 358 (`FULLY_RANDOMIZED`) to 3,864 (`RANDOMIC_BURST`)
and a family's pooled figure is weighted toward its low-loss cells, which have
the largest runs. The family table is context; the matched cells below are the
result.

### At matched parameters

| | missed trip | > 10 ms (incl. missed) | relative energy |
|---|---:|---:|---:|
| `SAG.DB`, burst 3 / 5 | **1.0000** | 1.0000 | **6.43** |
| `SAG.DB`, burst 10 | 0.8948 | 1.0000 | 6.09 |
| `SAG.PB`, burst 3, loss 5 / 15 / 30% | 0.0506 / 0.1457 / 0.2892 | = missed | 1.27 / 1.79 / 2.57 |
| `SAG.PBM`, burst 3, loss 5 / 15 / 30% | 0.0003 / 0.0034 / 0.0267 | 0.0500 / 0.1469 / 0.2881 | 1.08 / 1.25 / 1.60 |
| `FRG`, loss 5 / 15 / 30% | 0.0001 / 0.0040 / 0.0297 | 0.0465 / 0.1450 / 0.2958 | 1.07 / 1.25 / 1.63 |
| `BENIGN_CONGESTION_LOSS`, 5 / 15 / 30% | 0.0000 / 0.0028 / 0.0310 | 0.0503 / 0.1424 / 0.2909 | 1.07 / 1.24 / 1.63 |
| `BENIGN_QUEUE_OVERLOAD_BURST`, burst 3 / 5 / 10 | 0.1394 / 0.2978 / 0.4925 | 0.3375 / 0.4635 / 0.6079 | 2.18 / 2.96 / **3.92** |
| `BENIGN_LINK_FLAP`, burst 3 / 5 / 10 | 0.0475 / 0.1429 / 0.3831 | 0.1429 / 0.2379 / 0.4847 | 1.46 / 1.99 / 3.30 |

Run-bootstrap intervals are in the result file; none of the comparisons drawn
below rests on a difference inside them. `SAG.PB` and `SAG.PBM` at bursts 5 and
10 match burst 3 to within 0.01 on missed trip.

### How the preregistered readings came out (§6)

1. **Expected — held.** `SAG.DB` suppresses the trip in every observable event
   at burst 3 and 5, so the fault stays on for the backup's 450 ms instead of
   70 ms: 6.43x the energy and the sag duration (4.25x-9.06x across the 250-600
   ms backup range). `SAG.PB` misses at its trigger rate, independent of burst
   size. `SAG.PBM` and `FRG` miss almost nothing (≤ 0.03 at 30% loss).
2. **Could surprise — it did.** The probabilistic variants are not separated by
   *whether* they break the transfer-time requirement but by *how*. At matched
   loss, `SAG.PB`, `SAG.PBM` and `FRG` all break the 10 ms limit in ≈
   `loss_rate` of events. `SAG.PB` does it by suppressing the trip (backup,
   450 ms); `SAG.PBM` and `FRG` by dropping the fault's first frame, so the
   trip arrives on the +100 ms retransmission (170 ms). So `threat_model.md`
   §4's "`SAG.PBM` is the least harmful" holds for missed trips and **fails for
   timing**: it breaks the requirement as often as `SAG.PB` does, at a lower
   cost per violation.
3. **The benign comparison — the claim does not survive for `SAG.PB`.** A
   10-frame queue overload (3.92) and a 10-frame link flap (3.30) cost more
   than any `SAG.PB` cell (≤ 2.65), and the three uniform-loss variants
   (`SAG.PBM`, `FRG`, congestion loss) are indistinguishable at every loss rate
   — `FRG` ≡ `CONGESTION_LOSS` again, now at the protection layer. **Only
   `SAG.DB` is more harmful than every benign control**, and by a factor of
   1.6 against the worst of them. Caveat on "matched": the benign controls'
   loss parameters are not frame-loss equivalents of the attacks'
   (`QUEUE_OVERLOAD_BURST` draws per candidate burst start, `LINK_FLAP` is a
   deterministic period), so this compares mechanisms by name and burst
   length, not by frames lost.

### Ranking harm against detectability

| | relative energy, worst cell | detection AP (`f10`) |
|---|---:|---:|
| `SAG.DB` | 6.43 | 0.8444 |
| `SAG.PB` | 2.65 | 0.8285 |
| `FRG` | 1.63 | 0.5914 |
| `SAG.PBM` | 1.64 | 0.3716 |

Under this model the two orderings agree: the variants that cost the most are
the ones the detector finds best, and the one it finds worst costs about as
much as uniform random loss. That is the opposite of the "stealthy and
dangerous" framing a hardest-to-detect attack invites, and the paper should
say it in these terms.

### Artifacts of the generator, not results

- **Latency is quantised to 0, 100 and 200 ms**: burst frames carry no network
  delay in ERENO, so a delivered trip is either on time or one or two
  retransmissions late.
- **Negative latency** under `BENIGN_JITTER` (±20 ms applied symmetrically to
  timestamps, 49.5% of events below zero) and `BENIGN_REORDERING` (down to
  −505 ms, when a fault frame swaps timestamps with the preceding heartbeat).
  A frame cannot arrive before the event; the preregistered formula does not
  clip, so both families read at 0.99-1.00 relative energy. Neither affects any
  attack family.
- **Unobservable events** (fault and recovery both lost): 3,534 in `SAG.DB`
  (6.7%, all at burst 10) and 1,636 in `SAG.PB`, excluded as preregistered.
  Including them as missed would raise `SAG.DB` burst 10, not lower it.

### What this licenses

- "`SAG.DB` suppresses every transfer trip it targets; under a 400 ms zone-2
  backup the fault stays on 6.4x longer" — **as a model result**, with the
  three constants stated and the 250-600 ms range reported.
- "State-aware attacks are more harmful than benign loss" — **only for
  `SAG.DB`**. For `SAG.PB` it is false against bursty benign loss, and for
  `SAG.PBM` it is false against uniform loss.
- Nothing about a real relay, a real network or voltage and current
  magnitudes. That remains Reviewer 3's request, and this section is the
  preliminary evidence card G item 4 allows, not the validation item 3 asks
  for.

## 8. Open

- Provenance of the SILVIO waveforms (network, units, fault types) — needed
  before any absolute electrical quantity.
- Citations for the three layer-2 constants and for the IEC 61850-5 transfer
  times.
