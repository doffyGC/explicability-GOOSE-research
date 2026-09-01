# Benign degradation controls (Card C)

Checklist ref.: C.1–C.5 in `prioridades_revisao_gray_goose.pdf`, P0 "Controles
benignos realistas". See `README.md` for how this fits the rest of the
revision and `validation_protocol.md` for the grouped-validation workflow this
plugs into.

## 1. Why this exists

Every packet-loss event in the Gray-GOOSE dataset today is malicious: it is
produced by `OrientedGrayHoleCreator` (ERENO `uc09`) under one of four
variants. There is no benign packet loss, congestion, jitter, delay,
duplication or reordering anywhere in the dataset or in the generator. A model
trained on this data cannot distinguish "there was a gap in GOOSE traffic"
from "there was a *malicious* gap" — it has never seen the first without the
second. Reviewers flagged exactly this (see `CLAUDE.md`, "Known baseline
issues driving the revision").

The purpose of this workstream is to generate paired benign controls, plug
them into the grouped-validation pipeline as a distinct class, and report
which benign degradations get confused with which attack variants.

## 2. Label vocabulary

| Column | Value | Meaning |
|---|---|---|
| `class` | `benign_degradation` | The row-level model target. One new class, parallel to `normal` and the four `SAG.*`/`FRG` attack classes. Never split further — that would multiply classes needlessly for the classifier's actual job (attack vs. not). |
| `class` | `normal` | Unaffected message. Kept separate from `benign_degradation` so the confusion matrix can distinguish "ideal normal" from "benign but degraded" (checklist item C.3). |
| `impairment_mode` | one of the 7 modes in §3, or `NONE` | Metadata column recording *which* benign mechanism produced a `benign_degradation` row, or `NONE` for `normal`/attack rows. **Never a model feature** — see §6. |

This mirrors the existing `attack_variant` design (`add_experiment_metadata.py`):
one coarse class for the model, one metadata column carrying the fine-grained
mechanism for reporting.

## 3. Taxonomy of the seven mechanisms

Each mechanism is designed to falsify one specific hypothesis a classifier
might be relying on. The "pairs with" column is what makes a control honest:
same `loss_rate`/`burst_size`/seed as the attack it is compared against,
changing only the trigger mechanism.

| `impairment_mode` | Mechanism | Pairs with | Falsifies | Tier |
|---|---|---|---|---|
| `CONGESTION_LOSS` | Bernoulli draw per message at `rate`; independent of protocol state | FRG (`loss_rate∈{5,15,30}`, `burst_size=1`) | "isolated random loss ⇒ FRG" | 1 |
| `QUEUE_OVERLOAD_BURST` | Bernoulli draw triggers a burst of `burst` consecutive drops, **not aligned to a state change** | SAG.PB (`loss_rate=15`, `burst_size∈{3,5,10}`) | "burst loss ⇒ SAG.PB" | 1 |
| `JITTER` | No loss; perturbs `timestamp` by ±jitter ms while preserving `t` (the event time) | — (negative control) | "timing variance ⇒ attack" | 1 |
| `DELAY` | No loss; adds a queueing delay to `timestamp` | — (negative control) | "delayed delivery ⇒ attack" | 1 |
| `LINK_FLAP` | Contiguous outage window of `burst` messages, deterministic periodic trigger (not probabilistic) | SAG.DB (`burst_size∈{3,5,10}`) | "deterministic burst ⇒ SAG.DB" | 2 |
| `DUPLICATION` | Reinserts a copy of a message (same `StNum`/`SqNum`/`t`, later `timestamp`) | — | "anomalous `sqDiff` ⇒ attack" | 2 |
| `REORDERING` | Swaps the `timestamp` of two adjacent messages | — | "broken order ⇒ attack" | 2 |

Tier 1 (4 mechanisms) is generated first; tier 2 (3 mechanisms) follows if time
allows, per the checklist's own priority note: *"Se não der tempo para todos,
priorizar: jitter+delay, random loss pareado e burst loss benigno."*

### Labeling convention

The **affected, retained** message receives `benign_degradation` — the same
convention `OrientedGrayHoleCreator` uses for attacks (it labels the message
*following* a drop, not the dropped message itself, since dropped messages are
absent from the dataset). Concretely:

- Loss modes (`CONGESTION_LOSS`, `QUEUE_OVERLOAD_BURST`, `LINK_FLAP`): the
  message immediately after the gap.
- `JITTER`/`DELAY`: the perturbed message itself.
- `DUPLICATION`: the reinserted copy.
- `REORDERING`: the message that ends up out of order.

Keeping this identical to the attack convention is deliberate: if benign and
malicious rows were labeled by different rules, the model could learn to
separate them by a labeling artifact instead of by the actual traffic
signature, which would make the whole control worthless.

## 4. Pairing with attacks

Runs are **not** co-generated with attacks in the same trace. Each benign
control is its own independent run (own `run_id`, own `split_group`), so it
can be left out of a fold independently and — once enough benign families
exist — support leave-one-event-type-out without collapsing into an open-set
fold (see `validation_protocol.md` §"Leave-one-event-type-out status": this is
exactly the missing "event-type axis orthogonal to attack class" it asks for).

Pairing is expressed through matching *parameters*, not shared traces: a
benign run and its paired attack run share the same seed, `loss_rate` and/or
`burst_size` value, and the same base traffic timing. This is what lets the
final report say "at loss_rate=15%, burst_size=5, FRG confuses distinguishing
attack loss from congestion loss X% of the time" without the two runs ever
touching the same messages.

## 5. Matrix design

Same 5 seeds as the attack matrix (`run_matrix_plan.json`: `20260101`–
`20260105`), same target of 1,000 labelled messages per run.

| Mode | Axis | Values | Runs |
|---|---|---|---|
| `CONGESTION_LOSS` | rate | 5%, 15%, 30% | 3 × 5 = 15 |
| `QUEUE_OVERLOAD_BURST` | burst | 3, 5, 10 | 3 × 5 = 15 |
| `JITTER` | intensity | 2 levels (TBD in C1: e.g. ±5ms, ±20ms) | 2 × 5 = 10 |
| `DELAY` | intensity | 2 levels (TBD in C1) | 2 × 5 = 10 |
| **Tier 1 total** | | | **50** |
| `LINK_FLAP` | burst | 3, 5, 10 | 3 × 5 = 15 |
| `DUPLICATION` | intensity | 2 levels | 2 × 5 = 10 |
| `REORDERING` | intensity | 2 levels | 2 × 5 = 10 |
| **Tier 2 total** | | | **35** |
| **Grand total** | | | **85** |

`run_id` convention: `BENIGN_<MODE>-l<rate>-b<burst>-s<seed>` (unused axis
fixed at a nominal value, mirroring how the attack matrix collapses inactive
axes for `DETERMINISTIC_BURST`/`FULLY_RANDOMIZED`). `scenario_id`:
`SC-BENIGN_<MODE>-l<rate>-b<burst>`.

## 6. Pipeline integration constraints

Found while exploring the existing card A/B code; each of these gets fixed in
the milestone named:

1. `stationBusMessages` is a `PriorityQueue<EthernetFrame>` drained by
   `poll()` — write order follows the queue, not IED list order. Reordering
   and duplication must be expressed through `timestamp`, not list position.
   (C1)
2. All randomness must flow through `Rng` (`ereno/.../api/Rng.java`); `new
   Random(...)`/`Math.random()` would break the reproducibility the card A
   patch bought. (C1)
3. `merge_runs.py` requires an identical header across all pooled runs.
   Extending `RunContext.csvHeader()` invalidates every CSV already generated
   under the old schema (`data/validation-smoke/runs/`,
   `data/runs-matrix-smoke/`) — they must be regenerated. Expected, not a bug.
   (C2, C5)
4. `add_experiment_metadata.py` raises on an unknown `class` value
   ("Unknown class labels ...; extend VARIANT_OF_CLASS before running.").
   `benign_degradation` must enter `VARIANT_OF_CLASS`. (C3)
5. `generate_grouped_splits.infer_event_type` raises `SplitPlanningError` on
   any `scenario_id` without a recognized variant marker. `SC-BENIGN_*` needs
   its own mapping entry. (C3)
6. `attack_variant` describes the *message*, not the run — a benign row gets
   `none` there; `impairment_mode` carries the mechanism instead. Keep this
   reading consistent with how attack rows already work. (C1, C3)
7. `impairment_mode` is a trivial leak if it ever reaches the feature matrix
   (it directly encodes the ground truth). It must enter
   `IDENTIFIER_COLUMNS` in `run_grouped_validation.py` in the same commit that
   introduces the column. (C3)
8. `BalancedSamambaiaScenario.countMaliciousMessages()` counts everything that
   is not `normal`. Benign-labelled rows will now count too — convenient (the
   loop target becomes "labelled messages per run" and balances benign runs
   for free), but the log text and `merge_report.md`'s `attack_rows` column
   become misleading and need a naming pass. (C1, C3)

## 7. Status by milestone

- [x] **C0 — Design and pre-registration.** This document. No code touched.
- [x] **C1 — ERENO generator (Java).** `BenignImpairmentIED`/`Creator`
      (`br.ufu.facom.ereno.benign.uc01.*`), `RunContext.Impairment` enum,
      `attack.benignImpairment.*` block in `params.properties`, registry
      wiring in `BalancedSamambaiaScenario` (gated by `impairmentMode != NONE`,
      not an `attacks.properties` flag — it is an alternative run mode, not an
      attack). Verification: `mvn -q compile` clean; one manual run per mode
      (`java -cp target/classes ...BalancedSamambaiaScenario`) at
      `attack.benignImpairment.rate=15`/`burst=5`/`period=20`/`jitterMs=10`/
      `delayMs=50`, `run.seed=20260101`; reproducibility confirmed
      byte-identical across two runs per mode. All 7 modes produced both
      `normal` and `benign_degradation` rows.

  Decisions made while implementing (resolving the `TBD in C1` notes in §5):
  - **CSV schema.** `RunContext.csvHeader()`/`csvRow()` gained three columns:
    `impairment_mode`, `impairment_rate` (the Bernoulli/trigger percent
    actually used — kept separate from `loss_rate` because DUPLICATION/
    REORDERING trigger on a rate without losing any packet), and
    `impairment_intensity_ms` (JITTER/DELAY magnitude; 0 for every other
    mode). `loss_rate`/`burst_size` are reused for the three loss-y modes
    exactly as §4 describes: CONGESTION_LOSS reports `loss_rate=rate,
    burst_size=1`; QUEUE_OVERLOAD_BURST reports `loss_rate=rate,
    burst_size=burst`; LINK_FLAP reports `loss_rate=100, burst_size=burst`
    (same "100 means unconditional drop-on-trigger" convention as
    DETERMINISTIC_BURST). `attack_variant` reports `NONE` for every benign
    row instead of the idle `attack.orientedGrayhole.variant` value, so it
    never falsely claims a grayhole variant ran. All three new columns are
    also written into the `.run.json` sidecar. This is a breaking header
    change for already-generated CSVs — expected per constraint 3 in §6, not
    a bug.
  - **JITTER/DELAY apply to every message in the run**, not to a Bernoulli-
    gated subset: each run is already an isolated single-variable test (one
    mechanism, one run), so a "some jittered, some not" design would just add
    a second hidden rate parameter with no clear default. Every row from the
    `BenignImpairmentIED` in a JITTER/DELAY run is `benign_degradation`
    accordingly; `normal` rows in the pooled dataset still come from the
    co-generated legitimate baseline (`attacks.legitimate=true`) and from
    other runs. JITTER draws ±`jitterMs` uniformly per message and leaves `t`
    untouched; DELAY draws `[0, delayMs]` uniformly (one-directional, matches
    "queueing delay").
  - **DUPLICATION/REORDERING/CONGESTION_LOSS/QUEUE_OVERLOAD_BURST reuse one
    generic `attack.benignImpairment.rate` field** (a Bernoulli percent) —
    mirrors how the attack side reuses `discardRate`/`burstSize` across
    variants. `LINK_FLAP` uses a new `attack.benignImpairment.period`
    (messages between deterministic outages) instead of a rate, since it is
    explicitly non-probabilistic.
  - **`BenignImpairmentIED.addMessage` bypasses `ProtectionIED`'s message-count
    cap** (same override `LegitimateProtectionIED` already uses), because
    DUPLICATION inserts more rows than the batch size and the inherited cap
    would silently drop the surplus.
  - Left for C2/C3 as originally scoped: the exact 2-level intensity/rate
    matrix per mode (§5's `TBD` axis values are a *default single value* per
    field right now, not yet a matrix), `VARIANT_OF_CLASS`/`infer_event_type`/
    `IDENTIFIER_COLUMNS` wiring, and the `countMaliciousMessages` naming pass
    (a comment now documents why it intentionally counts `benign_degradation`
    rows too).
- [ ] **C2 — Benign run matrix (Python).** `generate_run_matrix.py` extended
      with `--family {attack,benign,all}` / `--tier {1,2,all}`,
      `benign_matrix_plan.json`. Verification: dry-run plan has 50/85 unique
      `run_id`s; mixed smoke (attack + benign) passes `merge_runs.py
      --check-only`; payload fingerprints of paired runs differ.
- [x] **C3 — Pipeline plumbing.** `VARIANT_OF_CLASS["benign_degradation"] =
      "none"` (§6 constraint 4 — not "BENIGN": `attack_variant` answers "which
      attack", so a benign row reads `none` exactly like `normal`, and
      `impairment_mode` carries the mechanism instead, per constraint 6).
      `infer_event_type` gained one marker per benign mode (`BENIGN_<MODE>` →
      `BENIGN.<MODE>`, §6 constraint 5), disjoint from the 4 attack markers.
      `run_grouped_validation.IDENTIFIER_COLUMNS` gained `impairment_mode`,
      `impairment_rate`, `impairment_intensity_ms` (§6 constraint 7).
      `merge_runs.py`'s `NATIVE_COLUMNS`/`PER_RUN_CONSTANT` gained the same
      three columns, since they are run-level constants exactly like
      `loss_rate`/`burst_size` (confirmed against `RunContext.csvRow()`: it
      writes the singleton `impairmentMode` into *every* row of a run,
      `normal` rows included — the §2 table's "NONE for normal rows" phrasing
      was aspirational, not what C1 shipped, and re-deriving it per-row was
      out of scope for C3). New `test_benign_controls.py` (10 tests).
      `add_experiment_metadata.py`'s naming pass (§6 constraint 8):
      `attack_rows` renamed to `labelled_rows` and made `class`-based instead
      of `attack_variant`-based (the old attack_variant-based count silently
      read 0 for every benign run, since attack_variant is `none` for
      `benign_degradation` too); the per-run `variant` report column now
      shows `BENIGN:<mode>` instead of a useless constant `NONE`; a new
      "Benign-degradation coverage" report section checks each impairment
      mode spans ≥2 runs; the attack-coverage section no longer prints a
      false "single-run" warning when a dataset has zero attack rows
      (surfaced by testing on a benign-only pool, which wasn't a possible
      input before C3).

  Verification: full suite green (29 tests, `python -m unittest discover -p
  "test_*.py"`). Canonical workflow (merge → metadata → prepare → splits →
  leakage → train) run on a real 17-run benign-only smoke pool (all 7
  mechanisms, seed 20260101, 535,330 rows after preparation — a subset of the
  85-run matrix generated in C2): `merge_runs.py` (0 rejected, `Coverage per
  variant` correctly breaks out 7 `BENIGN:<mode>` rows instead of collapsing
  into one `NONE` bucket), `add_experiment_metadata.py` (`classes` = `normal`,
  `benign_degradation`), `prepare_grouped_dataset.py`, `generate_grouped_splits.py`
  (`stratified-group-kfold`, 5 folds, 17 groups — and, as a direct
  demonstration of constraint 5's fix, `leave-one-event-type-out` over the 7
  benign families: **7/7 folds closed-set**, `open_set_diagnostic: false`,
  every fold's `test_only_labels` empty — contrast the attack-variant case in
  `validation_protocol.md`, which is open-set today), `check_no_leakage.py`
  (pass, 17/17 groups), `run_grouped_validation.py` (`--max-rows-per-group-class
  500` technical smoke: `classes: ["benign_degradation", "normal"]`, 40
  numeric features, zero `impairment_*` columns among them).
- [x] **C4 — Confusion report.** `benign_confusion_report.py` rejoins
      `impairment_mode` onto `run_grouped_validation.py`'s predictions (hash-
      verified against the run's own `dataset_sha256`, via the stable
      positional `row_index` — `prepare_grouped_dataset.py` writes its output
      with a clean `reset_index(drop=True)`, and `run_grouped_validation.py`
      never re-indexes, so `row_index` is a valid position into the exact
      dataset file the report names). Produces: a class×class confusion
      matrix over the 6-class vocabulary (`normal`, `benign_degradation`,
      SAG.DB/FRG/SAG.PB/SAG.PBM), gracefully reduced — not zero-filled — when
      a run's classes are a subset of the 6 (e.g. today's benign-only
      pools); a per-mode outcome breakdown of what each mechanism's
      `benign_degradation` rows actually get predicted as; and, per mode,
      `attack_fpr` (misclassified as one of the four attack classes — the
      specific harm §3's "falsifies" column targets) and `alert_rate`
      (misclassified as anything but `normal`), contrasted against two
      normal-traffic baselines kept deliberately separate: ideal normal
      (`impairment_mode=NONE`) and the normal baseline messages captured
      *inside* a benign-impairment run (impairment_mode is run-level, so
      these are not the same population — see §6 constraint 8). When no
      attack classes are present in a run, the report says so explicitly
      instead of implying a 0.00% attack_fpr means something.

      Verification: 20 tests in `test_benign_controls.py` (confusion-matrix
      class reduction, display names, `attack_fpr`/`alert_rate` arithmetic
      including the empty-slice n/a case, the two normal baselines, and the
      hash-bound/positional dataset rejoin including a wrong-hash and an
      out-of-range `row_index` rejection). Also run end-to-end on a fresh
      real 17-run benign-only smoke prediction set (regenerated via the same
      C3 canonical-workflow steps): `benign_confusion.md` produced, correctly
      reports 2/6 classes present (`normal`, `benign_degradation`; the 4
      attack classes explicitly listed as absent, not silently omitted),
      7/7 impairment modes broken out with sane per-mode alert rates
      (12%–99%, `DUPLICATION` producing the strongest signal at 98.9%,
      `QUEUE_OVERLOAD_BURST` the weakest at 22.2% — expected, since the
      un-gated benign mechanisms are the ones closer to the attack's own
      surface), `attack_fpr` correctly flat at 0.00% everywhere with an
      explicit note that this is uninformative absent any attack rows in the
      pool, and `Run status: technical_smoke` carried through from
      `run_grouped_validation.py`'s own report.
- [ ] **C5 — Full generation and documentation.** Regenerate attack matrix
      (120 runs) + benign tier-1 (50 runs) under the new schema, run the
      canonical workflow on the pooled dataset, populate `benign_confusion.md`
      with real numbers, update `validation_protocol.md`/`data_card.md`/
      `README.md`/`metadata_audit.md`, close this document's checklist.

To resume work on a later day: read this file's status table and
`README.md`'s script table, then continue at the first unchecked milestone.
