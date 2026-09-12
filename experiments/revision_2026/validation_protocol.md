# Section B — grouped validation protocol

## Decision

The revised paper uses **StratifiedGroupKFold by `run_id`** as its primary
closed-set protocol. It preserves the independent experimental unit while
reducing the risk that folds lose an attack class because run sizes differ.
Plain GroupKFold and LeaveOneGroupOut remain sensitivity variants.

No message-level random split is permitted. A model run consumes persisted
folds; it does not create new folds internally.

## Canonical workflow

### 1. Pool and annotate native ERENO runs

```bash
python experiments/revision_2026/merge_runs.py
python experiments/revision_2026/add_experiment_metadata.py \
  --dataset data/runs/gray-GOOSE-runs.parquet \
  --out data/runs/gray-GOOSE-runs-metadata.parquet
```

### 2. Recompute deltas inside traces

```bash
python experiments/revision_2026/prepare_grouped_dataset.py \
  --dataset data/runs/gray-GOOSE-runs-metadata.parquet \
  --out data/runs/gray-GOOSE-runs-prepared.parquet \
  --report experiments/revision_2026/preparation_audit.json
```

Rows are stably ordered by `trace_id`, `batch_index` and native source-row order,
and receive a pre-filter zero-based `message_index` inside the trace. Because
boundary row 0 is removed, retained rows begin at index 1. The native row order
is the sequence used by ERENO's `IntermessageCorrelation`; timestamp and
sequence-number fields can legitimately be non-monotonic under attack and are
therefore not used as ordering keys. The following ERENO features are recomputed:

- `stDiff = diff(StNum)`;
- `sqDiff = diff(SqNum)`;
- `gooseLengthDiff = diff(gooseLen)`;
- `apduSizeDiff = diff(APDUSize)`;
- `frameLengthDiff = diff(frameLen)`;
- `timestampDiff = diff(GooseTimestamp)`;
- `tDiff = diff(t)`;
- `cbStatusDiff = 1` when status changes, otherwise 0;
- `timeFromLastChange = GooseTimestamp - t`.

The first row of every trace is dropped because its predecessor is unavailable.
`T-UNRESOLVED` is rejected: computing within that mixed pool would cross hidden
trace boundaries.

### 3. Generate and persist folds

```bash
python experiments/revision_2026/generate_grouped_splits.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --protocol stratified-group-kfold \
  --n-splits 5 \
  --seed 42 \
  --out-json experiments/revision_2026/splits_grouped.json \
  --out-csv experiments/revision_2026/splits_grouped.csv
```

The JSON stores dataset SHA-256, protocol, seed, row counts, label coverage,
event-type mapping and exact train/test groups. The CSV is a reviewable long
form (`split_id,partition,split_group`). Generation fails if a test class is
absent from training.

### 4. Run the independent leakage audit

```bash
python experiments/revision_2026/check_no_leakage.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --splits experiments/revision_2026/splits_grouped.json \
  --report experiments/revision_2026/leakage_audit.json
```

This is intentionally separate from split generation. A non-zero exit status
must stop training.

### 5. Validate the pipeline before full training

```bash
python experiments/revision_2026/run_grouped_validation.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --preparation-report experiments/revision_2026/preparation_audit.json \
  --splits experiments/revision_2026/splits_grouped.json \
  --out-dir results/grouped-smoke \
  --model decision-tree \
  --max-rows-per-group-class 100
```

The cap marks the report as `technical_smoke`. Remove it for an
original-distribution grouped run. The runner verifies that the dataset hash
matches both preparation and split artifacts, and writes fold-linked
predictions. It never trains a final all-data model or runs SHAP.

### 6. Run the full, uncapped grouped validation

```bash
python experiments/revision_2026/run_grouped_validation.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --preparation-report experiments/revision_2026/preparation_audit.json \
  --splits experiments/revision_2026/splits_grouped.json \
  --out-dir results/grouped-validation-full \
  --model decision-tree
```

Dropping `--max-rows-per-group-class` loads and trains on every row; the
report status becomes `full_grouped_run`. On the 205-run/20.8M-row pool this
needs the whole prepared dataset in memory at once (model training inherently
requires it), so `run_grouped_validation.py` only reads the columns that
survive `feature_matrix`'s own discard sets (`load_frame(..., columns=...)`)
and casts the feature matrix to `float32` before `fit()` (sklearn's tree
splitter runs in float32 internally regardless, so this changes nothing about
the fitted model) — without both, materialising the training fold as a dense
array raised `numpy._core._exceptions._ArrayMemoryError` on a 16GB-RAM
machine. See "Full-scale results" below.

## Full-scale results (2026-09-12)

The 205-run pool (120 attack + 85 benign-degradation, `metadata_audit.md`)
has now been carried through the full canonical workflow above:

| Step | Result |
|---|---|
| Merge (`merge_runs.py`) | 205 runs, 20,797,126 rows, all passed validation |
| Metadata (`add_experiment_metadata.py`) | 205 traces/runs, 845,212 events |
| Delta preparation (`prepare_grouped_dataset.py`) | 20,796,921 rows; 205 trace-boundary rows removed |
| Splits (`generate_grouped_splits.py`) | 5-fold stratified-group-kfold, 205 groups |
| Leakage audit (`check_no_leakage.py`) | **pass** — 205/205 groups tested exactly once |
| Training (`run_grouped_validation.py --model decision-tree`) | `full_grouped_run`, 5 folds, 20,796,921 rows |

Fold results (mean over 5 folds): **accuracy 0.985 ± 0.007, macro-F1 0.276 ±
0.021.** The gap between those two numbers is the headline finding, not a
detail: per-class metrics are identical in shape across all 5 folds —

| Class | precision | recall | f1 |
|---|---:|---:|---:|
| `normal` | ~0.99 | ~0.99 | ~0.99 |
| `benign_degradation` | 0.44–0.94 | 0.44–0.82 | 0.57–0.88 |
| `SAG.DB` / `FRG` / `SAG.PB` / `SAG.PBM` (all four) | **0.000** | **0.000** | **0.000** |

The confusion matrix (`benign_confusion.md`) confirms this is not "mostly
misses, sometimes hits": across all 20.8M rows the model predicts an attack
class **for essentially no row at all** (all four attack columns are ~0 across
every true class). With attack rows at well under 1% of the pool and a plain
`DecisionTreeClassifier(max_depth=8)`, the tree finds it Gini-optimal to never
carve out a leaf for the rare classes.

**This is a class-imbalance artifact of running the first grouped model
completely unbalanced, not evidence about SAG detectability**, and not a
leakage problem — `check_no_leakage.py` passed. It does mean no claim about
SAG being detectable (or not) under grouped validation can be made from this
run. Checklist D (ablations/baselines) and E (balancing: no-SMOTE/SMOTE/
downsampling per fold) are the required next step; this run is the reference
point ("no balancing") the balanced runs must be compared against.

## Balancing scenarios (checklist E, 2026-09-12)

The "Full-scale results" run above is the **unbalanced** reference scenario
(checklist E's "sem balanceamento"). Checklist E calls for two more, both
implemented as `--balance {downsample,smote}` on `run_grouped_validation.py`:
each rebalances **only the current fold's TRAIN partition**; the test
partition is always the untouched original distribution, so precision/recall
numbers below are never inflated by evaluating on rebalanced data.

Neither scenario aims for exact parity with the majority (`normal`) class.
`normal` is ~16M rows in a typical fold's train partition against ~13-45k for
the rarest attack class — plain SMOTE-to-parity would synthesise on the
order of tens of millions of rows, the same class of failure as the OOM this
script already hit once (see "Run the full, uncapped grouped validation"
above). Both scenarios are therefore explicitly bounded:

```bash
# downsample: every class cut to the size of the smallest class in that fold's train partition
python experiments/revision_2026/run_grouped_validation.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --preparation-report experiments/revision_2026/preparation_audit.json \
  --splits experiments/revision_2026/splits_grouped.json \
  --out-dir results/grouped-validation-full-downsample \
  --model decision-tree --balance downsample

# smote: attack classes oversampled up to 20x their own count, capped at 200k;
# normal/benign_degradation (already above the cap) are left untouched
python experiments/revision_2026/run_grouped_validation.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --preparation-report experiments/revision_2026/preparation_audit.json \
  --splits experiments/revision_2026/splits_grouped.json \
  --out-dir results/grouped-validation-full-smote \
  --model decision-tree --balance smote
```

`--smote-oversample-factor` (default 20.0) and `--smote-max-target` (default
200,000) control the SMOTE cap; both are recorded in `grouped_validation_report.json`.
Requires `pip install imbalanced-learn` (added to `requirements.txt`).

### Downsample results

Mean over 5 folds: **accuracy 0.513 ± 0.041, macro-F1 0.197 ± 0.011.** Both
numbers *drop* relative to the unbalanced baseline (0.985 / 0.276) — expected,
since `normal` recall itself falls to ~44-55% once its training rows are cut
from ~16M to the size of the rarest attack class (~13-45k) per fold. The
headline change is recall on the four attack classes, previously exactly
0.000 for all of them:

| Class | recall (range across folds) | precision (range) |
|---|---:|---:|
| `SAG.DB` (`DETERMINISTIC_BURST`) | 0.74–1.00 | 0.005–0.022 |
| `FRG` (`FULLY_RANDOMIZED`) | 0.51–0.66 | 0.001–0.007 |
| `SAG.PB` (`RANDOMIC_BURST`) | 0.60–0.67 | 0.007–0.015 |
| `SAG.PBM` (`RANDOMIC_MESSAGE`) | 0.58–0.64 | 0.010–0.015 |
| `benign_degradation` | 0.51–0.87 | 0.23–0.40 |
| `normal` | 0.44–0.55 | 0.997–0.999 |

So the class-imbalance artifact reported above is confirmed, not contradicted:
the same decision tree **can** separate every attack class from `normal`
reasonably well once training sees them at comparable scale — it just never
tried to under the unbalanced default. The cost is precision: with `normal`'s
recall collapsing to ~50%, roughly half of all normal traffic in the original
test distribution is flagged as something else, so the *alert burden*
("Reportar false positive rate e alert burden em tráfego realista", checklist
E) at this operating point is far too high for direct deployment as-is. This
is the expected downsample trade-off (recall up, precision down from flooding
minority-class decision regions with too few majority examples to bound them
tightly) and is exactly why checklist E asks for three scenarios side by side
rather than picking one.

### SMOTE results

Mean over 5 folds: **accuracy 0.984 ± 0.007, macro-F1 0.274 ± 0.017** —
statistically indistinguishable from the unbalanced baseline (0.985 / 0.276).
Oversampling each attack class to 200,000 synthetic-plus-real rows (from
12k-45k) did **not** move the needle:

| Class | recall (range across folds) |
|---|---:|
| `SAG.DB` | 0.000–0.003 |
| `FRG` | 0.000–0.0002 |
| `SAG.PB` | 0.000–0.0012 |
| `SAG.PBM` | 0.000 (every fold) |
| `benign_degradation` | 0.43–0.82 |
| `normal` | 0.996–0.999 |

SMOTE's synthetic points are convex-combination neighbours of the real
minority rows already present, so they add density around existing minority
regions rather than new ones. At `DecisionTreeClassifier(max_depth=8)`, that
extra density still doesn't outweigh the accuracy gain from ignoring
attack classes altogether, when they remain ~1.2% of the training rows
(200k of ~16.7M) even after oversampling — the same Gini-optimality logic
documented in "Full-scale results" above, just less starved than before.
`benign_confusion_smote.md` confirms the model barely changed its behaviour
at all: normal-traffic `attack_fpr` stays at 0.04% (baseline: 0.00%).

### Cross-scenario comparison

| Scenario | mean accuracy | mean macro-F1 | attack-class recall | normal `attack_fpr` (ideal traffic) |
|---|---:|---:|---:|---:|
| none (baseline) | 0.985 | 0.276 | 0.000 (all 4 classes, every fold) | 0.00% |
| smote (capped, train-only) | 0.984 | 0.274 | ~0.000–0.003 (unchanged) | 0.04% |
| downsample (train-only) | 0.513 | 0.197 | 0.51–1.00 (all 4 classes detected) | **43.46%** |

None of the three scenarios is a usable operating point on its own: the
unbalanced and capped-SMOTE runs never detect an attack; the downsampled run
detects every attack class but at a false-positive rate on *ideal, unimpaired*
normal traffic that would flood any real deployment with alerts (44.60%
overall alert rate on that slice — see `benign_confusion_downsample.md`).
This is exactly the trade-off checklist E asks to be reported explicitly
rather than picked around: **detectability under grouped, leakage-free
validation depends entirely on how training balance is handled, and the two
balancing techniques tried so far sit at opposite, both-impractical ends of
the precision/recall trade-off.** Next steps this opens up (not yet done):
tuning the SMOTE cap/factor and tree depth together (a shallow tree may
simply lack the capacity to use denser minority regions), a class-weighted
loss as a third, cheaper alternative to explicit resampling, and comparing
against `xgboost`/Random Forest (checklist D) before drawing any conclusion
about SAG detectability being an inherent model-family limit versus a
decision-tree-at-depth-8 limit specifically. That is card D: its plan, agreed
scope and deferrals are in `ablations_baselines.md`.

### Metric labelling and prediction integrity (E.4/E.5)

`check_prediction_integrity.py` audits finished runs before any statistical
test is written, recomputing every number from a `numpy.bincount` confusion
matrix over the persisted `grouped_predictions.csv` — never from the
`sklearn` helpers the runner itself used, so a metric bug in the runner
cannot pass its own audit (the same decoupling rationale as
`check_no_leakage.py`):

```bash
python experiments/revision_2026/check_prediction_integrity.py \
  --run results/grouped-validation-full \
  --run results/grouped-validation-full-downsample \
  --run results/grouped-validation-full-smote \
  --out experiments/revision_2026/prediction_integrity.md \
  --json-out experiments/revision_2026/prediction_integrity.json
```

**Counts (E.5).** All three runs reconcile completely — 30 checks each, 0
failures: every `row_index` predicted exactly once, full coverage of rows
0..20,796,920 with no gaps, per-fold predictions equal to both `test_rows`
and the sum of per-class support, and per-class `y_true` totals matching
both the run report *and* the dataset's own class counts (17,094 `SAG.DB` /
21,436 `FRG` / 46,959 `SAG.PB` / 54,828 `SAG.PBM` / 270,680
`benign_degradation` / 20,385,924 `normal`). Recorded per-fold accuracy and
macro-F1 match the recomputation to 12 decimal places.

**Labelling (E.4).** Pooled over all folds on the original distribution, with
each averaging scheme named for what it is:

| Run | accuracy (micro) | macro F1 | weighted F1 |
|---|---:|---:|---:|
| none (baseline) | 0.9862 | 0.2794 | 0.9823 |
| smote | 0.9855 | 0.2783 | 0.9819 |
| downsample | 0.5234 | 0.1993 | 0.6766 |

This table is the reason checklist E.4 exists. The unbalanced baseline
detects **zero** attack rows, yet its *weighted* F1 is 0.9823 — a number that
would read as a near-perfect detector in a paper that did not say which
average it used. Its *macro* F1 over the same predictions is 0.2794. Every
metric reported from this pipeline must therefore carry its scheme; per-class
values (`prediction_integrity.md` §3) stay the primary evidence, macro is the
headline average, and weighted/accuracy are reported only as context for how
dominated by `normal` the pool is.

**Pairing (E.5).** All three runs predict exactly the same 20,796,921 rows
with the same ground truth, so they are pairable — a precondition for any
paired test in checklist F. The agreement tables also quantify the balancing
trade-off at row level:

| A vs. B | paired rows | only A correct | only B correct | discordant |
|---|---:|---:|---:|---:|
| none vs. downsample | 20,796,921 | 9,741,717 | 117,232 | 9,858,949 |
| none vs. smote | 20,796,921 | 20,301 | 6,186 | 26,487 |

Downsampling buys 117,232 rows the baseline got wrong at the cost of
9,741,717 it got right — an ~83:1 losing exchange, almost all of it `normal`
traffic turned into false alerts. SMOTE is not merely unhelpful but slightly
*net-negative* against doing nothing (20,301 lost vs. 6,186 gained).

## Smoke evidence (2026-08-25)

Six independent native runs were used: two seeds each for DB, FRG and PB.
PBM was excluded because only one smoke run existed.

| Check | Result |
|---|---|
| Native runs merged | 6 runs, 154,451 rows |
| Delta preparation | 154,445 rows; exactly 6 trace-boundary rows removed |
| Formula parity | 0 divergences across all 9 delta columns versus ERENO |
| Primary split | 2-fold StratifiedGroupKFold |
| Leakage audit | pass; 6/6 groups tested exactly once |
| Technical training | pass; 2 folds, 1,200 sampled rows, 1,200 held-out predictions |
| LeaveOneGroupOut | pass; 6 folds on a 600-row technical sample |
| Leave-one-event-type-out | generated and group-leakage-free (3 folds, 6/6 groups tested once); correctly flagged `open_set_diagnostic: true` and refused by the training runner (see below) |
| Test suite | 19 tests passed |

The smoke metrics are wiring diagnostics and must not be copied into the paper.

Plain 3-fold GroupKFold was also attempted and correctly blocked: both DB runs
landed in one test fold, leaving `DETERMINISTIC_BURST` absent from training.
This is why the primary protocol is stratified **and** grouped.

## Leave-one-event-type-out status

The implementation can infer DB, FRG, PB and PBM event families from
`scenario_id`, hold out all runs of one family and persist leakage-free LOETO
splits. However, in the current dataset the event family is the target attack
class. Holding it out therefore creates an open-set fold whose test class is
absent from training.

`generate_grouped_splits.py` blocks this by default. Passing
`--allow-unseen-test-classes` creates an explicitly marked open-set diagnostic;
`run_grouped_validation.py` still refuses to report it as standard multiclass
evaluation.

Confirmed on the six-run smoke dataset (`data/validation-smoke/`): generating
`--protocol leave-one-event-type-out --allow-unseen-test-classes` over the
3 available event families (DB, FRG, PB; PBM excluded, one smoke run only)
produced `splits_loeto.json`/`splits_loeto.csv` — 3 folds, group-independent
and leakage-free per `check_no_leakage.py` (`leakage_audit_loeto.json`: pass,
6/6 groups tested exactly once), each correctly marked
`"open_set_diagnostic": true` with its held-out family's attack class listed
under `test_only_labels`. Passing that split file to
`run_grouped_validation.py` is refused as designed:
`"open-set LOETO splits are diagnostic and cannot be used for standard
multiclass metrics"`. This confirms the current dataset does not have
sufficient event types for a closed-set LOETO fold — the variation is
included and wired end-to-end, but its result cannot be reported as a
standard multiclass evaluation until an event-type axis orthogonal to attack
class exists.

To make LOETO a closed-set supervised experiment, generate an event-type axis
orthogonal to attack class — for example physical fault/event categories — and
represent every event type under normal traffic and every SAG variant. Until
then, LOETO is implemented but not a publishable closed-set result **within
the attack-only scope described in this document.**

> **Update:** card C's benign-degradation controls (`benign_controls.md`)
> turned out to *be* that orthogonal axis, not a hypothetical future one.
> `infer_event_type` now maps 7 `BENIGN_<MODE>` markers alongside the 4 attack
> ones, and every benign run carries both `normal` and `benign_degradation`
> rows — so holding out one benign mechanism no longer holds out a class the
> way holding out an attack variant does. Confirmed on the 17-run benign-only
> smoke pool (`benign_controls.md` §7, C3): **7/7 LOETO folds closed-set**,
> `open_set_diagnostic: false` throughout. This resolves LOETO for the benign
> axis, but not for the attack axis: in the combined 205-run pool (120 attack
> + 85 benign, now merged — see "Full-scale results"), the 4 attack families are still each
> attack-class-exclusive, so a LOETO run over *all* 11 event types has 4
> open-set folds mixed with 7 closed-set ones, and `open_set_diagnostic: true`
> on the whole split file makes the closed-set folds hard to consume
> separately. An `--event-types` filter on `generate_grouped_splits.py`
> (restrict LOETO to a chosen subset of families, e.g. just the 7 benign ones)
> would let each axis be evaluated on its own terms; not yet implemented.

## Historical blocker (resolved 2026-09-12)

The legacy annotated dataset could not be used for the full section B results:

- `T-UNRESOLVED` mixes real traces;
- only four usable inferred groups remained;
- every group was tied to one attack class;
- the original deltas could not be safely recomputed for ambiguous rows.

This required executing the regenerated run matrix with at least five
independent seeds per attack variant, then repeating the canonical workflow
above end to end. Both have now happened — see "Full-scale results" above.
