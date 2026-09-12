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
