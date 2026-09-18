# Section B — grouped validation protocol

> # ⚠ Two pools appear in this file (2026-09-16)
>
> The pool these results were first measured on was **defective**: every attack
> row had a `normal` row, in the same run, whose content features were
> bit-identical, because ERENO wrote both the legitimate publisher's stream and
> the grayhole IED's forwarded copies of the same messages
> (`label_duplication_audit.md`). It was regenerated in `d2de01c` - 265 runs,
> 11,057,478 rows, SHA-256 `3109e4d4…`.
>
> **"Model family comparison (checklist D.3)" below has been re-run in full on
> the corrected pool and is current.** Every other results section -
> "Full-scale results", "Balancing scenarios", "The threshold axis" - is marked
> where it stands: re-run, superseded by the D.3 section, or withdrawn pending
> a re-run. Nothing unmarked is from the corrected pool.
>
> The **protocol** itself - grouped splitting, the leakage audit, the run-level
> bootstrap, the threshold axis - was never affected: it describes how an
> evaluation is run, not what the labels say.
>
> What the corrected pool changed is large. The same unbalanced decision tree
> went from predicting **no attack row at all** (macro F1 0.2791) to macro F1
> 0.7043 with 0.66-0.84 recall on three of the four attack classes. Card E's
> class-imbalance verdict is withdrawn; see the D.3 section.

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
report status becomes `full_grouped_run`. On the 265-run/23.2M-row pool this
needs the whole prepared dataset in memory at once (model training inherently
requires it), so `run_grouped_validation.py` only reads the columns that
survive `feature_matrix`'s own discard sets (`load_frame(..., columns=...)`)
and casts the feature matrix to `float32` before `fit()` (sklearn's tree
splitter runs in float32 internally regardless, so this changes nothing about
the fitted model) — without both, materialising the training fold as a dense
array raised `numpy._core._exceptions._ArrayMemoryError` on a 16GB-RAM
machine. See "Full-scale results" below.

`--feature-set` (checklist D.1) runs the same protocol with one preregistered
feature group removed. It resolves against the dataset's own columns and fails
the run on any mismatch, because an ablation that drops nothing is
indistinguishable from a null result. The groups, the six runs and what each
one asks are in `ablations_baselines.md` §13.

## Full-scale results

> **Re-run on the corrected pool (2026-09-16).** The figures below are the
> unbalanced decision tree on the 265-run / 11,057,478-row pool regenerated in
> `d2de01c`. They replace the defective-pool numbers (macro F1 0.2653, zero
> attack recall), which are in this file's git history; the 205-run figures
> those in turn replaced are in `archive_205runs/`. **None of the three sets is
> comparable row for row** - the folds are redrawn and attack prevalence moved
> 0.675% → 0.932% → 1.958%.

The corrected 265-run pool (180 attack + 85 benign-degradation) carried
through the canonical workflow above:

| Step | Result |
|---|---|
| Merge (`merge_runs.py`) | 265 runs, 11,057,743 rows, all passed validation |
| Metadata (`add_experiment_metadata.py`) | 265 traces/runs, 853,310 events |
| Delta preparation (`prepare_grouped_dataset.py`) | 11,057,478 rows; 265 trace-boundary rows removed |
| Splits (`generate_grouped_splits.py`) | 5-fold stratified-group-kfold, 265 groups |
| Leakage audit (`check_no_leakage.py`) | **pass** - 265/265 groups tested exactly once |
| Label duplication (`check_label_duplication.py`) | **pass** - zero findings (this is the gate the old pool failed) |
| Training (`run_grouped_validation.py --model decision-tree`) | `full_grouped_run`, 5 folds, 11,057,478 rows |

Fold results (mean over 5 folds): **accuracy 0.9799, macro-F1 0.7043
[0.6876, 0.7174]**, interval from the run-level bootstrap. Pooled over all
folds, per class:

| Class | precision | recall | f1 |
|---|---:|---:|---:|
| `normal` | 0.9843 | 0.9987 | 0.9914 |
| `benign_degradation` | 0.9099 | 0.5617 | 0.6946 |
| `SAG.DB` | 0.8146 | 0.8421 | 0.8281 |
| `FRG` | 0.7058 | 0.6590 | 0.6816 |
| `SAG.PB` | 0.8333 | 0.6741 | 0.7453 |
| `SAG.PBM` | 0.6562 | 0.1819 | 0.2848 |

**The gap between accuracy and macro F1 is still the thing to read** - 0.9799
against 0.7043 - but it no longer hides a model that detects nothing. It is
now carried by two classes: `SAG.PBM` at 0.18 recall and `benign_degradation`
at 0.56.

Reporting discipline is unchanged and matters as much as ever: **always name
the averaging scheme.** This same run scores weighted F1 0.9771 against macro
F1 0.7043 over identical predictions. Per-class values first, macro as the
headline average, weighted and accuracy as context only.

The comparison against the other model families, and what the correction
changed, is in "Model family comparison (checklist D.3)" below.

## Balancing scenarios (checklist E)

> **Numbers here are from the defective pool (2026-09-16).** The `downsample`
> scenario has been re-run on the corrected pool for all four model families -
> those results are in "Model family comparison (checklist D.3)" below and
> **supersede every `downsample` figure in this section**. The `smote` scenario
> has **not** been re-run: its result (capped SMOTE leaves attack recall at ~0)
> was measured on a pool where nothing could raise attack recall, so it is
> **withdrawn**, not merely superseded, and card E's conclusion that rebalancing
> is the only thing that moves attack detection is withdrawn with it. What
> stays valid is the *design* below: what each scenario does, why neither aims
> for parity with `normal`, and that rebalancing touches only the train
> partition.


The run above is the **unbalanced** reference scenario. Checklist E calls for
two more, both implemented as `--balance {downsample,smote}` on
`run_grouped_validation.py`: each rebalances **only the current fold's TRAIN
partition**; the test partition is always the untouched original
distribution, so the numbers below are never inflated by evaluating on
rebalanced data.

Neither scenario aims for exact parity with the majority (`normal`) class.
`normal` is ~18M rows in a typical fold's train partition against ~13-60k for
the rarest attack class - plain SMOTE-to-parity would synthesise tens of
millions of rows, the same class of failure as the OOM this script already
hit once. Both scenarios are therefore explicitly bounded:

```bash
# downsample: every class cut to the size of the smallest class in that fold's train partition
python experiments/revision_2026/run_grouped_validation.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --preparation-report experiments/revision_2026/preparation_audit.json \
  --splits experiments/revision_2026/splits_grouped.json \
  --out-dir results/d3-decision-tree-downsample \
  --model decision-tree --balance downsample

# smote: attack classes oversampled up to 20x their own count, capped at 200k
python experiments/revision_2026/run_grouped_validation.py \
  --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
  --preparation-report experiments/revision_2026/preparation_audit.json \
  --splits experiments/revision_2026/splits_grouped.json \
  --out-dir results/grouped-validation-full-smote \
  --model decision-tree --balance smote
```

`--smote-oversample-factor` (default 20.0) and `--smote-max-target` (default
200,000) control the SMOTE cap; both are recorded in `grouped_validation_report.json`.
Requires `pip install imbalanced-learn`.

### Downsample results

Mean over 5 folds: **accuracy 0.5176 +- 0.0297, macro-F1 0.2006 +- 0.0259.**
Both numbers *drop* relative to the unbalanced baseline (0.9837 / 0.2653) -
expected, since `normal` recall itself falls to ~48-56% once its training rows
are cut to the size of the rarest attack class. The headline change is recall
on the four attack classes, previously ~0 for all of them (pooled over folds):

| Class | pooled recall | pooled precision |
|---|---:|---:|
| `SAG.DB` (`DETERMINISTIC_BURST`) | 0.9156 | 0.0348 |
| `FRG` (`FULLY_RANDOMIZED`) | 0.5718 | 0.0119 |
| `SAG.PB` (`RANDOMIC_BURST`) | 0.6365 | 0.0090 |
| `SAG.PBM` (`RANDOMIC_MESSAGE`) | 0.6026 | 0.0107 |
| `benign_degradation` | 0.6319 | 0.3283 |
| `normal` | 0.5211 | 0.9978 |

So the class-imbalance artifact is confirmed, not contradicted: the same tree
**can** separate every attack class from `normal` once training sees them at
comparable scale - it just never tried to under the unbalanced default. The
cost is precision: with `normal` recall at ~52%, roughly half of all normal
traffic is flagged as something else, so the *alert burden* at this operating
point is far too high for deployment as-is.

### SMOTE results

Mean over 5 folds: **accuracy 0.9826 +- 0.0033, macro-F1 0.2587 +- 0.0283** -
close to the unbalanced baseline, and the paired test below says it is
*significantly worse* rather than merely equal. Oversampling each attack class
to 200,000 synthetic-plus-real rows did **not** move the needle: pooled recall
stays at 0.0001 (`SAG.DB`), 0.0000 (`FRG`), 0.0012 (`SAG.PB`) and 0.0000
(`SAG.PBM`).

SMOTE's synthetic points are convex-combination neighbours of real minority
rows already present, so they add density around existing minority regions
rather than new ones. At `DecisionTreeClassifier(max_depth=8)` that extra
density still does not outweigh the accuracy gain from ignoring attack classes
altogether when they remain ~1% of the training rows even after oversampling.

### Cross-scenario comparison

| Scenario | pooled accuracy | pooled macro-F1 | attack-class recall | normal `attack_fpr` (ideal traffic) |
|---|---:|---:|---:|---:|
| none (baseline) | 0.9844 | 0.2791 | ~0.0000 (all 4 classes) | 0.00% |
| smote (capped, train-only) | 0.9831 | 0.2706 | ~0.0000-0.0012 | - |
| downsample (train-only) | 0.5238 | 0.2076 | 0.57-0.92 (all 4 detected) | **44.14%** |

**Paired over the same 265 runs** (`run_bootstrap.none.md`), macro-F1
difference against the unbalanced baseline:

| vs. baseline | difference | 95% CI | separates? |
|---|---:|---|---|
| smote | -0.0085 | [-0.0126, -0.0047] | **yes - SMOTE is worse than doing nothing** |

None of the three scenarios is a usable operating point on its own: the
unbalanced and capped-SMOTE runs never detect an attack; the downsampled run
detects every attack class but at a false-positive rate on *ideal, unimpaired*
normal traffic that would flood any deployment. **Detectability under grouped,
leakage-free validation depends entirely on how training balance is handled,
and the two balancing techniques tried sit at opposite, both-impractical
ends.** Card D.3 below adds the model-family axis to this picture.

### Metric labelling and prediction integrity (E.4/E.5)

`check_prediction_integrity.py` audits finished runs before any statistical
test is written, recomputing every number from a `numpy.bincount` confusion
matrix over the persisted `grouped_predictions.csv` - never from the `sklearn`
helpers the runner itself used, so a metric bug in the runner cannot pass its
own audit (the same decoupling rationale as `check_no_leakage.py`).

**Counts (E.5).** All ten runs on the current pool reconcile completely -
**350 checks, 0 failures**: every `row_index` predicted exactly once, full
coverage of rows 0..23,226,529 with no gaps, per-fold predictions equal to both
`test_rows` and the sum of per-class support, and per-class `y_true` totals
matching both the run report *and* the dataset's own class counts (50,782
`SAG.DB` / 63,976 `FRG` / 46,959 `SAG.PB` / 54,828 `SAG.PBM` / 270,680
`benign_degradation` / 22,739,305 `normal`).

**Labelling (E.4).** Pooled over all folds on the original distribution, each
averaging scheme named for what it is:

| Run | accuracy (micro) | macro F1 | weighted F1 |
|---|---:|---:|---:|
| none (baseline) | 0.9844 | 0.2791 | 0.9792 |
| smote | 0.9831 | 0.2706 | 0.9780 |
| downsample | 0.5238 | 0.2076 | 0.6756 |

This table is the reason checklist E.4 exists. The unbalanced baseline detects
**essentially zero** attack rows, yet its *weighted* F1 is 0.9792 - a number
that would read as a near-perfect detector in a paper that did not say which
average it used. Its *macro* F1 over the same predictions is 0.2791. Every
metric reported from this pipeline must carry its scheme; per-class values
stay the primary evidence, macro is the headline average, and
weighted/accuracy are context for how dominated by `normal` the pool is.

**Pairing (E.5).** All ten runs predict exactly the same 23,226,530 rows with
the same ground truth, so they are pairable - a precondition for the paired
tests in checklist F and for the paired bootstrap used throughout card D.3.


## Model family comparison (checklist D.3)

> **Re-run in full on the corrected pool (2026-09-16).** Every number in this
> section comes from the 265-run / 11,057,478-row pool regenerated in `d2de01c`
> (SHA-256 `3109e4d4…`), which no longer emits the legitimate publisher's
> stream alongside the grayhole's forwarded copies. The defective-pool figures
> this replaces are in this file's git history. **The conclusions did not
> survive the correction** - see "What D.3 settles" below.

Card E left an open question: is never predicting an attack class a property
of `DecisionTreeClassifier(max_depth=8)`, or of every model at this class
balance? D.3 answers it by running four families - decision tree, XGBoost,
Random Forest, logistic regression - **at library defaults, with no tuning**
(tuning is D.4), on the same persisted folds, in both the `none` and
`downsample` scenarios. Nine runs, ~5 h wall clock. The per-fold train cap the
Random Forest needs is in `ablations_baselines.md` §7; the run matrix and the
pre-registered champion criterion are in its §8.

On the corrected pool the question has a different answer, because its premise
is gone: **the unbalanced models detect attacks.** What card E measured as a
class-imbalance wall was the duplicate legitimate stream making every attack
row indistinguishable from a `normal` row in the same run.

### Unbalanced (`none`)

Pooled over all folds, original-distribution test partitions. Intervals are the
run-level bootstrap (`run_bootstrap.none_v2.md`, 2,000 replicates over the 265
runs); `attack_fpr` is the fraction of **ideal** `normal` rows
(`impairment_mode=NONE`) predicted as one of the four attack classes
(`benign_confusion.v2-*.md`).

| Model | cap | accuracy (micro) | **macro F1 [95% CI]** | weighted F1 | mean attack recall | ideal-`normal` attack_fpr |
|---|---:|---:|---|---:|---:|---:|
| decision-tree | - | 0.9799 | 0.7043 [0.6876, 0.7174] | 0.9771 | 0.5893 | 0.02% |
| decision-tree | 4M | 0.9798 | 0.7033 [0.6870, 0.7166] | 0.9770 | 0.5858 | 0.02% |
| **xgboost** | - | 0.9815 | **0.7310 [0.7141, 0.7451]** | 0.9793 | 0.6338 | **0.01%** |
| random-forest | 4M | 0.9815 | **0.7355 [0.7183, 0.7494]** | 0.9799 | 0.6408 | 0.12% |
| logistic-regression | - | 0.9622 | 0.3677 [0.3486, 0.3879] | 0.9496 | 0.2248 | 0.08% |

Per-class recall, the numbers the macro average hides:

| Model | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` |
|---|---:|---:|---:|---:|
| decision-tree | 0.8421 | 0.6590 | 0.6741 | 0.1819 |
| xgboost | 0.8828 | 0.6529 | 0.7272 | 0.2721 |
| random-forest (4M) | 0.8415 | 0.5954 | 0.7296 | **0.3965** |
| logistic-regression | 0.5953 | **0.0000** | 0.3032 | 0.0007 |

Three things to read off these tables:

- **It was never a decision-tree limit, and it was never only imbalance.** The
  same unbalanced decision tree that found no attack row at all on the
  defective pool reaches 0.66-0.84 recall on three of the four classes here, at
  a 0.02% false-positive rate on ideal traffic. The 2026-09-13 conclusion that
  "no model family at library defaults escapes the imbalance" measured the
  defect, not the imbalance.
- **`SAG.PBM` is the hard class** for every family (0.18-0.40 recall), and
  `FRG` is the one logistic regression cannot touch at all (0.0000 - it
  predicts that class for no row).
- **Capacity still matters, but far less than it appeared to.** The 4M cap
  moves the decision tree by -0.0010 macro F1, 95% CI [-0.0029, +0.0011],
  which does not separate - so the cap is not producing any difference between
  families here.

### Balanced (`downsample`)

| Model | accuracy (micro) | **macro F1 [95% CI]** | mean attack recall | ideal-`normal` attack_fpr | `normal` recall |
|---|---:|---|---:|---:|---:|
| **xgboost** | 0.7880 | **0.4594 [0.4382, 0.4764]** | 0.8475 | 8.67% | 0.7850 |
| random-forest | 0.8048 | 0.4489 [0.4283, 0.4659] | 0.8321 | 8.16% | 0.8032 |
| decision-tree | 0.7608 | 0.4376 [0.4175, 0.4543] | 0.8254 | 10.46% | 0.7579 |
| logistic-regression | 0.6931 | 0.3131 [0.2983, 0.3265] | 0.6905 | 18.13% | 0.6992 |

Downsampling still buys attack recall (0.69-0.85 against 0.22-0.64) and still
pays for it, but the price has changed shape: the ideal-traffic false-positive
rate is **8-18%**, not the 37-44% the defective pool reported. It is no longer
an absurd operating point - it is merely a worse one than `none`, and that is a
claim the threshold axis has to settle rather than the argmax.

### Champion: XGBoost

Criterion fixed before execution (`ablations_baselines.md` §8): mean macro F1
in `downsample`, then per-class attack recall, then ideal-`normal`
`attack_fpr`. On the corrected pool that criterion picks **XGBoost**, and the
paired bootstrap - both models scored on the same redrawn runs, so the
run-to-run variation they share cancels - is what makes it sayable:

| Scenario | A | B | B - A macro F1 | 95% CI | separates? |
|---|---|---|---:|---|---|
| `downsample` | decision-tree | **xgboost** | **+0.0218** | [+0.0184, +0.0252] | yes |
| `downsample` | decision-tree | random-forest | +0.0113 | [+0.0064, +0.0165] | yes |
| `downsample` | decision-tree | logistic-regression | -0.1245 | [-0.1328, -0.1136] | yes |
| `none` | decision-tree | xgboost | +0.0267 | [+0.0208, +0.0322] | yes |
| `none` | decision-tree | random-forest (4M) | +0.0312 | [+0.0219, +0.0395] | yes |
| `none` | decision-tree | logistic-regression | -0.3366 | [-0.3572, -0.3127] | yes |
| `none` | decision-tree | decision-tree (4M cap) | -0.0010 | [-0.0029, +0.0011] | **no** |
| `none` | **xgboost** | random-forest (4M) | +0.0045 | [-0.0036, +0.0117] | **no** |

The last row is the one that decides. In `none`, XGBoost and the Random Forest
are **indistinguishable on macro F1** - the interval straddles zero - so argmax
accuracy does not pick between them. Three things do:

1. **Ranking.** Paired on average precision (`pr_curves_d3_v2.md`), the Random
   Forest is **worse**: -0.0394 AP on `ANY_ATTACK`, 95% CI [-0.0471, -0.0321],
   and it separates on `SAG.DB`, `FRG` and `SAG.PB` individually. It wins on
   nothing except `SAG.PBM`, where the two do not separate (+0.0118 [-0.0092,
   +0.0300]).
2. **It needs no cap.** Every Random Forest number here carries §7's 4M-row
   train cap; XGBoost trains on the full partition.
3. **Cost.** XGBoost 37 min/run against the capped forest's 56 min.

**The defensible sentence is "XGBoost and the Random Forest are
indistinguishable at the argmax, and XGBoost is chosen because it ranks
strictly better, needs no train cap and costs less", not "XGBoost is the more
accurate model".**

### The scenario choice inverts: `none` is the one to report

On the defective pool `downsample` was the only scenario that detected
anything, so card D.1's ablation was scoped to it. That is now backwards.
Paired over the same runs, `downsample` - `none` for the champion is **-0.0121
AP on `ANY_ATTACK`, 95% CI [-0.0172, -0.0071]**, and `none` also holds a 0.01%
ideal-traffic false-positive rate against 8.67%. Rebalancing moves a threshold
and throws away training rows to do it; the curve says that is a bad trade
here. **D.1 therefore ablates in `none`** (`ablations_baselines.md` §12).

### `FRG` is indistinguishable from benign congestion loss, by construction

This is a property of the phenomenon, not of the pool, and the decision
(2026-09-16) is to **keep the class and report the null**.

Fifteen of the 45 `FULLY_RANDOMIZED` runs are **byte-identical in payload** to
the `BENIGN_CONGESTION_LOSS` control runs of the same loss rate and seed - 250
distinct payloads across 265 runs, and the 15 collisions are exactly those
pairs. That is not a generation bug: a grayhole that drops uniformly at random
and a link that loses packets to congestion are the same stochastic process.
Nothing in the chain catches it - `merge_runs.py` includes `class` in its
payload fingerprint, so the two fingerprints differ, and
`check_label_duplication.py` is intra-run. Extending that gate to cross-run
payload collisions is the next dataset-integrity task.

The measurement agrees with the construction, in three independent places:

- `FRG` has the **second-lowest AP** of the four classes (0.6395 [0.5484,
  0.7284]) despite carrying the most positive rows.
- At a 10-alerts-per-10,000 budget, **99.4% of the champion's `FRG` false
  alarms are `benign_degradation` rows**, against 26.2% for `SAG.DB`.
- The champion classifies **73.33% of `CONGESTION_LOSS` benign rows as an
  attack**, while its false-positive rate on ideal `normal` traffic is 0.01%
  (`benign_confusion.v2-xgboost-none.md`). The model is not confusing attack
  with normal traffic; it is confusing attack with *the one benign mechanism
  that is physically the same process*.

The claim the paper can make is therefore bounded and, stated this way,
stronger than a uniform one: **a uniformly-random grayhole is indistinguishable
from congestion loss by construction; the three structured variants are not.**
`SAG.DB` and `SAG.PB` reach AP 0.88 and 0.86 with 13.5%/9.9% of their false
alarms coming from benign degradation at the same budget. Card C's benign
controls are what carry this claim; `benign_controls.md` §8 records the
decision.

### How many attack rows are actually being counted

The corrected pool holds **216,529 attack rows, 1.958% of 11,057,478** - nearly
the same attack rows as before over half the total rows, because the duplicate
legitimate stream is what was removed:

| Class | rows | % of pool | independent runs |
|---|---:|---:|---:|
| `SAG.DB` | 50,779 | 0.459% | 45 |
| `FRG` | 63,963 | 0.579% | 45 |
| `SAG.PB` | 46,959 | 0.425% | 45 |
| `SAG.PBM` | 54,828 | 0.496% | 45 |
| `benign_degradation` | 270,642 | 2.448% | 85 |
| `normal` | 10,570,307 | 95.594% | 245 |

`normal` is carried by 245 of the 265 runs rather than all of them: 20 runs
contain no ideal `normal` rows at all. Every per-class interval above is a
bootstrap over *runs*, so 45 is the effective sample size behind each attack
class, not ~50,000.

### What D.3 settles, and what it does not

- **Card E's imbalance verdict is withdrawn.** Unbalanced training detects
  three of the four attack classes at 0.65-0.88 recall and 0.01% ideal-traffic
  false positives. The zero-detection result was an artifact of the defective
  pool, not of the class balance.
- **Family matters less than the correction did.** Across XGBoost, the Random
  Forest and the decision tree the macro-F1 spread in `none` is 0.031 - one
  ninth of what removing the duplicate stream moved (0.279 → 0.704 for the same
  tree). Logistic regression remains the honest floor at 0.3677.
- **`SAG.PBM` is not solved by any family.** 0.18-0.40 recall, AP 0.41 at best,
  and the label-semantics audit says why: most of its discards happen at a
  state boundary, where a single message carries no information about whether a
  discard preceded it (`label_duplication_audit.md` §7). This is the class the
  window redesign exists for.
- **Detectability is now a real claim, with a real caveat.** At a
  1-alert-per-100-messages budget the champion reaches `ANY_ATTACK` recall
  0.4833 [0.4349, 0.5368] at precision 0.9467 [0.9254, 0.9658], and 93.9% of
  the remaining false alarms are benign degradation rather than ideal traffic.
  What the paper may *not* yet claim is that this separates attack from benign
  packet loss in general - see the `FRG` section above.
- **The label is still per-message, and that is still the open design
  question.** `label_duplication_audit.md` §7 measures that a meaningful share
  of attack rows are unidentifiable *in principle* from a single message. Every
  recall number here is bounded by that, D.3 included.
- Cost note for D.1: the champion's `none` run takes **37 min**, so the six-run
  ablation is ~3.7 h of serial compute.

### The Random Forest argmax disagreement, and what it was (2026-09-17)

`prediction_integrity_d3_v2.md` runs 351 checks over the nine runs: **0
failures**. Until 2026-09-17 it reported two, both of the same kind and both on
Random Forest - `argmax(posterior)` not reproducing `y_pred` on **7 rows of
11,057,478** (`none-cap4m`) and **17 rows** (`downsample`). The cause is now
confirmed, and it was in the audit's premise rather than in either run.

A scored run takes a fold's hard labels from `argmax(proba)` and verifies that
choice against `model.predict` on the fold's first block; on any disagreement
it falls back to `predict` for the whole fold and records
`fell_back_to_predict` in its report. **Random Forest triggered that fallback
in 9 of its 10 folds** - the exception, `none-cap4m` fold-03, contributes none
of the 24 rows. On a fallback fold `y_pred` is therefore an argmax over
scikit-learn's float64 posteriors, while the persisted scores are the float32
copy the writer stores.

All 24 rows are **exact ties at float32**: top-1 and top-2 bit-identical, gap
0.0. And on all 24 the audit's `numpy.argmax` took the *lower* class index
while `predict` had kept the higher one - the only direction that can produce a
mismatch, since on a genuine tie both pick the lower. The two runs carry 1,224
(`none-cap4m`) and 16,420 (`downsample`) rows tied at float32, so the 24 are
the few whose float64 values were not tied *as well*, by less than float32 can
represent. `v2-xgboost-none` has no tied row at all and `v2-decision-tree-none`
has 31, neither with any disagreement: their labels are that same argmax.

The audit now forgives a mismatch **only** where the report marks that fold
`fell_back_to_predict` **and** the two leading persisted posteriors are exactly
equal. A strict winner that disagrees with `y_pred` still fails, in any fold,
and so does a tie in a fold that did not fall back. At float32 - the precision
every threshold number in `pr_curves_d3_v2.md` is computed from - the model
expressed no preference on those rows, so there is nothing left to delimit:
**Random Forest figures are citable.**

### Runner reproducibility

D.3's first pass required reworking how `run_grouped_validation.py` loads data
and predicts (float32/dictionary cast at read time, the DataFrame released
before the first `fit()`, prediction in bounded blocks). Both card-E
decision-tree runs were re-executed on the reworked runner as regression checks
and `grouped_predictions.csv` came back **byte-identical by SHA-256** in both
scenarios over all 20,796,921 rows of the then-current pool. The rework changed
memory and wall clock only. That check predates the pool correction and was not
repeated on it; what binds the current runs to the current pool is the SHA-256
dataset hash every artifact in the chain carries.


## Feature-group ablation (checklist D.1)

Seven runs on the corrected pool — the champion plus six ablations, each
removing exactly one preregistered feature group, all XGBoost/`none` from the
same persisted splits. 136 min of serial compute; `check_prediction_integrity.py`
reports 273 checks, 0 failures, all seven pairable. The groups and the guard
that makes a null ablation trustworthy are in `ablations_baselines.md` §13; the
full result and its consequences are in its §14.

| Run | n feat | AP `ANY_ATTACK` | paired vs reference | 95% CI | separates? |
|---|---:|---:|---:|---|---|
| reference (`all`) | 40 | 0.8329 | — | — | — |
| `no-goose-header` | 33 | 0.8327 | -0.0002 | [-0.0005, +0.0001] | **no** |
| `no-electrical` | 22 | 0.8327 | -0.0001 | [-0.0005, +0.0003] | **no** |
| `no-counters` | 38 | 0.8306 | -0.0023 | [-0.0051, +0.0004] | **no** |
| `no-absolute-time` | 37 | 0.8304 | -0.0025 | [-0.0050, -0.0002] | yes |
| `no-sequence` | 36 | 0.8240 | -0.0089 | [-0.0128, -0.0055] | yes |
| **`no-delta`** | 31 | **0.1101** | **-0.7227** | **[-0.7478, -0.6956]** | **yes** |

**The result is binary, not a ranking.** The nine within-trace delta features
carry essentially all of the detection: without them AP falls to 0.1101 and the
1%-budget operating point goes from recall 0.4833 at precision 0.9467 to
**0.0903 at precision 0.1768**. Every other group is free or nearly so — the 18
electrical and 7 GOOSE-header columns are indistinguishable from the reference,
and at the 1% budget all six non-`no-delta` configurations sit inside each
other's intervals at recall 0.482-0.486.

**Within the deltas it is timing, not the sequence gap.** `no-sequence` removes
`StNum`, `SqNum`, `stDiff` and `sqDiff` — every piece of sequence information —
and still recovers to within 1% of the reference on the seven remaining deltas;
`no-delta` removes those seven as well and collapses. So the non-counter deltas
are *sufficient* and the sequence columns *redundant given them*. That is
consistent with `SAG.PBM`'s weakness (its discards fall at state boundaries
where `SqNum` resets) and with the `FRG`/congestion collision (random drops
leave a stretched interval, not a counter pattern).

### Which half of the deltas (2026-09-17, two runs outside the preregistered six)

The seven non-counter deltas mix two quantities — the interval between
messages and the change in the frame itself — so the paragraph above names the
carrier by elimination. Splitting the group in two and ablating each half
measures it directly (~40 min; `ablations_baselines.md` §15,
`pr_curves_d1_split.md`, `prediction_integrity_d1_split.md`: 195 checks, 0
failures). `no-delta` still drops the same nine columns, so the table above is
unchanged.

| Run | n feat | AP `ANY_ATTACK` | paired vs reference | 95% CI | separates? |
|---|---:|---:|---:|---|---|
| `no-size-state-deltas` | 36 | 0.8325 | -0.0004 | [-0.0007, -0.0001] | yes, and meaningless |
| `no-timing-deltas` | 37 | 0.7683 | **-0.0645** | **[-0.0832, -0.0501]** | **yes** |

**The three timing deltas** (`timestampDiff`, `tDiff`, `timeFromLastChange`)
**are the carriers.** Dropping them separates on every class (`FRG` -0.1293,
`SAG.PBM` -0.0673, `SAG.PB` -0.0500, `SAG.DB` -0.0387) and is the only
ablation besides `no-delta` that moves an operating point: at the 1% alert
budget recall falls 0.4833 → 0.4512 and precision 0.9467 → 0.8791. The four
size/state deltas are indistinguishable from the reference on every individual
class and reproduce the 1% operating point to the fourth decimal — 29 of the
40 columns are now demonstrably free.

Neither half alone reproduces the collapse (-0.0645 and -0.0004 against
-0.7227), so the deltas remain **jointly** necessary and no minimal sufficient
set is being claimed; a leave-one-group-out design cannot name one. The
operational statement is narrower and holds: the timing columns are the only
sub-group whose removal an operator would notice, which is what card F should
explain and what D.2's interval-threshold baseline has to beat.

Two things this does **not** say. It is not leakage: the deltas are computed
strictly within a trace, boundary rows dropped, and a live monitor can compute
them from messages it has already seen. And it does not rank the surviving
groups — it says the signal is not in them.

What it does say is that `prepare_grouped_dataset.py`'s within-trace delta
computation is now **load-bearing for every detection number in this revision**.
A bug there would not degrade the result; it would be the result.

## Rule-based baseline (checklist D.2)

The reviewer's question behind card D is not whether the model scores well but
whether it earns its complexity. Six single-threshold rules answer it, each
one feature and one comparison, each with its threshold fitted **only on the
train partition of the fold it is scored on**, from the same persisted
`splits_grouped.json` the learned runs consume. `run_rule_baseline.py` writes
what `run_grouped_validation.py` writes, so the baseline is audited by the
same scripts under the same invariants - `prediction_integrity_d2.md` reports
7 runs x 39 checks, 0 failures, every rule pairable with the champion row for
row. Full result and the projection caveat: `ablations_baselines.md` §16.

| Run | recall | precision | F1 | alert rate | AP `ANY_ATTACK` | paired vs champion |
|---|---:|---:|---:|---:|---|---:|
| champion (`v2-xgboost-none`) | 0.4833\* | 0.9467\* | — | 1%\* | 0.8329 [0.8037, 0.8618] | — |
| **`interval-timestamp`** | 0.4416 | 0.6229 | 0.5144 | 1.45% | **0.2411** [0.1895, 0.3052] | **-0.5918** [-0.6385, -0.5392] |
| `stnum-gap` | 0.4116 | 0.3981 | 0.4025 | 2.04% | 0.2192 [0.1754, 0.2712] | -0.6137 [-0.6509, -0.5708] |
| `interval-t` | 0.4714 | 0.1497 | 0.2258 | 6.29% | 0.1010 [0.0822, 0.1237] | -0.7319 [-0.7588, -0.7035] |
| `sqnum-gap` | 0.3559 | 0.1383 | 0.1982 | 5.20% | 0.0451 [0.0338, 0.0592] | -0.7878 [-0.8186, -0.7557] |
| `delay` | 0.8882 | 0.0246 | 0.0478 | 72.6% | 0.0175 [0.0147, 0.0211] | -0.8153 [-0.8439, -0.7867] |
| `time-since-change` | 1.0000 | 0.0202 | 0.0395 | 99.98% | 0.0127 [0.0105, 0.0153] | -0.8202 [-0.8489, -0.7921] |

\* the champion's row is its cross-fold operating point at a 1% alert budget,
not an argmax; each rule's is its own calibrated threshold. Every paired
difference separates.

**The model earns its complexity.** Against the best single threshold it is
worth -0.5918 AP paired, ~3.5x, and at a matched 1% budget it recovers recall
0.4833 at precision 0.9467 where the rule manages 0.2980 at 0.5842. **And the
baseline is a real detector**: AP 0.2411 against a 0.0196 prevalence floor is
12.3x chance, so the comparison is not a straw man - which is what makes the
verdict worth reporting.

Two secondary findings. The preregistered `sqNum` gap detector loses to the
interval rule by a factor of 5.3 in AP (paired -0.1960 [-0.2535, -0.1481]),
confirming what the feature ablation predicted - kept as an arm and measured
rather than dropped on that prediction. But `stnum-gap` is *indistinguishable*
from the interval rule (paired -0.0219 [-0.0884, +0.0452]), and the two carry
different classes: the interval rule ranks `FRG` at 25x its floor, the state
counter at 1x. Uniformly random loss leaves a stretched interval and no
counter pattern, which is the same mechanism behind the `FRG`/congestion
collision in `benign_controls.md` §8.

**A rule run's per-class rows are not a per-class result.** A rule has one
score and cannot name a family, so the emitted posterior carries it on one
designated attack class and exactly zero on the other three; those three come
back at the prevalence floor by construction. Every report records this under
`rule.per_class_curves`.

## The threshold axis (checklist D.5, 2026-09-13)

> **Numbers here are from the defective pool (2026-09-16).** The curves have
> been recomputed for all nine D.3 runs on the corrected pool
> (`pr_curves_d3_v2.md`); the champion's are summarised in "Model family
> comparison (checklist D.3)" above, and the headline moved a long way - AP on
> `ANY_ATTACK` is 0.8329 [0.8037, 0.8618] against the 0.563-at-budget figures
> below. **The method in this section is unaffected and is what the corrected
> curves were produced with**: the argmax/prior arithmetic, cross-fold
> threshold calibration, the alert-budget framing and the run-level bootstrap.
> One conclusion does survive verbatim and is worth flagging: downsampling
> *hurts* the ranking, on both pools.


Everything above this section reports `argmax(p)`. That is **one point** on a
curve, and which point it is was never a modelling decision - it is whatever
the training partition's class prior implies. This is not a detail: it is why
`none` and `downsample` read as opposite verdicts on the same question.

### The arithmetic that makes the two scenarios one result

`downsample` trains at 1/6 per class. Against the pool's own 0.932% attack
prevalence that multiplies the attack-vs-rest odds by

    (4/6) / (2/6)  /  (0.00932 / 0.99068)  =  2.0 / 0.009407  =  **212.6x**

so at a fixed argmax every row whose honest attack posterior exceeds
**1/212.6 = 0.47%** is alerted. That is where the reported 39-44%
false-positive rate on *ideal* `normal` traffic comes from: a threshold two
orders of magnitude looser than 0.5, set by the balancing rather than by
anyone. `none` sits at the other end of the same curve. Neither is an
operating point anybody chose - which is a separate question from whether a
better one exists, and the tables below answer that one too (mostly: no).

### How the curve is recovered

`run_grouped_validation.py --save-scores` persists the per-row posteriors to
`grouped_scores.parquet`; `grouped_pr_curves.py` reads them and reports
average precision (threshold-free) plus operating points at a fixed **alert
budget**. Three properties make the numbers reportable:

- **Thresholds are calibrated on the other folds**, never on the fold being
  scored. Folds are group-disjoint, so those rows are a legitimate calibration
  set. `--threshold-selection pooled` measures the optimism this avoids; it is
  not what the tables below use.
- **Intervals resample runs**, at the unit and with the `not estimable` floor
  `bootstrap_run_intervals.py` argues for.
- **Nothing else about the runs changed.** `grouped_predictions.csv` is
  **byte-identical by SHA-256** to the card-D.3 runs over all 23,226,530 rows,
  in both scenarios (`d5-xgboost-downsample` = `d3-xgboost-downsample`,
  `73613339c8ca58a5...`; `d5-xgboost-none` = `d3-xgboost-none`,
  `542dd05d09a86534...`) - across both `--save-scores` *and* the streaming
  loader those runs were the first to use. `check_prediction_integrity.py`
  independently re-derives `argmax(posterior)` over every scored row rather
  than trusting the runner's own first-block check: 39 checks per run, 0
  failures (`prediction_integrity_d5.md`).

### Result: both scenarios, each on its own curve

`pr_curves.md`. AP is read against the **prevalence** column - a
zero-information detector scores exactly its own prevalence, so that is the
floor, not 0. `--prior auto` leaves the unbalanced run's posteriors alone and
corrects the balanced one back to the deployment prior, which is what makes
the two comparable at all.

| Target | rows | prevalence | AP, `none` [95% CI] | AP, `downsample` [95% CI] | best lift |
|---|---:|---:|---|---|---:|
| `SAG.DB` | 50,782 | 0.219% | **0.0620 [0.0446, 0.0823]** | 0.0593 [0.0430, 0.0759] | 28.4x |
| `SAG.PBM` | 54,828 | 0.236% | **0.0298 [0.0213, 0.0393]** | 0.0247 [0.0178, 0.0325] | 12.6x |
| `SAG.PB` | 46,959 | 0.202% | **0.0231 [0.0154, 0.0333]** | 0.0187 [0.0138, 0.0248] | 11.4x |
| `FRG` | 63,976 | 0.275% | **0.0255 [0.0172, 0.0352]** | 0.0209 [0.0140, 0.0294] | 9.3x |
| `ANY_ATTACK` | 216,545 | 0.932% | **0.0724 [0.0617, 0.0844]** | 0.0625 [0.0535, 0.0730] | 7.8x |

**Every interval clears its prevalence floor by 8-28x**, so the features do
carry real signal about these attacks. The model that "never predicts an
attack class" carries the most of it.

### The comparison that settles card E

Card E asked whether the zero attack recall on the unbalanced pool meant the
attacks were undetectable, and answered it by rebalancing until the model
predicted them. The paired bootstrap - both configurations scored on the same
redrawn runs, each keeping its own cross-fold thresholds - says the
rebalancing bought nothing:

| Target | `downsample` - `none` AP | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0027 | [-0.0119, +0.0060] | no |
| `FRG` | **-0.0046** | [-0.0070, -0.0021] | **yes** |
| `SAG.PB` | **-0.0044** | [-0.0090, -0.0013] | **yes** |
| `SAG.PBM` | **-0.0051** | [-0.0076, -0.0030] | **yes** |
| `ANY_ATTACK` | **-0.0100** | [-0.0135, -0.0069] | **yes** |

**Downsampling does not add information; it moves a threshold, and it makes
the ranking measurably worse** on three of the four attack classes and on the
binary view. That is not surprising once it is stated: the balanced model fits
~96k rows per fold against the unbalanced model's ~18.6M, so it pays for its
convenient operating point with the training data it threw away.

The consequence for the paper is direct. "The unbalanced model detects no
attacks" and "the balanced model detects attacks at a 39% false-positive rate"
are **the same model's score read at two thresholds**, and the version that
detects nothing is the better detector of the two. Neither sentence should
appear without the curve.

### Where the published argmax sits on its own curve

The comparison has to be made at a **matched alert rate**, and per target: the
argmax alerts on 5.33% of rows as a `SAG.DB` detector but on 42.06% as a
binary attack detector, so quoting one class's recall against the other's
burden compares nothing. Same model (`downsample`), same folds, same rows;
only the threshold moves:

| Target | argmax alert rate | argmax recall | curve recall at the same rate [95% CI] | delta |
|---|---:|---:|---|---:|
| `SAG.DB` | 5.33% | 0.9024 | 0.9297 [0.9039, 0.9553] | +0.027 |
| `FRG` | 13.70% | 0.6292 | **0.7541 [0.6649, 0.8310]** | **+0.125** |
| `SAG.PB` | 11.04% | 0.6765 | 0.6986 [0.6585, 0.7393] | +0.022 |
| `SAG.PBM` | 11.99% | 0.5716 | **0.7543 [0.6973, 0.8057]** | **+0.183** |
| `ANY_ATTACK` | 42.06% | 0.9529 | 0.9550 [0.9388, 0.9688] | +0.002 |

The prior shift explains *where* the argmax lands - at a very loose threshold -
but the argmax is close to its own curve's frontier once the burden is
matched. The 39-44% false-positive rate is therefore **not mainly a
miscalibration**: at that alert rate the model really is near the best it can
do, and the curve is simply bad. Recalibration buys a real but modest +0.02 to
+0.18 recall, concentrated in `FRG` and `SAG.PBM`.

### The trade-off, finally stateable

`none`, the better ranker, across the alert budgets an operator might set:

| Budget | `SAG.DB` recall / precision | `ANY_ATTACK` recall / precision |
|---|---|---|
| 1,000 alerts per 10,000 msgs (10%) | 0.9947 / 0.022 | 0.5625 / 0.054 |
| 100 per 10,000 (1%) | 0.2960 / 0.066 | 0.1337 / 0.129 |
| 10 per 10,000 (0.1%) | 0.0514 / 0.113 | 0.0179 / 0.168 |
| 1 per 10,000 (0.01%) | 0.0024 / 0.049 | 0.0010 / 0.089 |

### What still does not become a detector

- The answer to "can this run quietly?" is **no**. Dropping `SAG.DB` to an
  operator-plausible 0.1% alert rate takes recall to 0.051; the other three
  classes sit at 0.022-0.037 there. Two orders of magnitude of alert burden
  buy roughly one order of magnitude of recall, all the way down.
- Precision never leaves 1-17% anywhere on any curve, for any class, at any
  budget, in either scenario.
- **Per-class scores rank better than the pooled attack score.** At the 10%
  budget the four one-vs-rest detectors reach 0.995/0.674/0.663/0.698 while
  `ANY_ATTACK` - the sum of the four posteriors - reaches only 0.563. An
  operator should run four detectors, not one binary one; pooling the classes
  into a single score destroys signal. This is a design recommendation the
  argmax evaluation could not have produced.
- The revision's finding is therefore not "undetectable" but **"detectable,
  and not at any alert burden a substation can absorb"** - a sharper and more
  defensible claim than either argmax point supported, and one that now rests
  on a threshold-free statistic with run-level intervals.

### An open question this section deliberately does not answer

`grouped_pr_curves.py` splits false alarms by the *class* that produced them,
not by `impairment_mode`. In aggregate `benign_degradation` is alerted at
0.03x-1.19x the rate of `normal` across the four budgets, which looks like the
card-C confound disappearing - **but that aggregate is not readable**, because
the seven impairment mechanisms behave nothing alike at the argmax
(`CONGESTION_LOSS` 92.45% attack_fpr against `DELAY` 0.61%) and three quiet
modes carry 43% of the `benign_degradation` rows. Whether the confound
survives calibration needs the per-mode rejoin `benign_confusion_report.py`
owns, applied to thresholded rather than argmax predictions. Until that runs,
**no claim about the benign confound may be updated from this section.**

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
> axis, but not for the attack axis: in the combined 265-run pool (180 attack
> + 85 benign — see "Full-scale results"), the 4 attack families are still each
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
