# Section B — grouped validation protocol

> # ⚠ Results below are measured on a defective dataset (2026-09-13)
>
> **Every attack row in the regenerated pool has a `normal` row, in the same
> run, whose content features are bit-identical** - 216,547 of 216,547, in the
> raw ERENO output. The attack class is the grayhole IED's *forwarded,
> unmodified copies* of the legitimate publisher's messages, so what any model
> here separates is which copy a row is, not whether an attack occurred. A
> real monitor sees one stream, and the copy it sees is the one labelled
> `normal`. See `label_duplication_audit.md`; the root cause is that ERENO
> models a *dropping* attack as an *emitting* one, which is a design mismatch
> rather than a coding slip.
>
> The models are not merely fitting noise - the champion scores 7.8x chance
> and does not reduce to any single feature or to the duplicate marker
> (`feature_signal.md`). It is the *target* that is wrong, not the fit.
>
> The **protocol** in this file - grouped splitting, the leakage audit, the
> run-level bootstrap, the threshold axis - is unaffected: it describes how an
> evaluation is run, not what the labels say. Re-running cards D and E on a
> regenerated pool is compute, not rework.

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

## Full-scale results

> **Pool extended 2026-09-13 (205 -> 265 runs).** The numbers below are from
> the current pool. The 205-run figures they replace are preserved in
> `archive_205runs/` and in this file's git history. The extension added 10
> seeds to `DETERMINISTIC_BURST` and `FULLY_RANDOMIZED` (3 cells each, +60
> runs) because those two classes had only 15 independent runs and a fold's
> test partition could hold a single one - see "How many attack rows are
> actually being counted" below. **Results before and after are not comparable
> row for row**: the folds are redrawn over 265 groups, and attack prevalence
> moved from 0.675% to 0.932%.

The 265-run pool (180 attack + 85 benign-degradation) carried through the
canonical workflow above:

| Step | Result |
|---|---|
| Merge (`merge_runs.py`) | 265 runs, 23,226,795 rows, all passed validation |
| Metadata (`add_experiment_metadata.py`) | 265 traces/runs, 950,315 events |
| Delta preparation (`prepare_grouped_dataset.py`) | 23,226,530 rows; 265 trace-boundary rows removed |
| Splits (`generate_grouped_splits.py`) | 5-fold stratified-group-kfold, 265 groups |
| Leakage audit (`check_no_leakage.py`) | **pass** - 265/265 groups tested exactly once |
| Training (`run_grouped_validation.py --model decision-tree`) | `full_grouped_run`, 5 folds, 23,226,530 rows |

Fold results (mean over 5 folds): **accuracy 0.9837 +- 0.0038, macro-F1
0.2653 +- 0.0307.** The gap between those two numbers is the headline finding,
not a detail - pooled over all folds, per class:

| Class | precision | recall | f1 |
|---|---:|---:|---:|
| `normal` | 0.9857 | 0.9985 | 0.9921 |
| `benign_degradation` | 0.8249 | 0.5823 | 0.6827 |
| `SAG.DB` / `FRG` / `SAG.PB` / `SAG.PBM` | ~0 | **~0.0000** | **~0.0000** |

The confusion matrix (`benign_confusion.md`) confirms this is not "mostly
misses, sometimes hits": across all 23.2M rows the model predicts an attack
class for essentially no row at all.

**This is a class-imbalance artifact of running the model completely
unbalanced, not evidence about SAG detectability**, and not a leakage problem -
`check_no_leakage.py` passed. Card D.3 has since confirmed it is not specific
to this model either (see "Model family comparison" below).

## Balancing scenarios (checklist E)

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

Card E left an open question: is never predicting an attack class a property
of `DecisionTreeClassifier(max_depth=8)`, or of every model at this class
balance? D.3 answers it by running four families - decision tree, XGBoost,
Random Forest, logistic regression - **at library defaults, with no tuning**
(tuning is D.4), on the same persisted folds, in both the `none` and
`downsample` scenarios. Ten runs, ~3 h wall clock on the 265-run pool. The
per-fold train cap the Random Forest needs is in `ablations_baselines.md` §7;
the run matrix and the pre-registered champion criterion are in its §8.

**Runner reproducibility.** D.3 required reworking how
`run_grouped_validation.py` loads data and predicts (float32/dictionary cast
at read time, the DataFrame released before the first `fit()`, and prediction
in bounded blocks). Both card-E decision-tree runs were re-executed on the
reworked runner as regression checks and `grouped_predictions.csv` came back
**byte-identical by SHA-256** in both scenarios over all 20,796,921 rows of
the then-current pool. The rework changed memory and wall clock only.

### Unbalanced (`none`): is it a decision-tree limit?

Pooled over all folds, original-distribution test partitions:

| Model | cap | accuracy (micro) | **macro F1** | weighted F1 | attack-class recall | ideal-`normal` attack_fpr |
|---|---:|---:|---:|---:|---|---:|
| decision-tree | - | 0.9844 | 0.2791 | 0.9792 | ~0.0000 (all four) | 0.00% |
| decision-tree | 4M | 0.9843 | 0.2789 | 0.9792 | ~0.0000 (all four) | - |
| xgboost | - | 0.9847 | 0.2823 | 0.9796 | ~0.0000 (all four) | 0.00% |
| logistic-regression | - | 0.9790 | 0.1649 | 0.9686 | **0.0000** (all four) | - |
| random-forest | 4M | 0.9827 | **0.3176** | 0.9790 | 0.0116-0.0565 | 0.20% |

**It is not a decision-tree artifact.** XGBoost at defaults, trained on the
full ~18M-row partition, predicts an attack class for essentially no row -
exactly like the tree. Logistic regression is worse still: it predicts
`normal` for *literally every row*, `benign_degradation` included (pooled
recall 0.0000037 there), which is what its macro F1 of 0.1649 measures. That
is not an optimisation failure - lbfgs converged in well under its 100
iterations in every fold - but a capacity limit, and it is the honest floor
this comparison needed.

**Random Forest is the one family that predicts attack rows**, and the
per-class detail matters more than the macro average:

| Class | pooled recall | pooled precision |
|---|---:|---:|
| `SAG.DB` | 0.0116 | 0.0362 |
| `FRG` | 0.0406 | 0.1543 |
| `SAG.PB` | 0.0357 | 0.1652 |
| `SAG.PBM` | 0.0565 | 0.2202 |

At under 6% recall this is **not a detector**. But it is the only operating
point in this revision where attack predictions carry non-trivial precision
(0.15-0.22 on three of the four classes) while the false-positive rate on
ideal `normal` traffic stays at **0.20%** - against 44.14% for the downsampled
tree, the only other configuration that detects anything. Fully grown,
unpruned trees carve out small genuine attack regions that a depth-8 tree and
a default-depth XGBoost never look for; they are just far too small to cover
the class.

**The cap is not doing this.** Paired over the same 265 runs, the decision
tree under the identical 4M cap differs from the uncapped tree by **-0.0002
macro F1, 95% CI [-0.0005, +0.0000] - an interval that does not exclude
zero.** The cap has no statistically detectable effect, so the Random Forest's
behaviour is a property of the family, not of §7's subsampling. (This controls
the cap's effect on a tree; it does not independently prove an uncapped
Random Forest would behave the same, which is why every Random Forest row
carries its cap.)

### Balanced (`downsample`): champion selection

The champion is chosen here, not above, because this is the scenario card
D.1's ablation runs in. Criterion fixed before execution: mean macro F1, then
mean attack-class recall, then ideal-`normal` attack_fpr.

| Model | **macro F1** (pooled) | mean attack recall | ideal-`normal` attack_fpr | `normal` recall |
|---|---:|---:|---:|---:|
| **xgboost** | **0.2099** | **0.7038** | **39.45%** | 0.5613 |
| decision-tree | 0.2076 | 0.6861 | 44.14% | 0.5211 |
| random-forest | 0.1874 | 0.6423 | 37.21% | 0.5630 |
| logistic-regression | 0.1351 | 0.5473 | 44.44% | 0.4420 |

**Champion: XGBoost** - but the honest statement is narrower than the ranking
suggests, and the paired bootstrap is what makes it sayable at all.

### The comparison the marginal intervals cannot make

Two models' confidence intervals overlapping does **not** mean they are
indistinguishable: both are evaluated on the same runs, so the run-to-run
variation they share cancels when the difference is taken draw by draw.
`bootstrap_run_intervals.py` resamples runs and scores every model on the same
draw (`run_bootstrap.downsample.md`, `run_bootstrap.none.md`):

| Scenario | A | B | B - A macro F1 | 95% CI | separates? |
|---|---|---|---:|---|---|
| `downsample` | decision-tree | **xgboost** | **+0.0023** | [+0.00002, +0.0045] | yes, barely |
| `downsample` | decision-tree | random-forest | -0.0202 | [-0.0233, -0.0165] | yes |
| `downsample` | decision-tree | logistic-regression | -0.0725 | [-0.0789, -0.0630] | yes |
| `none` | decision-tree | xgboost | +0.0032 | [+0.0011, +0.0060] | yes |
| `none` | decision-tree | **random-forest (4M)** | **+0.0385** | [+0.0329, +0.0442] | yes |
| `none` | decision-tree | logistic-regression | -0.1142 | [-0.1272, -0.0991] | yes |
| `none` | decision-tree | decision-tree (4M cap) | -0.0002 | [-0.0005, +0.0000] | **no** |
| `none` | decision-tree | smote | -0.0085 | [-0.0126, -0.0047] | yes |

Read that carefully. XGBoost beats the decision tree in `downsample`
**consistently but negligibly**: the difference is +0.0023 macro F1 and its
interval clears zero by 2e-5. It is a real ordering, not a coin flip, but it
is not a margin any claim should lean on - what actually separates XGBoost
from the tree at this operating point is the 4.7-percentage-point lower
false-positive rate on ideal traffic, not the macro F1. **The defensible
sentence is "the families are near-indistinguishable on macro F1 and XGBoost
was chosen for its lower alert burden", not "XGBoost is the better model".**

The same table also says the Random Forest's advantage in the `none` scenario
(+0.0385) is an order of magnitude larger than any difference among the other
families - it is the one genuinely distinct result in this card.

### How many attack rows are actually being counted

Rates hide the scale this evaluation operates at, so the same results in
counts. The pool holds **216,545 attack rows, 0.932% of 23,226,530**:

| Class | rows | % of pool | independent runs | test runs per fold (min-max) |
|---|---:|---:|---:|---:|
| `SAG.DB` | 50,782 | 0.219% | 45 | 3 - 16 |
| `FRG` | 63,976 | 0.275% | 45 | 6 - 12 |
| `SAG.PB` | 46,959 | 0.202% | 45 | 7 - 12 |
| `SAG.PBM` | 54,828 | 0.236% | 45 | 7 - 12 |
| `benign_degradation` | 270,680 | 1.165% | 85 | - |
| `normal` | 22,739,305 | 97.902% | 265 | - |

**1. The rows are learnable; the unbalanced models simply never look for
them.** The same 50,782 `SAG.DB` rows, the same folds and the same features:

| Run | `SAG.DB` found | of | recall |
|---|---:|---:|---:|
| xgboost, `none` | **1** | 50,782 | 0.0000 |
| xgboost, `downsample` | **45,826** | 50,782 | 0.9024 |

Nothing changed but the balance of the training partition, so "too few attack
samples to learn from" is not the explanation for the `none` results.

**2. The alert burden, in counts.**

| Run | attack alerts raised | of which real | precision |
|---|---:|---:|---:|
| random-forest, `none` (4M cap) | 57,301 | 7,957 | **13.9%** |
| xgboost, `downsample` | 9,769,498 | 149,187 | **1.5%** |

The champion configuration raises **9.8 million** attack alerts across the
pool to find 149,187 real attack rows. That is the "39% false-positive rate"
above expressed the way an operator would meet it, and it is why the Random
Forest's low-recall/13.9%-precision corner is worth more attention than its
macro F1 suggests.

**3. The unit-level weakness this pool was extended to fix.** `split_group`
(one ERENO run) is the experimental unit, so a class's effective sample size
is its **run** count, not its row count. `SAG.DB` and `FRG` originally had 15
runs each and a fold's test partition could hold a *single* one, making that
fold's recall a measurement of ~1,000 correlated messages from one run. The
2026-09-13 extension took both to 45 runs, and the run-level bootstrap shows
exactly the intended effect - with `SAG.PB`/`SAG.PBM` (unchanged at 45 runs)
as the control:

| Class | runs before -> after | recall 95% CI width before -> after | change |
|---|---|---|---:|
| `SAG.DB` | 15 -> 45 | 0.1184 -> 0.0690 | **-42%** |
| `FRG` | 15 -> 45 | 0.1820 -> 0.1279 | **-30%** |
| `SAG.PB` | 45 -> 45 | 0.0853 -> 0.0841 | -1% |
| `SAG.PBM` | 45 -> 45 | 0.0601 -> 0.0608 | +1% |

Only the classes that gained runs narrowed; the two that did not are flat.
That is the control which says the narrowing came from the added runs rather
than from anything else the regeneration changed.

### What D.3 settles, and what it does not

- Card E's class-imbalance explanation **survives the family comparison**. No
  model family at library defaults escapes it on the unbalanced pool, so the
  zero-detection result is not an artifact of the specific tree that produced
  it.
- The `downsample` trade-off is likewise **family-independent**: all four
  models land at macro F1 0.14-0.21 with `normal` recall ~0.44-0.56 and
  attack_fpr 37-44%. Changing the model does not buy a usable operating point;
  the balancing method dominates.
- **Model capacity, not model family, is the axis that moves attack
  detection.** The only configuration that predicts attack rows with
  non-trivial precision at a tolerable false-positive rate is the one with
  fully grown, unpruned trees. That makes depth and estimator count the first
  thing D.4's grid should spend on.
- Detectability itself is still **not** established. The best attack-class
  numbers here are either under 6% recall (Random Forest, `none`) or bought at
  a ~39% false-positive rate (XGBoost, `downsample`).
- Cost note for D.1: the champion's `downsample` run takes **2.7 min**, so the
  six-run ablation is ~20 min of compute rather than the 4-6 h
  `ablations_baselines.md` §4 budgeted against a possibly-expensive champion.


## The threshold axis (checklist D.5, 2026-09-13)

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
