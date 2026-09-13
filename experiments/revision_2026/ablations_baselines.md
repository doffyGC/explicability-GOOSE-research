# Ablations and baselines (Card D)

Checklist ref.: D.1–D.5 in `prioridades_revisao_gray_goose.pdf`, P1 "Ablation
study por grupos de features" + "Baselines fortes". See `README.md` for how
this fits the rest of the revision, `validation_protocol.md` for the grouped
workflow every run here must reuse, and its "Balancing scenarios" section for
the card-E results this card is the direct follow-up to.

**Status: D.3 closed (2026-09-13, re-run on the extended 265-run pool).**
§7 (the per-fold train subsampling policy), §8 (the run matrix) and §9 (the
result) are done; D.1, D.2, D.4 and D.5 are still plan only. §6 is the
authoritative tracker.

## 1. Why this exists

The first full grouped run (`validation_protocol.md`, "Full-scale results")
produced a model that **never predicts any attack class**, and card E showed
that the outcome flips entirely depending on how training balance is handled:
capped SMOTE changes nothing, full downsampling recovers attack recall at a
43–44% attack false-positive rate on ideal `normal` traffic. Both ends are
unusable operating points.

That leaves the paper's central question unanswered: is SAG undetectable under
leakage-free grouped validation, or is this specifically a
`DecisionTreeClassifier(max_depth=8)` limit? Card D is what separates the two —
by varying the feature groups, the model family and the hyperparameters, each
one at a time, on the same persisted folds.

## 2. Scope decision (2026-09-12)

Full card D was estimated at ~15–20 h of implementation plus ~25–40 h of
serial compute (≈6–8 calendar days on the current 15.6 GB / i7-1255U machine).
**We chose the minimum defensible scope: ~11–14 h implementation + ~12–18 h
compute, ≈3–4 calendar days.** Scope may be widened later if the revision
calendar allows — the deferred items in §3 are written down precisely so that
widening is a resumption, not a redesign.

### In scope now

| # | Item | Scope taken |
|---|---|---|
| D.1 | Feature-group ablation | **6 of 7 groups** (raw GOOSE/SV, delta features, electrical, sequence, no absolute time, no `StNum`/`SqNum`), run **on the champion model only**, in the `downsample` balancing scenario, with `none` kept as the reference point from the existing full run |
| D.2 | Rule-based baseline | Full: `sqNum`/`stNum` gap detector + delay threshold, thresholds calibrated **only on each fold's train partition**, emitting the same prediction format the existing audits consume |
| D.3 | Model comparison | XGBoost, Random Forest and one simple classifier (logistic regression), all on the persisted folds. **No temporal model.** |
| D.4 | Tuning inside train folds | Inner grouped split over the outer fold's train groups, **small grid on a documented subsample** rather than an exhaustive search on all 16.6M train rows |
| D.5 | Per-class reporting | Cross-run comparison table (ablation × model, per class + macro, with confusion matrices), reusing `check_prediction_integrity.py` and `benign_confusion_report.py` |

### Deferred, with the reason

| Deferred item | Why | Where it goes |
|---|---|---|
| **Top-k SHAP ablation** (the 7th feature group in D.1) | Requires SHAP importances computed on held-out grouped data, which is card F's deliverable. Running it now would mean either importances from a train fold (methodologically wrong, and exactly the baseline flaw the revision exists to fix) or pulling card F forward. | **Card F.** When held-out SHAP lands, add the top-k group as a 7th ablation run and update §4 here. |
| **Simple temporal model** (D.3's "se viável") | The checklist itself marks it conditional. It needs windowed features computed inside each trace without crossing boundaries — new preparation work, not just a new `--model` value; +4–6 h implementation before a single run. | Revisit after D.1–D.5 close, if the calendar allows. |
| **Ablation × every model family** | 6 feature groups × 3 model families ≈ 15 h of extra serial compute for a table nobody asked for. The checklist asks for an ablation, not an ablation per model. | Only if a second model family turns out to be competitive enough that its feature dependence is itself a finding. |
| **Ablation in both balancing scenarios** | `none` detects zero attack rows, so ablating it compares variants that are all equally blind. The signal to ablate only exists in `downsample`. | The existing `none` full run stays as the published reference row. |

## 3. Execution order

Ordering is a compute decision, not a preference: picking the champion **before**
the ablation is what keeps the ablation at ~6 runs instead of ~18.

1. **D.3** — model families, no tuning, default hyperparameters → pick champion.
2. **D.1** — 6 feature groups on the champion only.
3. **D.2** — rule-based baseline (cheap, independent; can run at any point).
4. **D.4** — nested tuning on the champion family, subsampled grid.
5. **D.5** — consolidate every run into the comparison tables.

## 4. Cost estimate per item

Calibration (as of the 2026-09-12 estimate, on the then-current 205-run /
20,796,921-row pool): the three existing full runs (decision tree, 5 folds)
each took **~20–50 min wall clock** on this machine. Actual costs on the
265-run pool are in §9/§10; the decision tree came in at 19 min, XGBoost at
37 min and the capped Random Forest at 56 min. Runs do not
parallelise here — each needs the whole prepared dataset resident — so compute
is serial.

| Item | Implementation | Compute | Main risk |
|---|---|---|---|
| D.3 | ~2 h (`random-forest`, `logistic-regression` + scaler pipeline in `--model`) | 10–20 h (XGB 2–5 h/run, RF 2–4 h/run, LR ~1 h/run) | **RAM.** The plain tree already came close to the 15.6 GB ceiling; RF will likely need a documented per-fold subsampling policy (+1–2 h) |
| D.1 | ~2–3 h (feature-set registry + flag + tests + docs) | 4–6 h (6 runs × 30–45 min, champion model) | If the champion is XGB/RF rather than the tree, multiply by that model's per-run cost |
| D.2 | ~3–4 h (new script + tests) | ~30 min | Low |
| D.4 | ~3–4 h (inner grouped split + grid) | 4–8 h subsampled | This is the multiplier that blows the schedule if run un-subsampled |
| D.5 | ~3–4 h (cross-run comparison + docs) | ~0 | Low |

## 5. Invariants every run in this card must respect

Non-negotiable, inherited from cards A/B/E — a card-D run that breaks one of
these is not a result:

- Runs consume the **persisted** `splits_grouped.json`; no script generates
  folds internally (`validation_protocol.md` §3).
- The dataset SHA-256 must match the preparation and split artifacts.
- Balancing and tuning touch **only the train partition** of each fold; every
  evaluation is on the untouched original distribution.
- Every reported metric names its averaging scheme; per-class values first,
  macro as the headline (`validation_protocol.md`, E.4).
- Finished runs go through `check_prediction_integrity.py` before any number
  from them is written into the paper.
- New scripts get a row in `README.md` in the same commit.

## 6. Status tracker

| Item | Status | Evidence |
|---|---|---|
| D.1 feature-group ablation (6/7) | not started — **next**, on XGBoost | — |
| D.1 top-k SHAP group | deferred to card F | — |
| D.2 rule-based baseline | not started | — |
| D.3 model comparison (XGB/RF/LR) | **done** — champion: **XGBoost** | §9; `validation_protocol.md`, "Model family comparison"; `results/d3-*` (9 runs); `prediction_integrity_d3.md` (315 checks, 0 failures) |
| D.3 temporal model | deferred | — |
| D.4 nested tuning | not started | — |
| D.5 per-class cross-run report | not started | — |
| D.5 threshold axis (AP + alert budgets) | **done for the champion, both scenarios** — see §11 | `grouped_pr_curves.py`; `validation_protocol.md`, "The threshold axis"; `pr_curves.md`; `results/d5-xgboost-{none,downsample}`; `prediction_integrity_d5.md` |

## 7. Per-fold train subsampling policy (D.3)

Agreed 2026-09-12, **before** any heavy run was launched. It exists because
one model family — Random Forest at library defaults — cannot fit a full
train partition on this machine, and because guessing at that limit instead
of measuring it is how the schedule in §4 gets blown.

### What was measured

Fold-01 of `splits_grouped.json`, 40 features, 6 classes, default
hyperparameters, on this machine (15.6 GB RAM, i7-1255U):

| Model | n = 500k | n = 2.28M | extrapolated to a full ~16M-row train partition |
|---|---:|---:|---:|
| Random Forest | 57 s, 2.19M nodes | 388 s, 9.12M nodes | **~50–55 min/fold, ~6.7 GB of fitted forest** |
| XGBoost | 15 s | 86 s | ~10 min/fold, negligible RAM |
| Logistic regression | 8 s | 27 s | ~3 min/fold, negligible RAM |

The Random Forest cost is structural and close to linear: **4.0–4.4 tree
nodes per training row** across its 100 trees, at ~112 bytes a node
(scikit-learn's node struct plus its per-class value array) ≈ **490 bytes of
fitted forest per training row**. The prediction is not theoretical — at
2.28M rows it predicted 0.95 GB against 0.94 GB of measured RSS growth.
A full train partition therefore needs ~6.7 GB of forest *on top of* the
resident feature matrix, which does not fit. XGBoost and logistic regression
need no cap at all.

### The policy

1. **The cap applies to Random Forest in the `none` scenario only.** XGBoost
   and logistic regression train on the full partition; so does the decision
   tree, which keeps the card-E baseline directly comparable.
2. **Cap = 4,000,000 train rows per fold** (~25% of a typical partition) →
   ~1.7 GB of forest, ~13 min/fold. Drawn **proportionally stratified by
   (`split_group` × `class`)**, seeded `--seed + fold_index`. Every stratum
   keeps its share, so the subsample is a *shrunken replica* of the fold's
   train distribution, **not** a rebalancing — `--balance` remains the only
   thing that deliberately changes class proportions, and it runs after this.
   A stratum that would round down to zero rows keeps one, so a cap can never
   silently delete a rare class from training.
3. **The test partition is never subsampled.** Every metric is still measured
   on the untouched original distribution, so the run stays a
   `full_grouped_run` rather than a `technical_smoke`.
4. **A cap control run is mandatory**: the same decision tree, same scenario,
   same cap. Without it, a Random-Forest-vs-tree difference cannot be
   attributed to the model family rather than to the cap.
5. **The `downsample` scenario takes no cap** — after balancing, a fold's
   train partition is 75k–96k rows.
6. The cap is recorded per run (`max_train_rows_per_fold`) and per fold
   (`subsample.applied`, `train_rows_available`, `train_rows_sampled`) in
   `grouped_validation_report.json`. **D.5 must never put a capped and an
   uncapped run in the same column without labelling it.**

### Memory prerequisite

> **Superseded in part (2026-09-13).** The figures below describe the
> DataFrame-based loader on the 205-run pool. On the 265-run pool that path
> peaked at **10.02 GB** and no longer fit on this machine at all - every full
> run died during the load, whatever model followed. `load_grouped_arrays`
> now fills the feature array straight from the Parquet row groups, taking the
> peak to **4.03 GB** against 3.46 GB of unavoidable feature matrix. The cap
> in this section is therefore no longer forced by the *load*; whether a
> Random Forest still needs it for its fitted trees has not been re-measured,
> so every Random Forest run keeps carrying its cap until it is.


Making the cap enough required cutting the resident footprint as well:
`run_grouped_validation.py` now reads the Parquet row group by row group,
casting features to `float32` and the group/target columns to dictionary
(Categorical) encoding on the way in, and releases the DataFrame before the
first `fit()` (`prepare_arrays`). Peak RSS during the load drops from
**11.7 GB to 7.3 GB** and the resident frame from ~8.7 GB to ~3.5 GB. Values
are unchanged — `feature_matrix` already applied exactly that `float32`
rounding — which is why the card-E baseline is re-run as a regression check
in §8 rather than assumed to still hold.

### What is *not* being done, and why

`RandomForestClassifier(max_samples=...)` would give the same node budget
while letting each tree draw its own subsample from the whole partition, so
the forest would collectively see every row — scientifically richer. It was
rejected for D.3 because it changes the model away from library defaults
(the one thing this untuned family comparison is supposed to hold fixed) and
because it cannot be mirrored on the decision tree, so the cap control in
point 4 would no longer be possible. Revisit under D.4, where deliberately
moving hyperparameters is the point.

## 8. D.3 run matrix

Eight runs, serial, all consuming the persisted `splits_grouped.json`
(dataset SHA-256 bound, leakage audit already passed). Champion criterion,
fixed before execution: **mean macro-F1 across the 5 folds in the
`downsample` scenario** — the scenario D.1's ablation runs in — with
per-class attack recall and then ideal-`normal` `attack_fpr` as tie-breakers.

| # | Run directory (`results/`) | Model | Balance | Cap | Purpose |
|---|---|---|---|---|---|
| 1 | `d3-decision-tree-none` | decision-tree | none | — | Regression check: must reproduce the card-E baseline fold for fold on the refactored runner |
| 2 | `d3-xgboost-downsample` | xgboost | downsample | — | Champion candidate |
| 3 | `d3-random-forest-downsample` | random-forest | downsample | — | Champion candidate |
| 4 | `d3-logistic-regression-downsample` | logistic-regression | downsample | — | Simple-classifier floor |
| 5 | `d3-logistic-regression-none` | logistic-regression | none | — | Family-limit question |
| 6 | `d3-xgboost-none` | xgboost | none | — | Family-limit question |
| 7 | `d3-random-forest-none-cap4m` | random-forest | none | 4M | Family-limit question, under §7's cap |
| 8 | `d3-decision-tree-none-cap4m` | decision-tree | none | 4M | Cap control for run 7 |

Runs 2–4 answer *which family to carry into D.1*; runs 5–8 answer the
question card E left open — whether never predicting an attack class is a
`DecisionTreeClassifier(max_depth=8)` limit or a limit of every family at
this class balance.

## 9. D.3 result (2026-09-13, on the 265-run pool)

Ten runs, ~3 h wall clock on the extended pool (23,226,530 rows). Full tables
are in `validation_protocol.md`, "Model family comparison (checklist D.3)";
this section records what the card carries forward.

> The card was first completed on the 205-run pool on 2026-09-12 and re-run in
> full after that pool was extended (see §10). The conclusions held; the
> numbers moved.

### Champion: XGBoost — with a narrower claim than the ranking suggests

| Model (`downsample`) | macro F1 | mean attack recall | ideal-`normal` attack_fpr |
|---|---:|---:|---:|
| **xgboost** | **0.2099** | **0.7038** | **39.45%** |
| decision-tree | 0.2076 | 0.6861 | 44.14% |
| random-forest | 0.1874 | 0.6423 | 37.21% |
| logistic-regression | 0.1351 | 0.5473 | 44.44% |

The paired bootstrap (same runs resampled for both models) puts XGBoost above
the decision tree by **+0.0023 macro F1, 95% CI [+0.00002, +0.0045]** — a real
ordering whose interval clears zero by 2e-5. That is not a margin to lean on.
**The defensible sentence is "the two are near-indistinguishable on macro F1
and XGBoost was chosen for its 4.7-point lower false-positive rate on ideal
traffic", not "XGBoost is the better model."**

### The question card E left open, answered

Zero attack recall on the unbalanced pool is **not** a
`DecisionTreeClassifier(max_depth=8)` limit: XGBoost at defaults also predicts
an attack class for essentially no row (1 row out of 216,545), and logistic
regression predicts `normal` for literally every row (macro F1 0.1649).

Random Forest is the only family that predicts attack rows at all, and it is
the one genuinely distinct result in this card: **+0.0385 macro F1 over the
tree, 95% CI [+0.0329, +0.0442]** — an order of magnitude larger than any
difference among the other families. At under 6% recall it is no detector, but
it raises 57,301 attack alerts of which **13.9% are real**, against 1.5% for
the champion's downsampled operating point. The decision-tree control under
the identical 4M cap differs from the uncapped tree by -0.0002 [-0.0005,
+0.0000] — **does not separate**, so §7's cap is not producing this.

### Consequences for the rest of card D

- **D.1 gets much cheaper than §4 budgeted.** The champion's `downsample` run
  takes 2.7 min, so six ablation runs are ~20 min of compute, not 4–6 h. This
  does **not** license widening the ablation scope — the deferrals in §3 were
  argued on methodology, not compute.
- **D.4 has a specific target.** What separates "predicts nothing" from
  "predicts something precisely but rarely" is model *capacity*, not family:
  the only configuration that finds attack rows at usable precision is the one
  with fully grown, unpruned trees. Tree depth and estimator count are where
  D.4's subsampled grid should go first.
- **Report per-class, and pair.** Marginal intervals for these models overlap
  almost everywhere; only the paired comparison separates them. Every D.1/D.4
  comparison should use `bootstrap_run_intervals.py`'s paired table rather
  than eyeballing overlapping error bars.

## 10. Pool extension (2026-09-13)

Card D.3's first pass exposed a weakness that was not about models at all.
`split_group` is the experimental unit, so a class's effective sample size is
its **run** count: `SAG.DB` and `FRG` had 15 runs each, and a fold's test
partition could hold a *single* run — making that fold's recall a measurement
of one run's ~1,000 correlated messages. The per-fold spread confirmed it: the
1.0000 `SAG.DB` recall came from the fold holding one run, the worst from the
fold holding two.

Ten extra seeds per variant (3 cells each, +60 runs) took both classes to 45
runs, matching `SAG.PB`/`SAG.PBM`. Cost: ~36 min of ERENO generation, ~9 min
to rebuild the chain, ~3 h to re-run every model. Effect, with the untouched
classes as the control:

| Class | runs | recall 95% CI width before → after |
|---|---|---|
| `SAG.DB` | 15 → 45 | 0.1184 → 0.0690 (**−42%**) |
| `FRG` | 15 → 45 | 0.1820 → 0.1279 (**−30%**) |
| `SAG.PB` | 45 → 45 | 0.0853 → 0.0841 (−1%) |
| `SAG.PBM` | 45 → 45 | 0.0601 → 0.0608 (+1%) |

Only the classes that gained runs narrowed. Two consequences to carry:

- **Before/after numbers are not comparable row for row.** Folds are redrawn
  over 265 groups and attack prevalence moved 0.675% → 0.932%. The 205-run run
  reports are kept in `archive_205runs/` for provenance, not for mixing into
  new tables.
- **The 0.932% prevalence is still a configured quantity**, set by ERENO's
  ~1,000-malicious-messages-per-run target. It dominates every result in cards
  D and E and remains an open decision for the paper (`data_card.md` §4).

## 11. The threshold axis, and what it changes for D.1/D.4 (2026-09-13)

Card D.3 compared model families at `argmax(p)`. That comparison is sound but
narrow: the argmax is one point on a curve, fixed by the training partition's
class prior rather than chosen, which is why `none` and `downsample` read as
opposite verdicts on the same model (`validation_protocol.md`, "The threshold
axis", has the 212.6x arithmetic). `run_grouped_validation.py --save-scores`
now persists the per-row posteriors and `grouped_pr_curves.py` reports
average precision plus operating points at a fixed alert budget, with
thresholds calibrated on the folds that are not being scored.

### What the champion's curves say

Both scenarios were re-run with `--save-scores` and compared
(`pr_curves.md`):

- The signal is real: AP clears its prevalence floor by **8-28x** on all four
  attack classes, intervals included.
- **Downsampling makes the ranking worse.** Paired over the same runs,
  `downsample` - `none` is -0.0100 AP on `ANY_ATTACK`, 95% CI [-0.0135,
  -0.0069], and separates on three of the four classes. Card E's rebalancing
  moved a threshold and paid for it with the ~18.6M training rows the
  downsampled fits threw away.
- The argmax was already close to its own frontier. At a **matched** alert
  rate, recalibration buys +0.02 (`SAG.DB`, `SAG.PB`, `ANY_ATTACK`) to +0.18
  (`SAG.PBM`) recall - real, worth reporting, and nowhere near enough to
  rescue the operating point.
- Running quietly is what fails. Dropping `SAG.DB` to a 0.1% alert rate takes
  recall to 0.051.
- Per-class scores rank better than the pooled attack score (0.995/0.674/
  0.663/0.698 against `ANY_ATTACK`'s 0.563 at the same budget), so the four
  detectors should stay separate.

### Three consequences for the rest of card D

1. **D.1 and D.4 must be judged on AP and on budgeted recall, not on argmax
   macro F1.** A feature ablation or a depth sweep that moves the score
   distribution without moving the ranking will look like a large macro-F1
   change and be worth nothing, and the reverse is equally possible. Every
   D.1/D.4 run therefore gets `--save-scores`, and its comparison table gets
   an AP column. This costs ~0.4 GB and ~1 min of curve computation per run.
2. **D.4's target sharpens.** D.3 pointed at capacity (unpruned trees) because
   the Random Forest was the only family predicting attack rows at the
   argmax - but "predicts attack rows at the argmax" is a statement about
   where its posteriors sit, not about how well they rank. Whether the
   Random Forest's advantage survives on AP is now a question that can be
   asked directly, and it should be asked **before** the grid is designed.
3. **The card-C confound has to be re-examined at a calibrated threshold**,
   not concluded from the argmax. `grouped_pr_curves.py` splits false alarms
   by true class only; the per-`impairment_mode` rejoin
   `benign_confusion_report.py` owns has not been run against thresholded
   predictions, so nothing about the benign confound may be updated yet.
