# Ablations and baselines (Card D)

Checklist ref.: D.1–D.5 in `prioridades_revisao_gray_goose.pdf`, P1 "Ablation
study por grupos de features" + "Baselines fortes". See `README.md` for how
this fits the rest of the revision, `validation_protocol.md` for the grouped
workflow every run here must reuse, and its "Balancing scenarios" section for
the card-E results this card is the direct follow-up to.

**Status: D.3 closed on the corrected pool (2026-09-16).** The defect that
blocked this card - 100% of attack rows having a content-identical `normal`
row in the same run, because ERENO wrote both the legitimate stream and the
grayhole's forwarded copies - was fixed by the regeneration in `d2de01c`. The
whole D.3 matrix was then re-run on the corrected pool (265 runs,
11,057,478 rows, SHA-256 `3109e4d4…`).

**Every number in this card now comes from the corrected pool.** The
defective-pool figures it replaces are in this file's git history; §12 records
what changed and which of this card's earlier conclusions did not survive.
D.1, D.2 and D.4 are **unblocked** and still plan only. §6 is the
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
| D.1 | Feature-group ablation | **6 of 7 groups** (raw GOOSE/SV, delta features, electrical, sequence, no absolute time, no `StNum`/`SqNum`), run **on the champion model only**, in the **`none`** balancing scenario (the scenario choice was inverted on 2026-09-16 — see §12), with `downsample` kept as the reference point from the existing full run |
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
| **Ablation in both balancing scenarios** | One scenario has to carry the ablation, and on the corrected pool that scenario is `none`: it is the better ranker on every attack class (§9) and it is the configuration the paper reports. Ablating both would double the runs to answer a question nobody asked. (Until 2026-09-16 this deferral read the other way round — on the defective pool `none` detected zero attack rows, so `downsample` was the only scenario with a signal to ablate. §12 has the inversion.) | The existing `downsample` full run stays as the published reference row. |

## 3. Execution order

Ordering is a compute decision, not a preference: picking the champion **before**
the ablation is what keeps the ablation at ~6 runs instead of ~18.

1. ~~**D.3** — model families, no tuning, default hyperparameters → pick champion.~~ **Done (§9): XGBoost.**
2. ~~**D.1** — 6 feature groups on the champion only.~~ **Done (§14): the deltas carry it.**
3. ~~**D.2** — rule-based baseline.~~ **Done (§16): the model earns its complexity.** §15 aimed it at the interval rather than the `sqNum` gap, and that aim was right — though `stDiff` turned out to be an equally good single threshold.
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
| D.1 | **done** (registry + flag + tests + docs, 2026-09-16) | **done: 136 min measured** (18-26 min per run, not the 37 projected) + 5.8 GB | Resolved: peak RSS 4.8-5.4 GB, flat, no training spike (§14) |
| D.2 | **done** (~3.5 h: script + 31 tests + docs, 2026-09-18) | **done: 14 min measured** (2 min 20 s per rule - one column, not forty) + 5.4 GB | Resolved |
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
| D.1 feature-group ablation (6/7) | **done** — the nine delta features carry essentially all of it (-0.7227 AP paired); 29 of 40 columns are free | §14; `results/d1-xgboost-*` (6 runs); `pr_curves_d1.md`; `prediction_integrity_d1.md` (273 checks, 0 failures) |
| D.1 delta split (unpreregistered follow-up) | **done** (2026-09-17) — inside the deltas the carrier is **timing** (-0.0645 AP paired, and the only ablation besides `no-delta` that moves the 1% operating point); the four size/state deltas are free | §15; `results/d1-xgboost-no-{timing,size-state}-deltas`; `pr_curves_d1_split.md`; `prediction_integrity_d1_split.md` (195 checks, 0 failures) |
| D.1 top-k SHAP group | deferred to card F | — |
| D.2 rule-based baseline | **done** (2026-09-18) — the champion beats the best single threshold by **-0.5918 AP paired** [-0.6385, -0.5392]; the baseline is real (12.3x chance), not a straw man | §16; `run_rule_baseline.py`; `results/d2-rule-*` (6 runs); `pr_curves_d2.md`; `pr_curves_d2_rules.md`; `prediction_integrity_d2.md` (273 checks, 0 failures) |
| D.3 model comparison (XGB/RF/LR) | **done on the corrected pool** — champion: **XGBoost** | §9; `validation_protocol.md`, "Model family comparison"; `results/v2-*` (9 runs); `prediction_integrity_d3_v2.md` (351 checks, 0 failures — §9, "The argmax disagreement on Random Forest"); `run_bootstrap.{none,downsample,champion}_v2.md`; `pr_curves_d3_v2.md` |
| D.3 temporal model | deferred | — |
| D.4 nested tuning | **unblocked, not started** | §12 |
| D.5 per-class cross-run report | not started | — |
| D.5 threshold axis (AP + alert budgets) | **done for all nine D.3 runs** on the corrected pool — see §11 | `grouped_pr_curves.py`; `validation_protocol.md`, "The threshold axis"; `pr_curves_d3_v2.md`; `prediction_integrity_d3_v2.md` |

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

> **Superseded again (2026-09-16).** §14 has the *measured* footprint of six
> full runs: peak RSS 4.8-5.4 GB, **flat** from the end of the load onward.
> There is no training spike on top of the load at all, which is the
> assumption both notes below reason from. Read this section as history.

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

## 9. D.3 result (2026-09-16, on the corrected pool)

Nine runs, ~5 h wall clock on the corrected 265-run / 11,057,478-row pool.
Full tables are in `validation_protocol.md`, "Model family comparison
(checklist D.3)"; this section records what the card carries forward.

> The card was completed twice before this: on the 205-run pool (2026-09-12)
> and on the extended 265-run pool (2026-09-13, §10). Both measured the
> duplicate-stream defect. **Those conclusions are withdrawn, not adjusted** -
> §12 lists which ones and why.

### Champion: XGBoost — chosen on ranking, not on accuracy

| Model (`none`) | macro F1 [95% CI] | mean attack recall | ideal-`normal` attack_fpr | AP `ANY_ATTACK` |
|---|---|---:|---:|---:|
| **xgboost** | 0.7310 [0.7141, 0.7451] | 0.6338 | **0.01%** | **0.8329** |
| random-forest (4M cap) | 0.7355 [0.7183, 0.7494] | 0.6408 | 0.12% | 0.7934 |
| decision-tree | 0.7043 [0.6876, 0.7174] | 0.5893 | 0.02% | 0.7572 |
| logistic-regression | 0.3677 [0.3486, 0.3879] | 0.2248 | 0.08% | 0.5087 |

Paired over the same 265 runs, XGBoost and the Random Forest **do not
separate** on macro F1 (+0.0045, 95% CI [-0.0036, +0.0117]) - so the argmax
does not choose between them. Average precision does: the forest is -0.0394 AP
on `ANY_ATTACK` [-0.0471, -0.0321] and loses on three of the four classes
individually. **XGBoost is champion because it ranks strictly better, needs no
train cap and costs 37 min against 56.**

### What the correction answered, and what it left open

- The question card E left - "is zero attack recall a `max_depth=8` limit or
  every family's?" - **had a false premise**. On the corrected pool the plain
  unbalanced tree reaches 0.66-0.84 recall on three of four classes at 0.02%
  ideal-traffic FPR. Nothing about imbalance was being measured.
- **`SAG.PBM` resists every family** (0.18-0.40 recall, AP 0.41 at best).
  `label_duplication_audit.md` §7 explains it: most of its discards happen at a
  state boundary, where a single message carries no information about whether a
  discard preceded it. No per-message model can fix that.
- **`FRG` ≡ benign congestion loss, by construction.** 15 of its 45 runs are
  byte-identical in payload to `BENIGN_CONGESTION_LOSS` controls; the champion
  calls 73.33% of `CONGESTION_LOSS` rows an attack while its ideal-traffic FPR
  is 0.01%. Decision (2026-09-16): keep the class, report the null - see
  `benign_controls.md` §8.

### Consequences for the rest of card D

- **D.1 ablates in `none`, not `downsample`** (§12), at ~37 min per run, so six
  ablation runs are ~3.7 h of serial compute.
- **D.1 and D.4 are judged on AP and budgeted recall**, not on argmax macro F1
  (§11). Every run gets `--save-scores`.
- **D.4's target is no longer "capacity".** That conclusion came from the
  Random Forest being the only family to predict attack rows on the defective
  pool. It now predicts them like everyone else and ranks worse, so the grid
  should be designed from the champion's own error structure - specifically
  `SAG.PBM` and the `benign_degradation` boundary - rather than from a
  depth-and-estimators hunch.
- **The two red integrity checks are closed** — see below. Random Forest
  figures are citable.

### The argmax disagreement on Random Forest, resolved (2026-09-17)

`prediction_integrity_d3_v2.md` reported 2 failures of 351 checks, both on
Random Forest: `argmax(posterior)` disagreeing with `y_pred` on 7 rows of
11,057,478 (`none-cap4m`) and 17 (`downsample`). The documented hypothesis was
a float tie in `predict`. It is a tie — and the audit, not the runs, is where
it had to be handled.

Both runs hit the runner's documented fallback (`fell_back_to_predict`) in 9 of
their 10 folds, which makes `y_pred` an argmax over sklearn's float64
posteriors while the persisted scores are float32. **All 24 rows are exact
float32 ties** (top-1 and top-2 bit-identical), and on all 24 the audit picked
the lower class index where `predict` had kept the higher one — the only
direction that can disagree. The runs carry 1,224 and 16,420 float32-tied rows
respectively; the 24 are those whose float64 values were not tied as well.
`v2-xgboost-none` has 0 tied rows, `v2-decision-tree-none` 31, neither with a
disagreement.

`check_prediction_integrity.py` now forgives a mismatch only on an exact top-1
tie in a fold the report marks `fell_back_to_predict`; a strict winner that
disagrees still fails, as does a tie outside a fallback fold. The audit is
**351 checks, 0 failures**, with every other number in it unchanged. Full
account in `validation_protocol.md`, "The Random Forest argmax disagreement".

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

> **Recomputed on the corrected pool (2026-09-16).** The numbers below are
> defective-pool; `pr_curves_d3_v2.md` has the current curves for all nine D.3
> runs and §9 the champion's. Two of this section's three consequences survive
> unchanged - judge D.1/D.4 on AP and budgeted recall, and pair every
> comparison. The third (D.4's target) is revised in §12. The claim that
> downsampling *hurts* the ranking survives on both pools: -0.0121 AP on
> `ANY_ATTACK`, 95% CI [-0.0172, -0.0071].


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

## 12. What the corrected pool changed (2026-09-16)

The regeneration in `d2de01c` removed the legitimate publisher's stream that
ERENO was writing alongside the grayhole's forwarded copies. The pool went from
23,226,530 rows to 11,057,478 with essentially the same 216.5k attack rows, so
attack prevalence doubled to 1.958% - but the real change is that an attack row
no longer has a bit-identical `normal` twin in its own run.

### Withdrawn

| Conclusion (defective pool) | Status |
|---|---|
| "No model family at library defaults escapes the class imbalance" | **Withdrawn.** Every tree-based family detects three of four classes unbalanced. It was the twin rows, not the imbalance. |
| "Model capacity, not family, is the axis that moves attack detection" | **Withdrawn.** It rested on the Random Forest being the only family to predict attack rows; it no longer is, and it ranks worst of the three tree families. |
| "The signal to ablate only exists in `downsample`" | **Inverted.** `none` is now both the better ranker and the configuration to report. |
| "Random Forest's low-recall/13.9%-precision corner deserves attention" | **Withdrawn.** That corner was an artifact of the only family that could see past the twins. |

### Survived

- The **protocol**: grouped splitting, the hash binding, the independent
  leakage audit, the run-level bootstrap, cross-fold threshold calibration.
- **Pairing every comparison.** Marginal intervals overlap on the corrected
  pool exactly as they did before; the paired table is still the only thing
  that separates two models.
- **Downsampling hurts the ranking** (§11).
- **The 4M cap has no detectable effect on the decision tree** (-0.0010 macro
  F1, 95% CI [-0.0029, +0.0011]).
- **Naming the averaging scheme.** The champion scores weighted F1 0.9793 and
  macro F1 0.7310 over identical predictions.

### Still open, and unchanged by the correction

The label is still **per-message**, and `label_duplication_audit.md` §7
measures that a meaningful share of attack rows are unidentifiable in principle
from a single message - most discards happen at a state boundary, where the
first `SqNum` does not separate attacked from unattacked states. Every recall
number in this card is bounded by that. The window redesign is the open scope
decision, and `SAG.PBM` is where the cost of not taking it is visible.

## 13. D.1: the feature groups and the run matrix (implemented 2026-09-16)

The registry is `FEATURE_GROUPS` in `run_grouped_validation.py` and the flag is
`--feature-set`. **The partition is the experiment**: every run removes exactly
one group, so whatever moves against the reference is attributable to that
group and to nothing else. `no-sequence` is written as a composition of two
disjoint groups rather than as a group of its own, because overlapping groups
would make two ablation runs mutually unreadable.

### The groups

The eight entries below partition all 40 model features of the prepared
dataset. Verified against the current pool (SHA `3109e4d4…`): no group names a
column the dataset lacks, and `delay` is the only feature no group claims.

| Group | n | Columns |
|---|---:|---|
| `electrical` | 18 | `isb{A,B,C}`, `vsb{A,B,C}`, the six `*RmsValue`, the six `*TrapAreaSum` — the Sampled Values process data |
| `goose-header` | 7 | `cbStatus`, `frameLen`, `gooseTimeAllowedtoLive`, `gooseLen`, `confRev`, `numDatSetEntries`, `APDUSize` |
| `absolute-time` | 3 | `Time`, `t`, `GooseTimestamp` |
| `counters` | 2 | `StNum`, `SqNum` |
| `counter-deltas` | 2 | `stDiff`, `sqDiff` |
| `timing-deltas` | 3 | `timestampDiff`, `tDiff`, `timeFromLastChange` |
| `size-state-deltas` | 4 | `gooseLengthDiff`, `cbStatusDiff`, `apduSizeDiff`, `frameLengthDiff` |
| *(ungrouped)* | 1 | `delay` |

The last two were a single `other-deltas` group of 7 until 2026-09-17, when
§15 split it. `no-delta` composes both halves and therefore still drops the
same nine columns the executed run dropped — a regression test pins that, so
the split refines the registry without invalidating §14.

**Why `delay` is ungrouped, and why that is stated rather than hidden.** It is
the simulated per-message transport delay (`GooseTimestamp - Time`, ±0.24 ms on
this pool): a *relative* timing measure, so it is not `absolute-time`, and not
a frame field, so it is not `goose-header`. It therefore survives all six
ablations, and every run report lists it under `features_in_no_group` — so a
column added to the dataset after this registry was written shows up as an
unclaimed survivor instead of quietly sitting outside the partition while the
D.1 table is read as if the groups covered everything.

### The runs, and what each one asks

| `--feature-set` | drops | n features | The question |
|---|---|---:|---|
| `all` | — | 40 | Reference. Same configuration as the D.3 champion run. |
| `no-electrical` | `electrical` | 22 | Is the grayhole visible in the process data at all, or only in the protocol stream? |
| `no-goose-header` | `goose-header` | 33 | Do the static frame fields carry anything, or are they near-constant padding? |
| `no-absolute-time` | `absolute-time` | 37 | Is the model partly identifying *when* a run happened — a run-identity proxy rather than a signature? |
| `no-delta` | `counter-deltas` + `timing-deltas` + `size-state-deltas` | 31 | Are the revision's own derived deltas doing the work? |
| `no-counters` | `counters` | 38 | Can the model still find the gap when only the deltas expose it? |
| `no-sequence` | `counters` + `counter-deltas` | 36 | With no sequence information at all, is anything left? |

Added 2026-09-17, after §14 closed and outside the preregistered six — §15
records why and reports them:

| `--feature-set` | drops | n features | The question |
|---|---|---:|---|
| `no-timing-deltas` | `timing-deltas` | 37 | Is the carrier inside the deltas the *interval between messages*? |
| `no-size-state-deltas` | `size-state-deltas` | 36 | Or is it the *change in the frame* the delta measures? |

The last two are the pair that matters most for the paper's claim. `SAG.PBM`'s
weakness and the `FRG`/congestion collision both say the models lean on gap
structure; `no-counters` and `no-sequence` are what measure how much.

### The guard that makes a null result trustworthy

`resolve_feature_set` is resolved against the dataset's own column names and is
**fatal** when a group names a column the dataset does not have — including
when only part of a group is missing. This is deliberate: an ablation that
silently drops nothing produces a run identical to the reference, which reads
exactly like the finding "this feature group does not matter". A typo, a
renamed column or a dataset from before a schema change would all land there.
`test_validation_protocol.py`'s `FeatureGroupRegistryTests` pins the partition
(pairwise disjoint, every named set composing real groups) and
`FeatureSetRunnerTests` runs the flag end to end through `main`, on a fixture
whose feature vocabulary *is* the registry.

### Run matrix (~3.7 h serial, ~5.8 GB)

All on the champion (XGBoost), in `none`, from the same persisted splits, with
`--save-scores` — D.5's threshold axis is what these runs are judged on, and it
is unanswerable from hard labels. Each run is ~37 min and ~968 MB. **Do not run
two in parallel**: each needs the whole prepared dataset resident.

```bash
for set in no-electrical no-goose-header no-absolute-time \
           no-delta no-counters no-sequence; do
  python experiments/revision_2026/run_grouped_validation.py \
    --dataset data/runs/gray-GOOSE-runs-prepared.parquet \
    --preparation-report experiments/revision_2026/preparation_audit.json \
    --splits experiments/revision_2026/splits_grouped.json \
    --out-dir results/d1-xgboost-$set \
    --model xgboost --balance none --feature-set $set --seed 42 --save-scores
done
```

**The reference row is the existing `results/v2-xgboost-none`.** Its report
predates `--feature-set`, but `all` resolves to zero dropped columns, so the
configuration is identical rather than merely equivalent. Re-running it with
`--feature-set all` is optional and costs 37 min; its value is a regression
check — `grouped_predictions.csv` must come back **byte-identical by SHA-256**,
the same check the loader rework was held to in `validation_protocol.md`.

Then, over the reference plus the six ablations:

```bash
python experiments/revision_2026/check_prediction_integrity.py \
  --run results/v2-xgboost-none $(printf -- '--run results/d1-xgboost-%s ' \
    no-electrical no-goose-header no-absolute-time no-delta no-counters no-sequence) \
  --out experiments/revision_2026/prediction_integrity_d1.md \
  --json-out experiments/revision_2026/prediction_integrity_d1.json

python experiments/revision_2026/grouped_pr_curves.py \
  --run results/v2-xgboost-none $(printf -- '--run results/d1-xgboost-%s ' \
    no-electrical no-goose-header no-absolute-time no-delta no-counters no-sequence) \
  --prior auto --out experiments/revision_2026/pr_curves_d1.md \
  --curve-csv experiments/revision_2026/pr_curves_d1.csv
```

The reference must be the **first** `--run`: `grouped_pr_curves.py` pairs every
later run against the first, and the paired table is the only thing that
separates two of these configurations (§11). Marginal intervals will overlap
almost everywhere.

### How to read the result

- **AP and budgeted recall, not argmax macro F1** (§11). An ablation that moves
  the score distribution without moving the ranking looks like a large macro-F1
  change and is worth nothing; the reverse is equally possible.
- **Per class.** The four attack classes do not depend on the same groups —
  `SAG.PBM` is a boundary phenomenon and `FRG` is partly unidentifiable by
  construction (`benign_controls.md` §8), so a group that matters for `SAG.DB`
  may be irrelevant to them.
- **A group whose removal changes nothing is a result**, not a failed run — it
  is what licenses dropping those columns from the paper's feature table. The
  guard above is what makes that reading safe.

## 14. D.1 result (2026-09-16)

Seven runs — the reference plus the six ablations of §13 — all XGBoost,
`none`, from the same persisted splits, on the corrected pool. 136 min of
serial compute (18, 23, 26, 21, 24, 24 min), well under the 3.7 h §13
budgeted from D.3's 37 min/run. `check_prediction_integrity.py`: **273 checks,
0 failures**, all seven pairable (`prediction_integrity_d1.md`).

### The table

AP is the verdict (§11); macro F1 is the argmax point estimate, shown because
it is what the run reports print, not because it decides anything. The paired
column is `grouped_pr_curves.py`'s bootstrap over runs, each configuration
scored on the same redrawn runs with its own cross-fold thresholds.

| Run | n feat | macro F1 | AP `ANY_ATTACK` | paired vs reference | 95% CI | separates? |
|---|---:|---:|---:|---:|---|---|
| reference (`all`) | 40 | 0.7288 | 0.8329 | — | — | — |
| `no-goose-header` | 33 | 0.7285 | 0.8327 | -0.0002 | [-0.0005, +0.0001] | **no** |
| `no-electrical` | 22 | 0.7271 | 0.8327 | -0.0001 | [-0.0005, +0.0003] | **no** |
| `no-counters` | 38 | 0.7237 | 0.8306 | -0.0023 | [-0.0051, +0.0004] | **no** |
| `no-absolute-time` | 37 | 0.7257 | 0.8304 | -0.0025 | [-0.0050, -0.0002] | yes |
| `no-sequence` | 36 | 0.7126 | 0.8240 | -0.0089 | [-0.0128, -0.0055] | yes |
| **`no-delta`** | 31 | **0.2259** | **0.1101** | **-0.7227** | **[-0.7478, -0.6956]** | **yes** |

### 1. The delta features are the model

Removing the nine within-trace deltas takes AP on `ANY_ATTACK` from 0.8329 to
**0.1101** — paired, **-0.7227 [-0.7478, -0.6956]**. Per class it is worse
still: `SAG.PB` falls from 0.8551 to **0.0256** (-0.8294), `SAG.DB` from 0.8839
to 0.1126. At a 1-alert-per-100-messages budget the champion's `ANY_ATTACK`
recall goes from 0.4833 at precision 0.9467 to **0.0903 at precision 0.1768**.

Nothing else in this card comes within two orders of magnitude of that. The
entire detection result of the revision rests on nine derived columns.

**This is not leakage, and the distinction matters.** The deltas are computed
by `prepare_grouped_dataset.py` strictly within a trace, the 265
trace-boundary rows are dropped, and `test_validation_protocol.py` asserts
they never cross a boundary. A monitor on a live stream can compute exactly
these quantities from the messages it has already seen. What the result does
mean is that **that one derivation is now load-bearing for every number in the
revision** — a bug in it would not degrade the result, it would be the result.

### 2. Twenty-five of the forty features are free

`no-electrical` (-18 columns) and `no-goose-header` (-7) are both
**indistinguishable from the reference** on `ANY_ATTACK`, and their budgeted
recall/precision at 1% is identical to three decimals (0.4836/0.9464 and
0.4841/0.9457 against 0.4833/0.9467). The Sampled Values — the entire
electrical side of the dataset, 45% of the feature matrix — contribute
nothing measurable to grayhole detection.

Both in fact *improve* `SAG.DB` slightly and the interval separates
(+0.0015 [+0.0001, +0.0032] and +0.0017 [+0.0005, +0.0033]). Two-thousandths
of AP is not a finding to lean on, but it is the expected direction: fewer
irrelevant features, less for the ensemble to split on by chance.

### 3. The signal is in the timing deltas, not in the sequence gap

This inverts §13's expectation, which called `no-counters`/`no-sequence` "the
pair that matters most".

- `no-counters` (drops `StNum`, `SqNum`) does **not** separate: -0.0023
  [-0.0051, +0.0004].
- `no-sequence` (drops the counters **and** `stDiff`/`sqDiff` — every piece of
  sequence information the model has) separates, but costs only **-0.0089**
  AP. It keeps the seven non-counter deltas, and with them it recovers to
  within 1% of the reference.
- `no-delta` removes those seven as well, and collapses.

By elimination, the seven non-counter deltas — `timestampDiff`, `tDiff`,
`timeFromLastChange`, `gooseLengthDiff`, `cbStatusDiff`, `apduSizeDiff`,
`frameLengthDiff` — are **sufficient**, and the sequence columns are
**redundant given them**. Stated carefully: this does not prove the counter
deltas carry nothing, only that whatever they carry is also carried elsewhere.

That is coherent with two things measured earlier and not understood at the
time. `SAG.PBM` is the weakest class everywhere, and its discards happen at
state boundaries where `SqNum` resets anyway (`label_duplication_audit.md`
§7) — a sequence-based detector cannot see them, a timing-based one partly
can. And `FRG`/congestion loss collide (`benign_controls.md` §8): uniformly
random drops leave no counter pattern, only a stretched interval, which is
exactly what congestion also produces.

### 4. Statistically real is not operationally relevant

At the 1% alert budget, every configuration except `no-delta` lands at
`ANY_ATTACK` recall 0.482-0.486 and precision 0.941-0.947 — inside each
other's intervals. `no-absolute-time` and `no-sequence` separate on AP and
change nothing an operator would notice.

The honest summary is **binary, not a ranking**: with the deltas, this
detector; without them, nothing. The other five ablations are evidence that
the remaining feature groups are not where the signal lives, not a league
table of their importance.

### Consequences

- **The paper's feature table can shrink to the deltas plus a small
  remainder.** 25 of 40 columns are demonstrably free, which is a positive
  result for deployment cost, not a negative one.
- **Card F (SHAP) should be aimed at the nine deltas.** An explanation of the
  electrical features would be explaining columns the model does not use.
- ~~**A finer split is worth two more runs.**~~ **Done — §15.** `other-deltas`
  mixed timing (`timestampDiff`, `tDiff`, `timeFromLastChange`) with
  size/state (`gooseLengthDiff`, `apduSizeDiff`, `frameLengthDiff`,
  `cbStatusDiff`). Both halves were ablated on 2026-09-17: the timing half
  carries it (-0.0645 AP paired, and the operating point moves), the
  size/state half is free (-0.0004).
- **D.2's rule-based baseline now has a specific target.** The rule to beat is
  not a `sqNum` gap detector but an inter-message interval threshold, and §2's
  scope for D.2 ("`sqNum`/`stNum` gap detector + delay threshold") should be
  read with the second half carrying the weight. §15 measured that inference rather than
  leaving it inferred.

### Measured memory footprint (supersedes §7's projection)

Sampled every 15 s during three of the six runs (process RSS, system free):

| Run | n feat | peak RSS | min system free |
|---|---:|---:|---:|
| `no-delta` | 31 | 4.77 GB | 3.66 GB |
| `no-sequence` | 36 | 5.21 GB | 3.37 GB |
| `no-counters` | 38 | 5.41 GB | 2.80 GB |

**There is no training peak.** With the streaming loader, RSS is flat from the
end of the load to the last fold — the `X_train` copy and XGBoost's internal
structures do not produce the spike §7 projected for a full partition. Peak
scales mildly with feature count (~0.05 GB per column here), so the 40-feature
reference sits near 5.6 GB.

Four of these runs were nonetheless killed by the environment's low-memory
watchdog before completing, at ambient free memory of 7.0-8.5 GB. The runs
were not the cause; they were merely resident when other pressure crossed the
threshold. One of the four was self-inflicted — relaunching immediately after
a kill, before the OS had reclaimed the dead process's pages. **Leave the
machine quiet and wait for memory to be returned before relaunching.**

## 15. Splitting the non-counter deltas (2026-09-17)

§14 named the seven non-counter deltas as the carriers **by elimination**:
`no-delta` collapses the detector and `no-sequence` barely moves it, so
whatever is left must be in those seven. That group mixed two different
physical quantities — the interval between messages and the change in the
frame itself — and the distinction matters twice over: D.2's rule-based
baseline is an interval threshold, and card F will have to explain these
columns by name.

Two runs, outside the preregistered six and recorded as such: the group was
split into `timing-deltas` (3 columns) and `size-state-deltas` (4), and each
half was ablated on its own. Same champion configuration as every other D.1
run — XGBoost, `--balance none`, the persisted `splits_grouped.json`, dataset
SHA `3109e4d4…`, `--save-scores`. **`no-delta` still drops the same nine
columns**, pinned by `test_no_delta_is_still_the_union_of_the_three_delta_groups`,
so §14's numbers stand unchanged and the new runs are comparable against them.

Cost: **~40 min** of serial compute for the two runs, 1.9 GB. Integrity:
`prediction_integrity_d1_split.md` — 5 runs × 39 checks, **0 failures**, all
four pairable against the reference. Curves:
`pr_curves_d1_split.md`.

### The result

AP on `ANY_ATTACK`, and the paired difference against the `v2-xgboost-none`
reference (same resampled runs, so shared variation cancels):

| Run | n feat | AP `ANY_ATTACK` | paired B − A | separates? |
|---|---:|---|---:|---|
| `v2-xgboost-none` (reference) | 40 | 0.8329 [0.8037, 0.8618] | — | — |
| `no-size-state-deltas` | 36 | 0.8325 [0.8032, 0.8616] | **−0.0004** [−0.0007, −0.0001] | yes, and meaningless |
| `no-sequence` | 36 | 0.8240 [0.7934, 0.8531] | −0.0089 [−0.0128, −0.0055] | yes |
| `no-timing-deltas` | 37 | 0.7683 [0.7296, 0.8021] | **−0.0645** [−0.0832, −0.0501] | yes |
| `no-delta` | 31 | 0.1101 [0.0906, 0.1344] | −0.7227 [−0.7478, −0.6956] | yes |

**The timing half is where the signal is, and it is the only ablation besides
`no-delta` that an operator would notice.** Dropping the three timing deltas
separates on *every* class — `FRG` −0.1293, `SAG.PBM` −0.0673, `SAG.PB`
−0.0500, `SAG.DB` −0.0387 — and it moves the operating point, which §14 said
nothing else did: at the 1% alert budget recall falls 0.4833 → 0.4512 and
precision 0.9467 → 0.8791 (paired recall −0.0321 [−0.0475, −0.0167]). It is a
7.7% relative loss of AP against a 7.5% loss of features.

**The size/state half is free.** −0.0004 AP on `ANY_ATTACK` and
*indistinguishable* on all four attack classes individually; at the 1% budget
it reproduces the reference to the fourth decimal (recall 0.4832 vs 0.4833,
precision 0.9464 vs 0.9467). The `ANY_ATTACK` interval excludes zero by
0.0001, which is a demonstration that the paired test has resolution, not a
finding. Four more of the forty columns are demonstrably free: **29 of 40**.

### What this does not say

It does not name a sufficient sub-group. Removing the timing half costs
−0.0645; removing the size/state half costs nothing; removing all nine costs
−0.7227. Neither half alone comes within an order of magnitude of the
collapse, so the remaining features **reconstruct most of the signal whenever
any one piece is taken away**. The honest statement is the same shape as
§14's, one level down: the deltas are jointly necessary, and within them the
timing columns are the only sub-group whose removal is operationally visible.
Naming a minimal sufficient set would need a different experiment — a forward
selection rather than a leave-one-group-out — and nothing in card D asks for
one.

### Consequences

- **D.2's target is confirmed, not just sharpened.** §14 inferred an
  inter-message interval threshold from the elimination argument; this
  measures it. The rule to beat operates on `timestampDiff`/`tDiff`, and
  `cbStatusDiff`-style frame-change rules are not worth implementing.
- **Card F should aim at the three timing deltas first.** They are the only
  columns whose removal changes what an operator sees.
- **Coherent with the two known weaknesses.** `SAG.PBM` discards at state
  boundaries, where `SqNum` resets and only the interval survives; `FRG` and
  congestion loss both leave a stretched interval and no counter pattern
  (`benign_controls.md` §8). Both are the classes the timing ablation hurts
  most.

## 16. D.2 result: the rule the model has to beat (2026-09-18)

Six rules, one feature and one comparison each, calibrated on each fold's
train partition alone and scored on the persisted `splits_grouped.json` the
learned runs use. Implementation, artifact contract and the reason the
multiclass projection is not a result: `run_rule_baseline.py`'s module
docstring and §2's preregistered scope. Cost: **~2 min 20 s per rule, 14 min
for all six**, 5.4 GB — the rule reads one column, so it never pays the
40-feature load. Integrity: `prediction_integrity_d2.md`, 7 runs × 39 checks,
**0 failures**, every rule pairable with the champion row for row.

### The result

At each rule's own calibrated threshold, and threshold-free as AP on
`ANY_ATTACK` (prevalence 1.9582%, so a coin scores 0.0196):

| Rule | column | recall | precision | F1 | alert rate | AP `ANY_ATTACK` |
|---|---|---:|---:|---:|---:|---|
| **`interval-timestamp`** | `timestampDiff` | 0.4416 | 0.6229 | **0.5144** | 1.45% | **0.2411** [0.1895, 0.3052] |
| `stnum-gap` | `stDiff` | 0.4116 | 0.3981 | 0.4025 | 2.04% | 0.2192 [0.1754, 0.2712] |
| `interval-t` | `tDiff` | 0.4714 | 0.1497 | 0.2258 | 6.29% | 0.1010 [0.0822, 0.1237] |
| `sqnum-gap` | `sqDiff` | 0.3559 | 0.1383 | 0.1982 | 5.20% | 0.0451 [0.0338, 0.0592] |
| `delay` | `delay` | 0.8882 | 0.0246 | 0.0478 | 72.6% | 0.0175 [0.0147, 0.0211] |
| `time-since-change` | `timeFromLastChange` | 1.0000 | 0.0202 | 0.0395 | 99.98% | 0.0127 [0.0105, 0.0153] |

Train and test F1 agree to within 0.003 on every rule, so no threshold is
overfitting its fold; the last two rules do not have an operating point at all
— their F1 optimum is "alert on almost everything", which is the shape of a
feature that carries nothing, not of a badly calibrated detector.

### The model earns its complexity

Paired against the champion (`v2-xgboost-none`, AP 0.8329 [0.8037, 0.8618]),
both scored on the same resampled runs:

| Rule (B) | B − A average precision | 95% CI | separates? |
|---|---:|---|---|
| `interval-timestamp` | **−0.5918** | [−0.6385, −0.5392] | **yes** |
| `stnum-gap` | −0.6137 | [−0.6509, −0.5708] | yes |
| `interval-t` | −0.7319 | [−0.7588, −0.7035] | yes |
| `sqnum-gap` | −0.7878 | [−0.8186, −0.7557] | yes |
| `delay` | −0.8153 | [−0.8439, −0.7867] | yes |
| `time-since-change` | −0.8202 | [−0.8489, −0.7921] | yes |

At a matched 1% alert budget the champion recovers recall 0.4833 at precision
0.9467 against the best rule's 0.2980 at 0.5842 (paired recall −0.1853
[−0.2436, −0.1332]). **The learned detector is worth roughly 3.5x the best
single threshold in AP, and the separation is unambiguous at every budget.**
That is the answer card D.2 exists to give.

The baseline is not a straw man, and saying so is part of the result: at AP
0.2411 against a 0.0196 floor, a single threshold on the inter-message
interval is **12.3x better than chance**. A reviewer asking "would a threshold
have done?" gets a quantified no rather than an assertion.

### Where §15's prediction held, and where it did not

§15 predicted the interval rule would beat the preregistered `sqNum` gap
detector. It does, decisively — paired **−0.1960 [−0.2535, −0.1481]**,
separating, a factor of 5.3 in AP (`pr_curves_d2_rules.md`). Keeping the
preregistered arm rather than dropping it on that prediction is what turned it
into a measurement.

What §15 did not anticipate: **`stnum-gap` is statistically indistinguishable
from the interval rule** — paired −0.0219 [−0.0884, +0.0452], interval as
reference. So "the interval is *the* best rule" is not established; what is
established is that the *state* counter and the interval are two equally good
single thresholds and the *sequence* counter is far behind. The two are not
redundant either: the interval rule's per-class curve carries `FRG` (AP 0.1457
against a 0.0058 floor, 25x) while `stnum-gap`'s does not (0.0066, at the
floor). They detect different attacks at the same aggregate score, which is
coherent with §15 — `FRG` is uniformly random loss, visible only as a
stretched interval.

### Reading the per-class rows of a rule run

A rule has one score and cannot name a family, so its posterior block puts
that score on one designated attack class (the fold's train-majority attack —
`FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` in all five folds of all six rules) and
**exactly zero** on the other three. Their per-class AP therefore comes back
at the prevalence floor by construction; it measures the projection, not the
rule. `run_rule_baseline.py` records this in every report under
`rule.per_class_curves` and `test_rule_baseline.py` pins the zero columns.
Read a rule run's `ANY_ATTACK` row, and its designated class's row only as a
statement about that one class.

### Status

D.2 closes. Execution order continues D.4 → D.5.
