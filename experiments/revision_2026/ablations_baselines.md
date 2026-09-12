# Ablations and baselines (Card D)

Checklist ref.: D.1–D.5 in `prioridades_revisao_gray_goose.pdf`, P1 "Ablation
study por grupos de features" + "Baselines fortes". See `README.md` for how
this fits the rest of the revision, `validation_protocol.md` for the grouped
workflow every run here must reuse, and its "Balancing scenarios" section for
the card-E results this card is the direct follow-up to.

**Status: D.3 closed (2026-09-12).** §7 (the per-fold train subsampling
policy), §8 (the run matrix) and §9 (the result) are done; D.1, D.2, D.4 and
D.5 are still plan only. §6 is the authoritative tracker.

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

Calibration: the three existing full runs (decision tree, 5 folds, 20,796,921
rows) each took **~20–50 min wall clock** on this machine. Runs do not
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

## 9. D.3 result (2026-09-12)

Nine runs, ~2.5 h wall clock, all `full_grouped_run` on the 20,796,921-row
pool. Full tables are in `validation_protocol.md`, "Model family comparison
(checklist D.3)"; this section records what the card needs to carry forward.

### Champion: XGBoost

Decided on the pre-registered criterion in §8, and it went to the third
tie-breaker because the first two were ties:

| Model (`downsample`) | macro F1 | mean attack recall | ideal-`normal` attack_fpr |
|---|---:|---:|---:|
| **xgboost** | **0.1972** | **0.6862** | **38.69%** |
| decision-tree | 0.1970 | 0.6858 | 43.46% |
| random-forest | 0.1766 | 0.5600 | 36.67% |
| logistic-regression | 0.1355 | 0.5205 | 42.21% |

XGBoost and the decision tree are separated by 0.0002 macro F1 against a
per-fold standard deviation of ~0.013, and by 0.0004 mean attack recall — so
the decision rests entirely on attack_fpr, where XGBoost costs 4.8 percentage
points less on ideal traffic. Random Forest has the lowest attack_fpr but
never reaches that tie-breaker, losing the first criterion by ~0.02.

### The question card E left open, answered

Zero attack recall on the unbalanced pool is **not** a
`DecisionTreeClassifier(max_depth=8)` limit: XGBoost at defaults, on the full
~16M-row partition, also predicts an attack class for essentially no row, and
logistic regression predicts `normal` for literally every row (macro F1
0.1649). Random Forest is the only family that predicts any attack row at all
(`SAG.PBM` recall 0.047–0.078 at precision 0.169–0.248) — at ≤8% recall it is
no detector, but its 0.11% attack_fpr on ideal `normal` traffic is the first
point in this revision where attack predictions are not simply noise. The
decision-tree control at the same 4M cap is indistinguishable from the
uncapped tree (macro F1 0.2759 vs 0.2758; 0.026% discordant rows), so that is
the family, not §7's cap.

### Consequences for the rest of card D

- **D.1 gets much cheaper than §4 budgeted.** The champion's `downsample` run
  takes 2.2 min, so six ablation runs are ~15 min of compute, not 4–6 h. The
  §4 estimate assumed the champion might be an expensive family; it is not.
  This does **not** license widening the ablation scope — the deferrals in §3
  were argued on methodology (top-k SHAP needs card F's held-out importances)
  and on what the checklist asks for, not on compute alone.
- **D.4 has a specific target.** The Random Forest result says model *capacity*
  is what separates "predicts nothing" from "predicts something precisely but
  rarely", which makes tree depth / estimator count the hyperparameters D.4
  should spend its subsampled grid on first.
- **Two attack classes are under-powered at the unit level, and this bounds
  D.1 and D.4.** `SAG.DB` and `FRG` have only 15 independent runs each, so a
  fold's test partition can hold a single run of them and its per-fold recall
  becomes a one-run measurement (`validation_protocol.md`, "How many attack
  rows are actually being counted"). An ablation or tuning result that moves
  only `SAG.DB`/`FRG` is not yet evidence; one that moves `SAG.PB`/`SAG.PBM`
  (45 runs, 6–13 test runs per fold) is. Read the ablation tables that way.
- **A caveat travels with the logistic-regression rows.** It did not converge
  in `downsample` (`n_iter_` = `max_iter` = 100 in all 5 folds), though it did
  in `none` (49–61 iterations). Raising `max_iter` is D.4's business, not
  D.3's; the number is reported with the caveat rather than tuned.

### Runner rework and its regression check

D.3 needed `run_grouped_validation.py` to load and predict differently (§7,
"Memory prerequisite", plus bounded-block prediction after the
`logistic-regression` run overran RAM while predicting a 5.58M-row test
partition). Both card-E decision-tree runs were re-executed on the reworked
runner: `grouped_predictions.csv` came back **byte-identical by SHA-256** in
both scenarios over all 20,796,921 rows. The rework is a memory and wall-clock
change only — nothing in cards A/B/E needs revisiting.
