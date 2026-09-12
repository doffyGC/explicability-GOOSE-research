# Ablations and baselines (Card D)

Checklist ref.: D.1–D.5 in `prioridades_revisao_gray_goose.pdf`, P1 "Ablation
study por grupos de features" + "Baselines fortes". See `README.md` for how
this fits the rest of the revision, `validation_protocol.md` for the grouped
workflow every run here must reuse, and its "Balancing scenarios" section for
the card-E results this card is the direct follow-up to.

**Status: planned, not started.** Nothing in this document has been executed
yet; the tables below are the plan and the scope decision, not results.

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
| D.1 feature-group ablation (6/7) | not started | — |
| D.1 top-k SHAP group | deferred to card F | — |
| D.2 rule-based baseline | not started | — |
| D.3 model comparison (XGB/RF/LR) | not started | — |
| D.3 temporal model | deferred | — |
| D.4 nested tuning | not started | — |
| D.5 per-class cross-run report | not started | — |
