# Card F: explainability on held-out grouped data

**Preregistered 2026-09-21, before any SHAP value was computed.** Written
first for the same reason `ablations_baselines.md` §17 was: an explanation
shaped after seeing its own output is not an explanation, it is a caption.
What follows fixes the protocol, the expectations and the falsification
conditions; the result belongs in a later section of this file.

## 1. Why this card exists

`prioridades_revisao_gray_goose.pdf`, card F ("Estatística e
explicabilidade"), lists five items. **Two are already closed** by work done
for cards D/E, and are recorded here so the card is not reopened for them:

| Item | Status |
|---|---|
| F.1 Replace "one million independent messages" statistics with run/fold/group-based statistics | **done** — `bootstrap_run_intervals.py` resamples `split_group`, never rows; `grouped_pr_curves.py` pairs over the same runs |
| F.2 Report a confidence interval per class and per scenario | **done** — `run_bootstrap.{none,downsample,champion}_v2.md`, per class, both scenarios |
| **F.3 Recompute SHAP only on held-out grouped data, with background from train** | **this card** |
| **F.4 Report the stability of the importances across folds/runs** | **this card** |
| F.5 Remove causal language: SHAP indicates association in the model, not physical cause | manuscript work (card H); §7 below fixes the wording this card is allowed to produce |

F.3 is the **last unfixed defect** of the four that motivated the whole
revision (`README.md`): message-level splitting, missing benign controls and
message-level statistics are closed; "SHAP computed on a CV validation fold
rather than a proper held-out set" is not. `explainability/shap_analysis.py`
in the frozen baseline calls `shap.Explainer(model)` on the **last CV fold's
validation set**, which under `StratifiedKFold` at message level contains
messages from the same runs the model was fitted on. Every importance in the
submitted paper is therefore an importance measured partly on training data.

## 2. The protocol

### 2.1 Held-out means the outer test groups, and nothing else

SHAP is computed on rows from each fold's **test** partition of the persisted
`splits_grouped.json` — the same folds every card-D number comes from, bound
by the same dataset SHA-256. No fold is generated internally. The background
distribution comes from the **train** partition of the *same* fold, which is
what F.3 asks for and what makes the attribution "against what this model was
fitted on" rather than "against the rows being explained".

### 2.2 The model explained must be the model published

No run in `results/` persists a fitted estimator - `run_grouped_validation.py`
writes predictions, posteriors and metrics, not models. So this card refits
the champion per fold, and that refit is the card's first hazard: a SHAP
value computed on a *different* model than the one the paper reports explains
nothing about the paper.

It is therefore verified rather than assumed. The refit uses the same
persisted fold, the same seed, the same feature set and the same library
defaults as `results/v2-xgboost-none`, and **its predictions on the test fold
must reproduce that run's `grouped_predictions.csv` row for row**. A
mismatch is fatal and stops the card. This is the same reproduction check
`validation_protocol.md` ("Runner reproducibility") already applies to the
streaming loader, pointed at a new consumer.

### 2.3 What is computed

For each fold, on the sampled held-out rows:

- `shap.TreeExplainer(model, data=background, feature_perturbation=
  "interventional")` — the background is required by F.3, and interventional
  is the perturbation that uses it. `tree_path_dependent` would ignore the
  background entirely and quietly answer a different question.
- Per-class SHAP values (the champion is 6-class `multi:softprob`), kept
  **per class**, never summed across classes into one "importance" — a
  feature that separates `SAG.DB` from `SAG.PB` and a feature that separates
  `normal` from everything are not the same finding, and `ablations_baselines.md`
  §5 requires per-class before any average.
- Per-feature mean |SHAP| per class per fold, which is the quantity F.4 asks
  to be reported across folds.

## 3. The sample, and the bias it carries

A fold's test partition holds ~2.2M rows and the pool is 11,057,478; SHAP with
an interventional background is O(rows × background × trees), so the full
partition is not affordable and a sample is preregistered rather than chosen
later.

- **Explained rows**: sampled per fold, stratified inside every
  (`split_group`, `class`) stratum by the same `subsample_train` used for
  D.4's inner search, so all ~53 test runs of a fold stay represented and no
  rare class is emptied. What shrinks is rows per run.
- **Background**: a separate stratified sample of that fold's **train**
  partition, small by necessity (interventional cost is linear in it).

The bias this leaves points one way and is stated in advance: a small
background makes the reference distribution coarser, which tends to *flatten*
differences between features rather than invent them. So a feature that still
separates under this budget is separating in spite of the sample, and a
feature that does not separate is the weaker claim of the two.

## 4. What is expected, written before running

### 4.1 The prediction D.1/D.5 already earned

`ablations_baselines.md` §15: dropping the three timing deltas
(`timestampDiff`, `tDiff`, `timeFromLastChange`) costs **-0.0645 AP**
[-0.0832, -0.0501] on `ANY_ATTACK`, separates in **every** class, and is the
only ablation besides `no-delta` that moves the 1% operating point (recall
0.4833 → 0.4512, precision 0.9467 → 0.8791).

**So the three timing deltas are expected to rank at or near the top of the
per-class attributions.** This is the tightest prediction any card in this
revision has made before running, and it is falsifiable: if they do not, the
next section says what that means.

### 4.2 Ablation and attribution are different questions - stated now, so a
disagreement is not written up as a discovery

This is the trap this section exists to disarm. A feature-group ablation
measures **necessity given everything else**; SHAP measures **attribution
inside the fitted model**. A group can be

- *unnecessary but heavily attributed* — it carries signal the model does
  use, which the surviving columns can reconstruct once it is gone. §14 found
  the 18 electrical and 7 GOOSE-header columns **free** (indistinguishable
  from the reference when removed). That is a statement about redundancy, and
  it does **not** predict that SHAP ignores them. If SHAP attributes to the
  electrical columns, the honest reading is "redundant, not unused" — not
  "SHAP contradicts D.1".
- *necessary but modestly attributed* — attribution spread thinly across a
  group whose members substitute for one another.

Therefore: **no result of this card may be reported as SHAP confirming or
refuting an ablation** unless the two are compared on the axis where they
answer the same question. The one place they do is the timing deltas, because
§15 measured their removal as the only change that moves the operating point
— there, "necessary" and "attributed" should coincide, and a divergence is a
real finding about one of the two methods.

### 4.3 Expected nulls, stated in advance

- **`SAG.PBM`**: expect weak and unstable attributions. It is the weakest
  ranker (AP 0.4094, recall 0.2721) and `label_duplication_audit.md` §7 bounds
  it at the label level — most of its discards happen at a state boundary,
  where an isolated message carries no information about whether a discard
  preceded it. An explanation cannot be more stable than the label it
  explains.
- **`FRG`**: expect attributions that look like benign congestion, because
  `FRG` **is** the `CONGESTION_LOSS` control by construction — 15 of its 45
  runs are byte-identical to it (`benign_controls.md` §8). If SHAP produces a
  clean, distinctive `FRG` signature, something is wrong with the reading,
  not right with the model.
- **Absolute time** (`Time`, `t`, `GooseTimestamp`): expect *low*
  attribution. High attribution here would mean the model is partly
  identifying *when* a run happened — a run-identity proxy rather than a
  signature — which is exactly what the grouped protocol exists to expose.
  §14 measured its removal as free, so under §4.2 low attribution is expected
  but not guaranteed; this one is worth watching because a surprise here is a
  leakage signal rather than a redundancy story.

### 4.4 Stability (F.4)

Importances are reported **per fold**, with the spread across the five folds
shown rather than averaged away. Expected: tight agreement on the top
features, wider spread on classes carried by few runs — `SAG.DB` and `FRG`
have 45 runs each across five folds, so a fold's test partition holds ~9 runs
of each, and `bootstrap_run_intervals.py` already warns that a class resting
on a handful of runs moves in steps of roughly one run's worth of the metric.
The same floor applies here, and a fold-to-fold spread is **not** an error
bar over the generator: it is five measurements over one partition of the
same 265 runs.

## 5. Deliberately excluded

| Not in this card | Why |
|---|---|
| SHAP on a second model family | Card F explains the reported detector. Random Forest ranks worse (§19) and is not the published configuration; explaining it would produce a second set of importances nobody will cite. |
| SHAP on the tuned champion | §18: the tuned model is indistinguishable from the default one (-0.0001 AP paired). Two explanations of the same model is not two results. |
| The top-k SHAP ablation (D.1's deferred 7th group) | It consumes this card's output, so it follows the result rather than accompanying it. `ablations_baselines.md` §2 defers it here explicitly. |
| Causal language of any kind | F.5. SHAP attributes a model's output to its inputs. It says nothing about whether the feature *causes* the attack, and this card's output must not be written as if it does. |

## 6. Cost

Refit: five folds of XGBoost on ~8.8M train rows each. Measured elsewhere,
not estimated - the champion's full run took **37 min** for five folds (§4)
and D.4's individual refits ran ~8 min each (§18). SHAP itself is linear in
(explained rows × background rows × trees) and is the term the sample in §3
controls. The budget for this card is the same 4-8 h band §4 uses.

## 7. The wording this card is allowed to produce (F.5)

Fixed in advance so the manuscript inherits it rather than negotiating it:

- **Allowed**: "the model's output for `SAG.DB` is attributed mainly to
  `timestampDiff`"; "removing the timing deltas costs -0.0645 AP".
- **Not allowed**: "`timestampDiff` causes the detection"; "the attack is
  characterised by a longer interval"; "SHAP shows that grayholes stretch the
  interval". The first is about the model, the second and third are claims
  about the physical process that this experiment does not test.
- The label is per-message and bounds every explanation exactly as it bounds
  every recall number (`label_duplication_audit.md` §7). Any sentence about
  *what the detector sees* carries that bound.

## 8. Status

**Preregistered; not implemented, not executed.** The implementation and the
result each belong in a new section rather than in this one.
