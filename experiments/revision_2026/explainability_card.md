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

### 3.1 Amendment, 2026-09-21: the budget, measured

**Made before any importance value was read.** The wiring smoke was killed
mid-fold and its output was never inspected beyond its timing lines; nothing
below was informed by a ranking.

Two numbers came out of pricing the card rather than estimating it, and both
change §3 as originally written:

**The background sampler had a floor.** Sampling the background with the same
(`split_group`, `class`) stratification used for the explained rows cannot
return fewer rows than it has non-empty strata. Asked for 100, it returned
**411** on a 265-run smoke, and would return ~1,000 on the real 212-run train
partitions. Interventional SHAP costs O(explained × background), so the card
was silently an order of magnitude more expensive than the flag said.

*Corrected:* the background is now stratified **by class only**
(`background_sample`). It is a reference distribution - "what does this model
see normally?" - not an evaluation sample, so it does not need every training
run represented; the explained rows keep their (`split_group`, `class`)
stratification, because there representation is the point. Classes are kept
whatever their prevalence: a background missing a class gives that class's
attributions a reference the model never sees.

**The preregistered 200,000 explained rows per fold was wrong by two orders of
magnitude.** Measured on the smoke: **1,980 rows × 6 classes against 411
background rows took 6.9 minutes** - ~4.8 rows/second, on a model fitted to
164k rows. The real model is fitted to ~8.8M and carries fuller trees, so the
rate is lower. At 200,000 rows per fold this card would have run for days.

*Corrected budget, which is what will be run:* **20,000 explained rows and
100 background rows per fold**, five folds. That keeps the card inside §4's
4-8 h band, which is the same band every other card in this revision was held
to.

What this costs the claim, stated now: 20,000 rows spread over a fold's ~53
test runs and six classes is ~60 rows per (run, class) stratum. That is
comfortable for a **mean** |SHAP| per feature, which is what F.3 asks for, and
it is thin for anything that needs the tail of the distribution - so no claim
in this card may rest on a small number of high-attribution rows. The
fold-to-fold spread reported for F.4 is the check on that: a feature whose
ranking is stable across five independent samples of five different fold
partitions is not an artifact of 20,000 rows.

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

`--limit-folds N` prices the card on the real pool before committing to all
five folds, and marks its output `partial_probe` so a run made for timing can
never be mistaken for a result. §3.1 is what that flag was added to record.

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

**Preregistered and implemented (`run_grouped_shap.py`, 33 tests); not yet
executed on the pool.** The result belongs in a new section rather than in
this one.

The implementation's load-bearing check is the one §2.2 demanded, and it is
established by outcome rather than by code reading: `test_grouped_shap.py`
produces a reference run with `run_grouped_validation.py` and then explains it
**with verification on**, so a passing test means the refit reproduces that
run's predictions row for row - same fold rows, same seed, same estimator.

## 9. F.3/F.4 result: the model leans on the counters, and needs the clocks (2026-09-21)

Executed 2026-09-21, five folds, **99,744 held-out rows explained** against
100-row train backgrounds. `results/f3-xgboost-shap`.

### The two gates, before any ranking was read

- **The model explained is the model published.** Every fold was refit from
  the persisted split at the reference run's seed and thread count, and the
  refits reproduce `results/v2-xgboost-none/grouped_predictions.csv` across
  **11,057,478 rows, 0 mismatches** — the whole pool, since the five test
  partitions cover it exactly once.
- **The conditional axis changed nothing that already existed.** The run was
  repeated after adding the class-conditional aggregation; its 720 global
  values (6 classes × 40 features × 3 statistics) are **identical** to the
  archived global-only run in `results/f3-xgboost-shap-globalonly`. The
  sample and the seed did not move, so anything else would have been a bug.

### §4.1's prediction did not hold

§4.1 preregistered the three timing deltas at or near the top, on the
strength of §15's ablation. They are not there. Share of total conditional
attribution (a class's own score, on held-out rows whose true class it is):

| Class | rows | dominant groups | timing deltas |
|---|---:|---|---:|
| `SAG.DB` | 461 | counter-deltas **51.6%** + counters 28.6% | 5.3% |
| `SAG.PB` | 394 | counter-deltas **47.0%** + counters 23.9% | 11.3% |
| `FRG` | 590 | counters **32.9%** + counter-deltas 28.3% | 17.1% |
| `SAG.PBM` | 485 | counters **32.1%** | **21.0%** |
| `benign_degradation` | 2,576 | **absolute time 47.6%** | 25.2% |
| `normal` | 95,238 | **timing deltas 28.9%** | 28.9% |

`stDiff` alone carries `SAG.DB` (7.78, about 3× the next feature) and
`SAG.PB` (6.25). The prediction was wrong, and it was written down in
advance, which is the only reason that sentence can be said plainly.

### What replaced it, and why it is not a contradiction of D.1

§4.2 exists for exactly this moment: **ablation measures necessity given
everything else; SHAP measures attribution inside the fitted model**, and a
group can be heavily attributed yet removable.

Put the two together and they are consistent, not in tension:

- §14 measured that removing **all** sequence information costs only
  **-0.0089 AP**. The counters are *replaceable* — the timing deltas
  reconstruct what they carried.
- §15 measured that removing the three timing deltas costs **-0.0645 AP**,
  separates in every class, and is the only ablation besides `no-delta` that
  moves the 1% operating point. The clocks are *not* replaceable.
- This card measures that the fitted model nonetheless **attributes** 70-80%
  of its burst-family scores to the counters.

**The model leans on the counters because they are the cleaner signal when
present; it needs the clocks because nothing else covers the case where the
counters go blind.** That is a sharper statement than the one predicted, and
it is only sayable because two methods that answer different questions were
both run.

The case where the counters go blind is not hypothetical, and the card
locates it: `SAG.PBM`'s discards happen at a state boundary, where `SqNum`
resets (`label_duplication_audit.md` §7). It is the **only attack class
where timing leads** — `timeFromLastChange` second, 21.0% share — and it is
the class the detector is worst at (AP 0.4094, recall 0.2721). The
explanation and the limitation are the same fact seen twice.

### §4.3's leakage watch fired

`benign_degradation` attributes **47.6% to absolute time** — `Time` first,
`GooseTimestamp` second, `t` fourth — the largest group share of any class
in the table. §4.3 wrote, before the run: high attribution here "would mean
the model is partly identifying *when* a run happened — a run-identity proxy
rather than a signature — which is exactly what the grouped protocol exists
to expose."

It is also the **least stable** group. Across the five folds, top features
elsewhere sit at max/median **1.03-1.5**; the absolute-time features on
`benign_degradation` sit at **2.13-2.53**. A feature whose attribution
doubles depending on which runs are held out is behaving like a run
identifier, not like a signature.

**This is a signal, not a proof**, and the remedy is cheap but **not free** —
a claim worth stating precisely, because getting it wrong in the other
direction would make the recommendation look costless when it is not. §14's
paired table:

| Ablation | paired AP on `ANY_ATTACK` | 95% CI | separates? |
|---|---:|---|---|
| `no-electrical` (-18 cols) | -0.0001 | [-0.0005, +0.0003] | no |
| `no-goose-header` (-7 cols) | -0.0002 | [-0.0005, +0.0001] | no |
| **`no-absolute-time` (-3 cols)** | **-0.0025** | **[-0.0050, -0.0002]** | **yes** |

So dropping the three clocks costs a small but statistically separating
-0.0025 AP, and per class -0.0046 (`SAG.PB`) to -0.0396 (`FRG`). At the 1%
alert budget the picture is softer still: of five targets only `SAG.PB`
separates, at -0.0014 recall.

That makes this a **trade-off rather than a free win**, and the trade is
worth naming: -0.0025 AP against a detector that no longer has wall-clock
columns to lean on. An attribution that doubles depending on which runs are
held out will not survive deployment in a substation whose clock has no
relationship to this generator's, so the 0.0025 is arguably a price for a
number that transfers rather than a loss. The evaluation is §10 — and the
question it has to answer is not only *what does removing the clocks cost*,
which §14 already measured, but *whether the model then finds another
run-identity proxy to lean on*, which only SHAP can see.

### F.4: stability

Reported per fold and shown rather than averaged. Top features are stable
across the five folds at **max/median 1.03-1.5**, which is tighter than the
~44-190 rows per attack class per fold would suggest, and is the check that
those thin per-class samples are not driving the ranking. The exceptions are
the absolute-time features, above.

Two floors travel with every number here, both preregistered: the five folds
are one partition of the same 265 runs rather than five draws from the
generator, and a class's conditional rows come from ~9 test runs per fold, so
the spread moves in steps of roughly one run's worth of attribution.

### What this card does not say

No statement here is causal. SHAP attributes **this model's output** to
**its inputs**; that `stDiff` carries `SAG.DB` is a fact about the detector,
not about what a grayhole does to a substation (F.5, §7). The label is
per-message, which bounds every explanation exactly as it bounds every
recall number.

### Status

**Done.** Evidence: `results/f3-xgboost-shap` (`shap_importances.json` /
`.md`); `results/f3-xgboost-shap-globalonly` (the regression reference).

## 10. Does removing the clocks remove the proxy? (2026-09-21)

§9 found `benign_degradation` attributing 47.6% of its score to absolute
time, unstably, and §14 had already measured what removing those three
columns costs in detection (-0.0025 AP paired, separating). What neither
measured is the question that decides whether the removal is worth making:
**with the clocks gone, does the model find another position proxy to lean
on?**

`results/f3-xgboost-shap-no-absolute-time`, explaining
`results/d1-xgboost-no-absolute-time` (37 features). Five folds, 99,744
held-out rows, refit verified against that run across **11,057,478 rows, 0
mismatches**.

### On the four attack families: clean

Attribution redistributes proportionally across the groups that were already
there. Share of conditional own-class attribution, full model → no-clocks
model:

| Class | counter-deltas | counters | timing deltas | abs-time |
|---|---|---|---|---|
| `SAG.DB` | 51.6% → 57.1% | 28.6% → 30.1% | 5.3% → 6.5% | 9.2% → 0 |
| `SAG.PB` | 47.0% → 51.0% | 23.9% → 23.2% | 11.3% → 10.9% | 3.2% → 0 |
| `FRG` | 28.3% → 30.3% | 32.9% → 38.2% | 17.1% → 20.6% | 15.1% → 0 |
| `SAG.PBM` | 15.9% → 15.7% | 32.1% → 37.9% | 21.0% → 21.3% | 5.6% → 0 |

No group absorbs the freed share disproportionately and the top-ranked
feature is unchanged in all four. For the attack classes, the clocks were
carrying 3-15% of a score that the remaining columns reconstruct without
reorganising themselves.

### On `benign_degradation`: the proxy moved rather than vanished

| Group | full | no-clocks |
|---|---:|---:|
| counters | 13.7% | **62.5%** |
| timing deltas | 25.2% | 24.1% |
| counter-deltas | 5.8% | 4.8% |
| absolute time | 47.6% | — |

`StNum` alone goes from 0.70 (6th) to **6.69**, nearly ten times, and becomes
the class's dominant feature. That is the same *kind* of quantity the clock
was: `StNum` increases monotonically within a run, so "what value of StNum is
this" is a statement about position in the execution rather than about
whether a message was dropped.

**But one thing does not transfer, and it is the thing that made the clocks
suspicious.** The absolute-time attribution was *unstable* across folds
(max/median **2.13-2.53**) - it moved with which runs were held out, which is
how a run identifier behaves. The new `StNum` concentration is
max/median **1.01**, the most stable entry in the whole table. Attribution
that does not move with the held-out set is not behaving like an identifier
of those runs; it is more consistent with a systematic difference - for
instance, the 85 benign-control runs occupying a different `StNum` range than
the attack runs by construction.

**So this experiment does not decide the question, and saying otherwise would
be reading the result we hoped for.** What it establishes is narrower and
still useful: removing the clocks is clean for every attack family, and for
`benign_degradation` it relocates the model's dependence onto a stable
position feature rather than eliminating dependence on position.

### The cheap experiment that would decide it, preregistered here

D.1 measured `no-counters` alone at **-0.0023 AP [-0.0051, +0.0004], which
does not separate** - the raw counters are nearly free on their own. The
combination `no-absolute-time` + `no-counters` was never run. It is one
training run (~25 min) plus one SHAP run (~2 h).

Written before running it:

- **If** detection holds (paired against the reference, no separation beyond
  the -0.0025 the clocks already cost) **and** `benign_degradation`'s
  attribution moves onto the *deltas* - `stDiff`, `sqDiff`, `timestampDiff`,
  which measure a **gap** rather than a **position** - then the detector can
  be reported as not depending on any position variable, and that is the
  configuration the paper should publish.
- **If** detection holds but the attribution concentrates on some third
  position-like feature, the honest conclusion is that this pool cannot
  separate position from signature for benign degradation, which is a
  limitation to state rather than a result to fix.
- **If** detection degrades materially, the clocks and counters were carrying
  signal after all, and §9's leakage reading was too strong.

The expected null, stated now: the four **attack** classes should be
essentially unchanged again, since `no-counters` alone did not separate and
the clocks were already shown to redistribute cleanly there. A surprise in
the attack classes would mean the two ablations interact, which neither D.1
nor this card predicts.

### Status

**Done**, and it narrows rather than closes the question. Evidence:
`results/f3-xgboost-shap-no-absolute-time`. The combined run above is the
open step.
