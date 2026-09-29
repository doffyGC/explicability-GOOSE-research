# Response matrix: every review comment, its evidence, what is left

Started 2026-09-23. One row per comment of the TSG-00914-2026 decision
(editor, Reviewers 1-3), paraphrased. The review text itself is kept out of
the repository. This is the traceability table the response letter is written
from, and the list of what the rewrite (card H) still owes.

Every number below comes from a versioned artifact on the corrected pool
(SHA-256 `3109e4d4…`, 265 runs), named in the row.

**The published detector** is `f10-xgboost-no-abs-no-counters`: 35 features,
no absolute clocks and no raw `StNum`/`SqNum` (`ablations_baselines.md` §20).
Rows marked *(v2)* were measured on the 40-feature model. The D.3/E/D.5 batch
was run on 2026-09-29 (`ablations_baselines.md` §21); what is still on `v2`
only is D.4.

Status: ✅ evidence in hand · 🟡 partial, or measured on v2 only · ❌ not done ·
✍️ text only, no experiment needed.

## Editor

| # | The comment | Status | Evidence | Left to do |
|---|---|---|---|---|
| E.1 | No justification against IEC 62351-6 | 🟡 ✍️ | `threat_model.md` §3: authentication does not stop a grayhole — it never forges a frame, and `StNum` stays in clear | Citations; write it into Introduction and Related Work (H) |
| E.2 | AI detection without its limits or a defence-in-depth position | 🟡 ✍️ | `threat_model.md` §3 (PRP/HSR, `StNum` monitoring, 62351-6); D.2's rule wins at a 0.1% budget (§20) — a limit to state | Write the layered-defence paragraph (H) |
| E.3 | No ablation | ✅ | D.1 (`ablations_baselines.md` §14-15): without the 9 deltas AP 0.8329 → 0.1101; timing deltas −0.0645; 29 of 40 columns free *(v2)*. Identifier ablations −0.0025 / −0.0085 (§20) | Present on `f10` as the reference row |
| E.4 | Unclear dataset validation | 🟡 | `data_card.md`, `metadata_audit.md`, `label_duplication_audit.md`, `check_no_leakage.py` (265/265), `threat_model.md` §1 (generator read from code) | Realism is not validated — see R1.4 |
| E.5 | Class imbalance unaddressed | ✅ | §21 on `f10`: `smote` **does not separate** (+0.0024 AP [−0.0004, +0.0055]), `downsample` separates the wrong way (−0.0104). `none` vs `downsample` remain one score ~213x apart in threshold | — |

## Reviewer 1

| # | The comment | Status | Evidence | Left to do |
|---|---|---|---|---|
| R1.1 | Discuss that IEC 62351-6 mitigates many of these attacks; position novelty against it | 🟡 ✍️ | as E.1; authentication closes injection/masquerade/replay (ERENO uc01-uc07), not selective suppression | as E.1 |
| R1.2 | AI as a complementary layer; explicit defence-in-depth | 🟡 ✍️ | as E.2 | as E.2 |
| R1.3 | Ablation per feature group | ✅ | as E.3 | — |
| R1.4 | How ERENO generated, validated and made the data representative; traffic diversity; injection method | 🟡 | Generation: `data_card.md` §§3-5, `threat_model.md` §1 (event model, retransmission profile, offline attacker). Diversity: 5 seeds × loss × burst, 7 benign mechanisms | **One substation configuration and one traffic rate** — cannot be fixed without regenerating; state it as a limitation. SILVIO waveform provenance unknown |
| R1.5 | Consider SMOTE or another balancing strategy | ✅ | as E.5; `results/f20-xgboost-smote`, capped partial oversample (5 classes to 200k, `normal` untouched) | — |
| R1.6 | Compare with and without SMOTE | ✅ | Paired at the natural prior (`pr_curves_f20.md`, `--prior auto`): +0.0024 AP, interval straddles zero. Reported as a null | — |

## Reviewer 2

| # | The comment | Status | Evidence | Left to do |
|---|---|---|---|---|
| R2.0 | Novelty is application-oriented, not methodological | ✍️ | `prior_work_comparison.md` §4: contributions split into attack model / dataset / evaluation; XGBoost and SHAP as instruments | Rewrite the contribution list (H) |
| R2.1 | How were the folds built — message level or grouped? Optimism for deltas? | ✅ | Submitted: message-level stratified 5-fold (acknowledge). Revision: StratifiedGroupKFold by run, deltas recomputed within trace, leakage audit (cards A-B); the pool correction that followed (`label_duplication_audit.md`) | Methodology rewrite (H) |
| R2.2 | Downsampling before CV? Metrics on which distribution? FPR and alert burden | ✅ | Test folds always on the original distribution; alert budgets from `grouped_pr_curves.py`. On `f10`: **0.01% attack FPR and 0.05% alert rate on ideal `normal`** (`benign_confusion.f10-xgboost-no-abs-no-counters.md` §3); recall 0.4804 at a 1% budget (`pr_curves_f10.md`) | Acknowledge the submitted paper's downsampled-then-shuffled pipeline |

## Reviewer 3

| # | The comment | Status | Evidence | Left to do |
|---|---|---|---|---|
| R3.1a | Message-level split leaks; define runs/events, group, re-run | ✅ | as R2.1; `event_id ⊂ trace_id ⊂ run_id` (`data_card.md` §4) | — |
| R3.1b | Deltas computed within trace, after grouping | ✅ | `prepare_grouped_dataset.py`, `preparation_audit.json` | — |
| R3.1c | Ablation without absolute time and other identifiers | ✅ | `ablations_baselines.md` §20: `d1` −0.0025, `f10` −0.0085 AP, 1% recall 0.4833 → 0.4804; card F §§10-11 show why (the model used the clocks, then `StNum`, as position) | — |
| R3.2a | 1,006,989 vs 1,009,086 | ✅ | `data_card.md` §2: 1,009,086 is a typo; the McNemar table closes exactly on 1,006,989 (n01 − n10 = 125,438) | State it in the letter |
| R3.2b | Messages are not independent; report across runs | ✅ | Run-level bootstrap (`bootstrap_run_intervals.py`) and paired run-level comparisons (`grouped_pr_curves.py`) replace McNemar | — |
| R3.3a | Benign conditions matched to the attacks | 🟡 | 85 runs, 7 mechanisms (`benign_controls.md`). Matching is by name and burst length, not by frames lost (`protection_consequence.md` §7 caveat) | State the matching limit |
| R3.3b | How often each benign condition is confused with each SAG | ✅ | `benign_confusion.f10-xgboost-no-abs-no-counters.md` §§2-3, on `f10`: congestion loss 69.5% attack FPR, 62.5% as `FRG` (FRG ≡ congestion by construction, `benign_controls.md` §8); queue overload 18.5%, link flap 14.8%; delay, jitter, duplication, reordering ≤ 0.15%. Unchanged from v2 within 4 points | — |
| R3.3c | Which features remain informative under benign disturbance | 🟡 | D.1 (timing deltas carry it) *(v2)*; card F §11: on `f10`, `benign_degradation` leans on `delay` (44.8%), untested | Say it is an attribution, not a test |
| R3.4 | Operational assumptions per variant; VLAN, redundancy, 62351, monitoring; one table | ✅ | `threat_model.md` §§2-3 | Citations |
| R3.5 | Link loss to a protection function; else claims as hypotheses | 🟡 | `protection_consequence.md` §7: SAG.DB suppresses every trip (6.43x fault energy under a 400 ms backup); only SAG.DB is worse than every benign control. **Preliminary model, not HIL** | Citations for the three constants; hypothesis wording (card G.5); HIL out of scope — say so |
| R3.6 | Relationship to Ref. 9: reused vs new | ✅ | `prior_work_comparison.md` | Ref. 9's PDF to confirm the inferred cells; cite SBSeg 2025 |
| R3.7a | Rule / sequence-gap baseline | ✅ | D.2: best rule AP 0.2411, −0.5832 behind `f10`; `stnum-gap` wins at 0.1% budget (§20) | — |
| R3.7b | Classifiers from prior GOOSE/ERENO studies | ✅ | DT, RF, LR, XGBoost on the published 35 features (§21): XGBoost separates from all three; RF is the closest at −0.0327 AP | — |
| R3.7c | A temporal model | ❌ | deferred in card D scope (`ablations_baselines.md` §6) | Decide: implement, or justify the omission |
| R3.7d | Tuning inside grouped training data | 🟡 | D.4 nested tuning: −0.0001 AP, defaults stand *(v2)* | **Open decision, now the only one left on `v2`**: re-run on `f10` (~5 h) or report as validated on v2 |
| R3.7e | Per-class P/R and confusion matrices | ✅ | D.5 `cross_run_comparison.md`, **30 runs** incl. `f10` and the five `f20-*` (§21) | Cosmetic: `cross_run_report.py` files the `f20-*` runs under card D.1 |
| R3.7f | Generalisation across event types, substation configs, traffic rates, burst, loss, noise | ❌ | LOETO implemented but open-set (blocked by default); one substation config and one traffic rate exist | Leave-one-burst/loss-out is feasible on the pool; configs and rates are not — limitation |
| R3.8a | SHAP on grouped held-out runs; background, output scale, sample size, multiclass | ✅ | `explainability_card.md` §§1-9 | — |
| R3.8b | Stability across runs, not one global ranking | ✅ | `explainability_card.md` §9 (max/median across folds) and §§10-11 | — |
| R3.8c | Association, not causality; limit "inherently difficult" | ✍️ | `explainability_card.md` §7 fixes the vocabulary; `protection_consequence.md` §7: the hardest-to-detect variant is not the most harmful | Rewrite (H) |
| R3.8d | Release run/trace IDs, scripts, seeds, labels, preprocessing order, exact split | 🟡 | All in the repo (`splits_grouped.json`, sidecars, scripts) | Publish the corrected pool (Kaggle); **no dataset license declared** (`data_card.md` §2) |

## What is left, by kind

**Compute:** done 2026-09-29 — D.3 families and E (`downsample`, `smote`) on 35
features, D.5 regenerated (`ablations_baselines.md` §21). Only D.4 on `f10`
is left, and it is a decision rather than a queued run.

**Decisions:** temporal model (R3.7c); leave-one-burst/loss-out (R3.7f); D.4.

**Needed from the authors:** Ref. 9's PDF; SILVIO waveform provenance;
citations (IEC 61850-5 transfer times — `leon2019real` is already cited for
~3 ms type 1A — IEC 62351-6, IEC 62439-3, relay/breaker/zone-2 times); a
dataset license.

**Text (card H):** Introduction and Related Work (62351-6, defence in depth,
SBSeg 2025), contribution list, Methodology, Threat Model table, the
consequence section, limitations (one substation, one traffic rate, offline
attacker, per-message label, preliminary protection model).
