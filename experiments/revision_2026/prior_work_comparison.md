# What Gray-GOOSE reuses and what it adds (Reviewer 3, comment 6)

Started 2026-09-23. The reviewer asks for "a compact comparison" that
identifies the reused attack rules, traces, generation code, labels and
features, what was newly generated, and which conclusions the earlier dataset
could not support — and for a contribution statement that separates the attack
model from the dataset extension and the evaluation tools.

Ref. 9 in the submitted manuscript is Gonçalves, Quincozes, Quincozes and
Kazienko, *Modelagem e Detecção de Ataques Grayhole ao Protocolo GOOSE usando
o Framework ERENO*, SBSeg 2023, pp. 417-430 (`gonccalves2023modelagem`).

Every cell is marked:

- **[code]** — ERENO's git history (`../ereno`), commit and author named.
- **[paper]** — stated in the submitted manuscript (`main.tex`, not versioned)
  or in the cited work's text.
- **[inferred]** — consistent with the code and the dates, not confirmed from
  Ref. 9's own text, which is not in this repository.

## 1. Lineage

| | ERENO base [8] | Ref. 9 (Gonçalves et al., 2023) | Zomer & Mundt, SBSeg 2025 (published after the submission; not cited) | Gray-GOOSE as submitted | Gray-GOOSE revision |
|---|---|---|---|---|---|
| **Attack** | injection, masquerade, replay, high-`stNum` (uc01-uc07) | uniform random grayhole | uniform random grayhole, discard 3-90% | FRG + SAG.DB / PB / PBM | same four, unchanged code |
| **Generation code** | [code] `sequincozes`, 2022-2024 | [inferred] `attacks/uc08/GrayHoleVictimCreator` — present since `669c1bc` (2023-06-03), before SBSeg 2023; Quincozes co-authors both | [paper] ERENO grayhole with a discard rate, Algorithm 1 | [code] `attacks/uc09/OrientedGrayHoleCreator` — SAG `d2aef13` (Zomer, 2025-06-25), FRG added `b6cbc88` (2025-10-24) | [code] same `uc09`, plus seeded RNG and run metadata `7e6eb7d` (2026-08-25) and benign impairments `37f5737` |
| **Drop rule** | — | [code, uc08] keep each frame with probability `selectionRate` | [paper] per-frame discard probability | [code, uc09] FRG: drop each frame with probability `discardRate`; SAG: triggered on `StNum` change (`threat_model.md` §2) | same |
| **Label** | — | [code, uc08] **every delivered frame** of the attack trace is labeled grayhole | [paper] "the first record after an attack is labeled attack" | [code, uc09] the **next delivered frame** after a drop | same; its limits in `label_duplication_audit.md` §7 |
| **Delta features** (`timestampDiff`, `sqDiff`, `stDiff`, ...) | [code] **in ERENO's writers since 2022** (`a6c7eb8`, 2022-07-25) | [inferred] available | [paper] used, described as ERENO's "enriched attributes" | [paper] the with/without-delta comparison is the central experiment | recomputed **within trace** after grouping (`prepare_grouped_dataset.py`) |
| **Classifiers** | — | [paper, per Reviewer 3] ML classifiers | [paper] XGBoost, Random Forest, Naive Bayes | [paper] XGBoost only | XGBoost, RF, DT, LR, six rules, nested tuning (card D) |
| **Validation** | — | not known here | [paper] K-fold, K = 5 | [paper] stratified 5-fold, **message-level** | StratifiedGroupKFold by run, leakage-audited (cards A-B) |
| **XAI** | — | none reported [paper, main.tex §II-B] | [paper] **SHAP** | [paper] SHAP on a model fit on all data | SHAP on held-out groups, verified refit, stability (card F) |
| **Traces** | — | its own | its own | one trace per class, no run identity (`data_card.md` §3) | 265 seeded runs, 11.06M rows, with benign controls |

## 2. What is new, and what is not

**New in Gray-GOOSE, and only here:**

1. **The SAG attack model** — dropping triggered by `StNum` transitions, three
   variants. No prior artifact in the lineage conditions a drop on protocol
   state: `uc08` and the 2025 paper both drop uniformly. [code, `d2aef13`; the
   2025 paper's text has no `stNum` trigger]
2. **The dataset with experimental units** (revision): seeded runs,
   `run_id`/`trace_id`/`event_id`, a preregistered matrix over loss and burst,
   85 benign-degradation runs. [code, `7e6eb7d`, `37f5737`]
3. **The evaluation**: grouped validation, benign controls, baselines,
   ablations, run-level statistics, held-out SHAP, and the protection
   consequence model (`protection_consequence.md`).

**Not new, and the submitted manuscript does not make this clear:**

- **The delta features are ERENO's**, present since 2022. The submitted
  paper's central experiment — XGBoost with and without them — measures an
  existing feature set, not a proposed one. The revision's contribution there
  is computing them **within trace** and measuring which of them carry the
  signal (D.1: the three timing deltas), not the features themselves.
- **FRG is a re-implementation, not Ref. 9's code.** It lives in `uc09`
  (2025-10-24), not in `uc08`, and it labels differently: Ref. 9's `uc08`
  labels every surviving frame of the attack trace, `uc09`'s FRG only the frame
  after each drop. Calling it "the FRG of Ref. 9" overstates the reuse; calling
  it "Ref. 9's attack rule, re-implemented with Gray-GOOSE's labeling" is
  accurate. [code; which code Ref. 9 actually ran is **inferred**]
- **SHAP on grayhole detection is not new to this group.** The 2025 SBSeg
  paper by two of the authors applied SHAP, XGBoost and Random Forest to
  ERENO grayhole traffic. The submitted manuscript states that "existing
  studies ... do not investigate which protocol or temporal features
  effectively distinguish selective forwarding" (§II-B) and does not cite it
  — the submission predates that paper's publication (§5), but the revision
  has to.

## 3. Conclusions the earlier data could not support

| Conclusion | Why Ref. 9's setup could not reach it |
|---|---|
| Detection of **state-triggered** loss, per variant | its attack has no state trigger |
| Separating attack loss from **benign** loss at matched rates — and the null for FRG ≡ congestion (`benign_controls.md` §8) | no benign-loss controls |
| Which drops **suppress a trip** — only SAG.DB by construction (`protection_consequence.md` §7) | uniform loss rarely removes a whole fault state (FRG ≤ 3%) |
| Any estimate that survives **grouped** validation | no run identity to group by |

## 4. For the contribution statement

The reviewer asks that the attack model, the dataset extension and the
evaluation tools be separated, and that XGBoost and SHAP not be presented as
methodological contributions. Consistent with the table:

1. **Attack model** — the SAG family (new).
2. **Dataset** — a seeded, run-identified Gray-GOOSE with benign controls
   (new), extending ERENO and re-implementing Ref. 9's uniform grayhole as the
   reference class.
3. **Evaluation** — a leakage-safe protocol with benign controls, baselines
   and a protection-consequence model. XGBoost and SHAP are the instruments,
   and the delta features are ERENO's.

## 5. Open — decisions for the authors

- **Cite the 2025 SBSeg paper — in the revision, yes.** It was not cited in
  the submitted manuscript because the IEEE submission predates its
  publication (authors, 2026-09-23), so the omission was chronological, not an
  oversight. It is published now, it is prior work by two of the authors on
  the same attack family with SHAP, and the revision's §II-B claim of an XAI
  gap cannot stand next to it. Cite it, and narrow the claim to what it did
  not do: state-triggered attacks, grouped validation, held-out SHAP.
- **Ref. 9's own text** — to confirm its code (`uc08`?), labels, features,
  classifiers and validation. Every [inferred] cell above waits on it.
