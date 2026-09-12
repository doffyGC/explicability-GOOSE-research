# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 16:35:09 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 20,339,735 | 35,397 | 7,229 | 202 | 3,017 | 344 | 20,385,924 |
| benign_degradation | 112,869 | 155,921 | 1,025 | 865 | 0 | 0 | 270,680 |
| SAG.DB | 17,078 | 0 | 15 | 0 | 1 | 0 | 17,094 |
| FRG | 21,421 | 2 | 12 | 1 | 0 | 0 | 21,436 |
| SAG.PB | 46,930 | 0 | 8 | 0 | 21 | 0 | 46,959 |
| SAG.PBM | 54,810 | 0 | 13 | 0 | 5 | 0 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 21,427 (100.0%) | 1 (0.0%) | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DELAY | 89,990 | 2,373 (2.6%) | 87,323 (97.0%) | 177 (0.2%) | 117 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,520 | 4,158 (30.8%) | 9,159 (67.7%) | 203 (1.5%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 34,445 (38.3%) | 54,258 (60.3%) | 620 (0.7%) | 667 (0.7%) | 0 (0.0%) | 0 (0.0%) |
| LINK_FLAP | 20,205 | 20,197 (100.0%) | 1 (0.0%) | 7 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 23,201 (100.0%) | 1 (0.0%) | 4 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| REORDERING | 12,333 | 7,068 (57.3%) | 5,178 (42.0%) | 6 (0.0%) | 81 (0.7%) | 0 (0.0%) | 0 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 0.04% | 0.04% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 0.18% | 1.64% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 0.04% | 0.04% |
| benign_degradation: DELAY | 89,990 | 0.33% | 97.36% |
| benign_degradation: DUPLICATION | 13,520 | 1.50% | 69.25% |
| benign_degradation: JITTER | 89,990 | 1.43% | 61.72% |
| benign_degradation: LINK_FLAP | 20,205 | 0.03% | 0.04% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 0.02% | 0.02% |
| benign_degradation: REORDERING | 12,333 | 0.71% | 42.69% |

> Highest attack_fpr: **DUPLICATION** (1.50%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

