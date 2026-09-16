# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:55:54 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,556,388 | 11,106 | 154 | 0 | 1,054 | 1,605 | 10,570,307 |
| benign_degradation | 98,770 | 152,008 | 2,292 | 14,145 | 2,072 | 1,355 | 270,642 |
| SAG.DB | 6,055 | 133 | 42,762 | 0 | 1,794 | 35 | 50,779 |
| FRG | 16,380 | 2,598 | 437 | 42,151 | 179 | 2,218 | 63,963 |
| SAG.PB | 8,963 | 441 | 5,886 | 1 | 31,656 | 12 | 46,959 |
| SAG.PBM | 38,464 | 769 | 966 | 3,423 | 1,233 | 9,973 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 5,576 (26.0%) | 878 (4.1%) | 131 (0.6%) | 14,041 (65.5%) | 69 (0.3%) | 733 (3.4%) |
| DELAY | 89,980 | 34,538 (38.4%) | 55,426 (61.6%) | 3 (0.0%) | 7 (0.0%) | 1 (0.0%) | 5 (0.0%) |
| DUPLICATION | 13,519 | 459 (3.4%) | 13,060 (96.6%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,980 | 50,048 (55.6%) | 39,905 (44.3%) | 4 (0.0%) | 10 (0.0%) | 1 (0.0%) | 12 (0.0%) |
| LINK_FLAP | 20,205 | 2,387 (11.8%) | 15,587 (77.1%) | 1,039 (5.1%) | 12 (0.1%) | 1,012 (5.0%) | 168 (0.8%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 2,645 (11.4%) | 18,224 (78.6%) | 1,115 (4.8%) | 75 (0.3%) | 974 (4.2%) | 167 (0.7%) |
| REORDERING | 12,330 | 3,117 (25.3%) | 8,928 (72.4%) | 0 (0.0%) | 0 (0.0%) | 15 (0.1%) | 270 (2.2%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 0.02% | 0.03% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 0.11% | 1.15% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 69.88% | 73.98% |
| benign_degradation: DELAY | 89,980 | 0.02% | 61.62% |
| benign_degradation: DUPLICATION | 13,519 | 0.00% | 96.60% |
| benign_degradation: JITTER | 89,980 | 0.03% | 44.38% |
| benign_degradation: LINK_FLAP | 20,205 | 11.04% | 88.19% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 10.05% | 88.60% |
| benign_degradation: REORDERING | 12,330 | 2.31% | 74.72% |

> Highest attack_fpr: **CONGESTION_LOSS** (69.88%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

