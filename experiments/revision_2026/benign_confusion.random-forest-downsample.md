# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 20:53:41 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `random-forest`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 11,424,592 | 872,811 | 775,934 | 2,805,085 | 2,504,094 | 2,003,408 | 20,385,924 |
| benign_degradation | 12,883 | 189,360 | 6,744 | 50,596 | 2,397 | 8,700 | 270,680 |
| SAG.DB | 490 | 434 | 8,485 | 1,424 | 2,379 | 3,882 | 17,094 |
| FRG | 2,020 | 2,227 | 1,589 | 12,759 | 526 | 2,315 | 21,436 |
| SAG.PB | 3,890 | 377 | 6,451 | 1,589 | 28,688 | 5,964 | 46,959 |
| SAG.PBM | 3,835 | 1,018 | 8,859 | 5,385 | 7,724 | 28,007 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 999 (4.7%) | 615 (2.9%) | 863 (4.0%) | 17,589 (82.1%) | 277 (1.3%) | 1,093 (5.1%) |
| DELAY | 89,990 | 19 (0.0%) | 89,737 (99.7%) | 17 (0.0%) | 195 (0.2%) | 3 (0.0%) | 19 (0.0%) |
| DUPLICATION | 13,520 | 486 (3.6%) | 12,205 (90.3%) | 165 (1.2%) | 327 (2.4%) | 140 (1.0%) | 197 (1.5%) |
| JITTER | 89,990 | 837 (0.9%) | 73,252 (81.4%) | 906 (1.0%) | 14,530 (16.1%) | 137 (0.2%) | 328 (0.4%) |
| LINK_FLAP | 20,205 | 4,995 (24.7%) | 1,544 (7.6%) | 2,238 (11.1%) | 7,482 (37.0%) | 882 (4.4%) | 3,064 (15.2%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 4,708 (20.3%) | 2,144 (9.2%) | 2,310 (10.0%) | 9,615 (41.4%) | 874 (3.8%) | 3,555 (15.3%) |
| REORDERING | 12,333 | 839 (6.8%) | 9,863 (80.0%) | 245 (2.0%) | 858 (7.0%) | 84 (0.7%) | 444 (3.6%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 36.67% | 39.74% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 62.13% | 75.47% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 92.47% | 95.34% |
| benign_degradation: DELAY | 89,990 | 0.26% | 99.98% |
| benign_degradation: DUPLICATION | 13,520 | 6.13% | 96.41% |
| benign_degradation: JITTER | 89,990 | 17.67% | 99.07% |
| benign_degradation: LINK_FLAP | 20,205 | 67.64% | 75.28% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 70.47% | 79.71% |
| benign_degradation: REORDERING | 12,333 | 13.22% | 93.20% |

> Highest attack_fpr: **CONGESTION_LOSS** (92.47%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

