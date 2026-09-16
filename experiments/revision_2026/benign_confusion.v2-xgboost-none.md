# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:55:24 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `xgboost`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,561,353 | 7,601 | 65 | 118 | 914 | 256 | 10,570,307 |
| benign_degradation | 92,119 | 156,104 | 1,723 | 16,731 | 2,031 | 1,934 | 270,642 |
| SAG.DB | 4,162 | 66 | 44,828 | 4 | 1,715 | 4 | 50,779 |
| FRG | 12,327 | 5,157 | 338 | 41,761 | 448 | 3,932 | 63,963 |
| SAG.PB | 5,953 | 458 | 6,200 | 124 | 34,148 | 76 | 46,959 |
| SAG.PBM | 31,942 | 1,525 | 868 | 4,213 | 1,361 | 14,919 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 4,171 (19.5%) | 1,543 (7.2%) | 95 (0.4%) | 14,119 (65.9%) | 160 (0.7%) | 1,340 (6.3%) |
| DELAY | 89,980 | 32,032 (35.6%) | 57,948 (64.4%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,519 | 15 (0.1%) | 13,504 (99.9%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,980 | 48,948 (54.4%) | 41,027 (45.6%) | 1 (0.0%) | 1 (0.0%) | 0 (0.0%) | 3 (0.0%) |
| LINK_FLAP | 20,205 | 2,544 (12.6%) | 14,740 (73.0%) | 760 (3.8%) | 959 (4.7%) | 937 (4.6%) | 265 (1.3%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 2,600 (11.2%) | 16,832 (72.6%) | 867 (3.7%) | 1,651 (7.1%) | 933 (4.0%) | 317 (1.4%) |
| REORDERING | 12,330 | 1,809 (14.7%) | 10,510 (85.2%) | 0 (0.0%) | 1 (0.0%) | 1 (0.0%) | 9 (0.1%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 0.01% | 0.04% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 0.03% | 0.60% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 73.33% | 80.53% |
| benign_degradation: DELAY | 89,980 | 0.00% | 64.40% |
| benign_degradation: DUPLICATION | 13,519 | 0.00% | 99.89% |
| benign_degradation: JITTER | 89,980 | 0.01% | 45.60% |
| benign_degradation: LINK_FLAP | 20,205 | 14.46% | 87.41% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 16.24% | 88.79% |
| benign_degradation: REORDERING | 12,330 | 0.09% | 85.33% |

> Highest attack_fpr: **CONGESTION_LOSS** (73.33%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

