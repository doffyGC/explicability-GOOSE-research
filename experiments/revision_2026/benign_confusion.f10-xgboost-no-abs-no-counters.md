# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-23 23:01:11 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `xgboost`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,557,348 | 11,660 | 0 | 95 | 1,086 | 118 | 10,570,307 |
| benign_degradation | 97,149 | 151,293 | 1,720 | 16,509 | 2,283 | 1,688 | 270,642 |
| SAG.DB | 4,165 | 31 | 45,038 | 1 | 1,542 | 2 | 50,779 |
| FRG | 13,123 | 6,582 | 338 | 39,617 | 604 | 3,699 | 63,963 |
| SAG.PB | 5,984 | 298 | 8,418 | 84 | 32,172 | 3 | 46,959 |
| SAG.PBM | 33,630 | 1,483 | 958 | 4,928 | 1,436 | 12,393 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 4,444 (20.7%) | 2,085 (9.7%) | 94 (0.4%) | 13,399 (62.5%) | 209 (1.0%) | 1,197 (5.6%) |
| DELAY | 89,980 | 33,236 (36.9%) | 56,739 (63.1%) | 0 (0.0%) | 3 (0.0%) | 0 (0.0%) | 2 (0.0%) |
| DUPLICATION | 13,519 | 5 (0.0%) | 13,514 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,980 | 49,722 (55.3%) | 40,257 (44.7%) | 1 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| LINK_FLAP | 20,205 | 2,567 (12.7%) | 14,653 (72.5%) | 770 (3.8%) | 932 (4.6%) | 1,027 (5.1%) | 256 (1.3%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 2,538 (10.9%) | 16,371 (70.6%) | 854 (3.7%) | 2,174 (9.4%) | 1,047 (4.5%) | 216 (0.9%) |
| REORDERING | 12,330 | 4,637 (37.6%) | 7,674 (62.2%) | 1 (0.0%) | 1 (0.0%) | 0 (0.0%) | 17 (0.1%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 0.01% | 0.05% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 0.03% | 0.84% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 69.53% | 79.26% |
| benign_degradation: DELAY | 89,980 | 0.01% | 63.06% |
| benign_degradation: DUPLICATION | 13,519 | 0.00% | 99.96% |
| benign_degradation: JITTER | 89,980 | 0.00% | 44.74% |
| benign_degradation: LINK_FLAP | 20,205 | 14.77% | 87.30% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 18.50% | 89.06% |
| benign_degradation: REORDERING | 12,330 | 0.15% | 62.39% |

> Highest attack_fpr: **CONGESTION_LOSS** (69.53%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

