# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:56:39 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `random-forest`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,550,340 | 6,951 | 1,246 | 2,384 | 2,225 | 7,161 | 10,570,307 |
| benign_degradation | 85,459 | 165,350 | 1,596 | 14,840 | 1,532 | 1,865 | 270,642 |
| SAG.DB | 4,001 | 624 | 42,728 | 165 | 2,983 | 278 | 50,779 |
| FRG | 9,982 | 11,577 | 302 | 38,085 | 223 | 3,794 | 63,963 |
| SAG.PB | 5,092 | 1,200 | 5,774 | 126 | 34,260 | 507 | 46,959 |
| SAG.PBM | 25,968 | 1,815 | 548 | 3,833 | 925 | 21,739 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 3,502 (16.3%) | 2,469 (11.5%) | 42 (0.2%) | 14,227 (66.4%) | 55 (0.3%) | 1,133 (5.3%) |
| DELAY | 89,980 | 28,503 (31.7%) | 61,469 (68.3%) | 0 (0.0%) | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,519 | 5 (0.0%) | 13,514 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,980 | 45,354 (50.4%) | 44,614 (49.6%) | 1 (0.0%) | 10 (0.0%) | 0 (0.0%) | 1 (0.0%) |
| LINK_FLAP | 20,205 | 2,198 (10.9%) | 15,908 (78.7%) | 752 (3.7%) | 217 (1.1%) | 777 (3.8%) | 353 (1.7%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 2,193 (9.5%) | 18,750 (80.8%) | 801 (3.5%) | 378 (1.6%) | 700 (3.0%) | 378 (1.6%) |
| REORDERING | 12,330 | 3,704 (30.0%) | 8,626 (70.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 0.12% | 0.14% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 0.16% | 0.67% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 72.13% | 83.66% |
| benign_degradation: DELAY | 89,980 | 0.01% | 68.32% |
| benign_degradation: DUPLICATION | 13,519 | 0.00% | 99.96% |
| benign_degradation: JITTER | 89,980 | 0.01% | 49.60% |
| benign_degradation: LINK_FLAP | 20,205 | 10.39% | 89.12% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 9.73% | 90.55% |
| benign_degradation: REORDERING | 12,330 | 0.00% | 69.96% |

> Highest attack_fpr: **CONGESTION_LOSS** (72.13%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

