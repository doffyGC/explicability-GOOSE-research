# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:57:24 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `logistic-regression`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 7,391,226 | 1,069,383 | 41,639 | 724,364 | 250,750 | 1,092,945 | 10,570,307 |
| benign_degradation | 21,302 | 127,683 | 11,617 | 79,224 | 4,556 | 26,260 | 270,642 |
| SAG.DB | 0 | 45 | 47,052 | 174 | 3,507 | 1 | 50,779 |
| FRG | 2,091 | 22,881 | 1,425 | 24,107 | 197 | 13,262 | 63,963 |
| SAG.PB | 518 | 390 | 9,077 | 738 | 33,137 | 3,099 | 46,959 |
| SAG.PBM | 1,242 | 3,447 | 3,133 | 3,049 | 2,675 | 41,282 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 715 (3.3%) | 7,661 (35.8%) | 445 (2.1%) | 7,978 (37.2%) | 76 (0.4%) | 4,553 (21.2%) |
| DELAY | 89,980 | 3,309 (3.7%) | 57,232 (63.6%) | 1,225 (1.4%) | 23,585 (26.2%) | 417 (0.5%) | 4,212 (4.7%) |
| DUPLICATION | 13,519 | 11,430 (84.5%) | 419 (3.1%) | 0 (0.0%) | 93 (0.7%) | 0 (0.0%) | 1,577 (11.7%) |
| JITTER | 89,980 | 2,805 (3.1%) | 51,747 (57.5%) | 1,294 (1.4%) | 25,414 (28.2%) | 165 (0.2%) | 8,555 (9.5%) |
| LINK_FLAP | 20,205 | 384 (1.9%) | 2,039 (10.1%) | 3,759 (18.6%) | 9,745 (48.2%) | 1,929 (9.5%) | 2,349 (11.6%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 291 (1.3%) | 2,778 (12.0%) | 4,726 (20.4%) | 11,136 (48.0%) | 1,946 (8.4%) | 2,323 (10.0%) |
| REORDERING | 12,330 | 2,368 (19.2%) | 5,807 (47.1%) | 168 (1.4%) | 1,273 (10.3%) | 23 (0.2%) | 2,691 (21.8%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 18.13% | 26.81% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 39.13% | 64.33% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 60.91% | 96.66% |
| benign_degradation: DELAY | 89,980 | 32.72% | 96.32% |
| benign_degradation: DUPLICATION | 13,519 | 12.35% | 15.45% |
| benign_degradation: JITTER | 89,980 | 39.37% | 96.88% |
| benign_degradation: LINK_FLAP | 20,205 | 88.01% | 98.10% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 86.77% | 98.75% |
| benign_degradation: REORDERING | 12,330 | 33.70% | 80.79% |

> Highest attack_fpr: **LINK_FLAP** (88.01%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

