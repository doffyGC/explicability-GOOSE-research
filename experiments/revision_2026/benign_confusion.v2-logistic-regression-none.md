# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:57:09 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `logistic-regression`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,557,096 | 2,366 | 8,526 | 5 | 2,280 | 34 | 10,570,307 |
| benign_degradation | 223,164 | 38,235 | 7,205 | 82 | 1,321 | 635 | 270,642 |
| SAG.DB | 14,495 | 3,085 | 30,230 | 1 | 2,953 | 15 | 50,779 |
| FRG | 41,412 | 21,724 | 774 | 0 | 17 | 36 | 63,963 |
| SAG.PB | 23,058 | 3,460 | 5,958 | 0 | 14,238 | 245 | 46,959 |
| SAG.PBM | 50,955 | 2,730 | 1,058 | 0 | 48 | 37 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 13,946 (65.1%) | 7,228 (33.7%) | 238 (1.1%) | 0 (0.0%) | 6 (0.0%) | 10 (0.0%) |
| DELAY | 89,980 | 89,161 (99.1%) | 584 (0.6%) | 235 (0.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,519 | 13,519 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,980 | 89,545 (99.5%) | 147 (0.2%) | 288 (0.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| LINK_FLAP | 20,205 | 2,782 (13.8%) | 13,668 (67.6%) | 2,773 (13.7%) | 5 (0.0%) | 643 (3.2%) | 334 (1.7%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 2,494 (10.8%) | 16,095 (69.4%) | 3,574 (15.4%) | 77 (0.3%) | 672 (2.9%) | 288 (1.2%) |
| REORDERING | 12,330 | 11,717 (95.0%) | 513 (4.2%) | 97 (0.8%) | 0 (0.0%) | 0 (0.0%) | 3 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 0.08% | 0.09% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 0.35% | 0.50% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 1.19% | 34.92% |
| benign_degradation: DELAY | 89,980 | 0.26% | 0.91% |
| benign_degradation: DUPLICATION | 13,519 | 0.00% | 0.00% |
| benign_degradation: JITTER | 89,980 | 0.32% | 0.48% |
| benign_degradation: LINK_FLAP | 20,205 | 18.58% | 86.23% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 19.88% | 89.25% |
| benign_degradation: REORDERING | 12,330 | 0.81% | 4.97% |

> Highest attack_fpr: **QUEUE_OVERLOAD_BURST** (19.88%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

