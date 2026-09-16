# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:56:25 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,556,512 | 10,885 | 96 | 0 | 1,217 | 1,597 | 10,570,307 |
| benign_degradation | 98,918 | 151,959 | 2,285 | 14,093 | 2,015 | 1,372 | 270,642 |
| SAG.DB | 6,190 | 128 | 42,504 | 0 | 1,921 | 36 | 50,779 |
| FRG | 16,387 | 2,815 | 436 | 41,778 | 177 | 2,370 | 63,963 |
| SAG.PB | 9,443 | 444 | 5,783 | 1 | 31,287 | 1 | 46,959 |
| SAG.PBM | 38,482 | 760 | 970 | 3,187 | 1,204 | 10,225 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 5,576 (26.0%) | 920 (4.3%) | 131 (0.6%) | 13,984 (65.3%) | 69 (0.3%) | 748 (3.5%) |
| DELAY | 89,980 | 34,617 (38.5%) | 55,350 (61.5%) | 3 (0.0%) | 5 (0.0%) | 1 (0.0%) | 4 (0.0%) |
| DUPLICATION | 13,519 | 558 (4.1%) | 12,961 (95.9%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,980 | 49,964 (55.5%) | 39,993 (44.4%) | 5 (0.0%) | 7 (0.0%) | 2 (0.0%) | 9 (0.0%) |
| LINK_FLAP | 20,205 | 2,386 (11.8%) | 15,586 (77.1%) | 1,039 (5.1%) | 14 (0.1%) | 1,010 (5.0%) | 170 (0.8%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 2,703 (11.7%) | 18,218 (78.5%) | 1,107 (4.8%) | 83 (0.4%) | 918 (4.0%) | 171 (0.7%) |
| REORDERING | 12,330 | 3,114 (25.3%) | 8,931 (72.4%) | 0 (0.0%) | 0 (0.0%) | 15 (0.1%) | 270 (2.2%) |

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
| benign_degradation: CONGESTION_LOSS | 21,428 | 69.68% | 73.98% |
| benign_degradation: DELAY | 89,980 | 0.01% | 61.53% |
| benign_degradation: DUPLICATION | 13,519 | 0.00% | 95.87% |
| benign_degradation: JITTER | 89,980 | 0.03% | 44.47% |
| benign_degradation: LINK_FLAP | 20,205 | 11.05% | 88.19% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 9.82% | 88.35% |
| benign_degradation: REORDERING | 12,330 | 2.31% | 74.74% |

> Highest attack_fpr: **CONGESTION_LOSS** (69.68%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

