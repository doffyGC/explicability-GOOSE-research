# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:56:54 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `random-forest`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 8,489,822 | 1,155,520 | 24,655 | 150,904 | 116,125 | 633,281 | 10,570,307 |
| benign_degradation | 6,355 | 229,471 | 2,004 | 25,065 | 2,603 | 5,144 | 270,642 |
| SAG.DB | 44 | 154 | 45,510 | 325 | 4,147 | 599 | 50,779 |
| FRG | 610 | 4,122 | 675 | 49,982 | 453 | 8,121 | 63,963 |
| SAG.PB | 321 | 166 | 5,810 | 189 | 39,186 | 1,287 | 46,959 |
| SAG.PBM | 1,601 | 597 | 969 | 4,871 | 2,045 | 44,745 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 116 (0.5%) | 378 (1.8%) | 84 (0.4%) | 18,997 (88.7%) | 86 (0.4%) | 1,767 (8.2%) |
| DELAY | 89,980 | 1,962 (2.2%) | 87,523 (97.3%) | 1 (0.0%) | 481 (0.5%) | 3 (0.0%) | 10 (0.0%) |
| DUPLICATION | 13,519 | 1 (0.0%) | 13,518 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,980 | 3,840 (4.3%) | 85,267 (94.8%) | 16 (0.0%) | 784 (0.9%) | 16 (0.0%) | 57 (0.1%) |
| LINK_FLAP | 20,205 | 239 (1.2%) | 14,467 (71.6%) | 864 (4.3%) | 1,810 (9.0%) | 1,261 (6.2%) | 1,564 (7.7%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 166 (0.7%) | 16,223 (69.9%) | 986 (4.2%) | 2,976 (12.8%) | 1,218 (5.2%) | 1,631 (7.0%) |
| REORDERING | 12,330 | 31 (0.3%) | 12,095 (98.1%) | 53 (0.4%) | 17 (0.1%) | 19 (0.2%) | 115 (0.9%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 8.16% | 17.50% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 14.89% | 42.51% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 97.69% | 99.46% |
| benign_degradation: DELAY | 89,980 | 0.55% | 97.82% |
| benign_degradation: DUPLICATION | 13,519 | 0.00% | 99.99% |
| benign_degradation: JITTER | 89,980 | 0.97% | 95.73% |
| benign_degradation: LINK_FLAP | 20,205 | 27.22% | 98.82% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 29.36% | 99.28% |
| benign_degradation: REORDERING | 12,330 | 1.65% | 99.75% |

> Highest attack_fpr: **CONGESTION_LOSS** (97.69%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

