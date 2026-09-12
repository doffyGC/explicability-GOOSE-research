# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 16:34:31 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,614,267 | 383,270 | 1,000,786 | 2,366,764 | 3,360,139 | 2,660,698 | 20,385,924 |
| benign_degradation | 15,481 | 179,831 | 11,590 | 46,717 | 403 | 16,658 | 270,680 |
| SAG.DB | 18 | 17 | 15,741 | 203 | 396 | 719 | 17,094 |
| FRG | 2,192 | 885 | 2,426 | 12,091 | 56 | 3,786 | 21,436 |
| SAG.PB | 1,170 | 251 | 8,877 | 1,025 | 30,016 | 5,620 | 46,959 |
| SAG.PBM | 693 | 697 | 12,648 | 1,580 | 5,833 | 33,377 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 2,075 (9.7%) | 973 (4.5%) | 2,438 (11.4%) | 12,107 (56.5%) | 59 (0.3%) | 3,784 (17.7%) |
| DELAY | 89,990 | 31 (0.0%) | 88,403 (98.2%) | 671 (0.7%) | 686 (0.8%) | 0 (0.0%) | 199 (0.2%) |
| DUPLICATION | 13,520 | 16 (0.1%) | 12,459 (92.2%) | 399 (3.0%) | 159 (1.2%) | 0 (0.0%) | 487 (3.6%) |
| JITTER | 89,990 | 83 (0.1%) | 68,400 (76.0%) | 1,944 (2.2%) | 19,053 (21.2%) | 0 (0.0%) | 510 (0.6%) |
| LINK_FLAP | 20,205 | 6,617 (32.7%) | 488 (2.4%) | 2,420 (12.0%) | 5,826 (28.8%) | 183 (0.9%) | 4,671 (23.1%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 6,263 (27.0%) | 572 (2.5%) | 2,868 (12.4%) | 8,369 (36.1%) | 144 (0.6%) | 4,990 (21.5%) |
| REORDERING | 12,333 | 396 (3.2%) | 8,536 (69.2%) | 850 (6.9%) | 517 (4.2%) | 17 (0.1%) | 2,017 (16.4%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 43.46% | 44.60% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 65.44% | 72.81% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 85.78% | 90.32% |
| benign_degradation: DELAY | 89,990 | 1.73% | 99.97% |
| benign_degradation: DUPLICATION | 13,520 | 7.73% | 99.88% |
| benign_degradation: JITTER | 89,990 | 23.90% | 99.91% |
| benign_degradation: LINK_FLAP | 20,205 | 64.84% | 67.25% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 70.55% | 73.01% |
| benign_degradation: REORDERING | 12,333 | 27.58% | 96.79% |

> Highest attack_fpr: **CONGESTION_LOSS** (85.78%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

