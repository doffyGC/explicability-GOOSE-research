# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-13 06:58:06 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `xgboost`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 12,764,684 | 479,573 | 1,154,539 | 3,090,082 | 2,521,997 | 2,728,430 | 22,739,305 |
| benign_degradation | 15,382 | 187,195 | 8,837 | 45,896 | 1,828 | 11,542 | 270,680 |
| SAG.DB | 19 | 46 | 45,826 | 1,048 | 1,534 | 2,309 | 50,782 |
| FRG | 6,585 | 1,292 | 7,104 | 40,252 | 744 | 7,999 | 63,976 |
| SAG.PB | 1,024 | 31 | 8,481 | 1,314 | 31,768 | 4,341 | 46,959 |
| SAG.PBM | 1,123 | 78 | 12,424 | 3,390 | 6,472 | 31,341 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 1,341 (6.3%) | 278 (1.3%) | 2,407 (11.2%) | 14,035 (65.5%) | 236 (1.1%) | 3,139 (14.6%) |
| DELAY | 89,990 | 106 (0.1%) | 89,335 (99.3%) | 77 (0.1%) | 413 (0.5%) | 7 (0.0%) | 52 (0.1%) |
| DUPLICATION | 13,520 | 9 (0.1%) | 13,505 (99.9%) | 5 (0.0%) | 0 (0.0%) | 1 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 970 (1.1%) | 71,080 (79.0%) | 1,161 (1.3%) | 16,512 (18.3%) | 85 (0.1%) | 182 (0.2%) |
| LINK_FLAP | 20,205 | 6,617 (32.7%) | 304 (1.5%) | 2,348 (11.6%) | 6,268 (31.0%) | 831 (4.1%) | 3,837 (19.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 6,329 (27.3%) | 547 (2.4%) | 2,773 (11.9%) | 8,653 (37.3%) | 668 (2.9%) | 4,236 (18.3%) |
| REORDERING | 12,333 | 10 (0.1%) | 12,146 (98.5%) | 66 (0.5%) | 15 (0.1%) | 0 (0.0%) | 96 (0.8%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 20,333,504 | 39.45% | 40.75% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 61.24% | 70.23% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 92.45% | 93.74% |
| benign_degradation: DELAY | 89,990 | 0.61% | 99.88% |
| benign_degradation: DUPLICATION | 13,520 | 0.04% | 99.93% |
| benign_degradation: JITTER | 89,990 | 19.94% | 98.92% |
| benign_degradation: LINK_FLAP | 20,205 | 65.75% | 67.25% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 70.37% | 72.73% |
| benign_degradation: REORDERING | 12,333 | 1.44% | 99.92% |

> Highest attack_fpr: **CONGESTION_LOSS** (92.45%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

