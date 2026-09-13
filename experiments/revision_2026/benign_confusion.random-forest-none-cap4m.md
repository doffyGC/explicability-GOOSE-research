# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-13 06:59:56 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `random-forest`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 22,658,309 | 34,488 | 15,114 | 12,770 | 8,172 | 10,452 | 22,739,305 |
| benign_degradation | 109,823 | 159,375 | 151 | 1,054 | 68 | 209 | 270,680 |
| SAG.DB | 49,562 | 194 | 590 | 317 | 47 | 72 | 50,782 |
| FRG | 59,970 | 1,002 | 294 | 2,596 | 28 | 86 | 63,976 |
| SAG.PB | 45,016 | 54 | 46 | 23 | 1,675 | 145 | 46,959 |
| SAG.PBM | 51,272 | 164 | 83 | 63 | 150 | 3,096 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 20,308 (94.7%) | 60 (0.3%) | 5 (0.0%) | 1,001 (4.7%) | 17 (0.1%) | 45 (0.2%) |
| DELAY | 89,990 | 4,160 (4.6%) | 85,830 (95.4%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,520 | 2,197 (16.2%) | 11,322 (83.7%) | 1 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 35,225 (39.1%) | 54,756 (60.8%) | 1 (0.0%) | 7 (0.0%) | 0 (0.0%) | 1 (0.0%) |
| LINK_FLAP | 20,205 | 19,917 (98.6%) | 41 (0.2%) | 128 (0.6%) | 13 (0.1%) | 30 (0.1%) | 76 (0.4%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 22,969 (99.0%) | 81 (0.3%) | 16 (0.1%) | 33 (0.1%) | 21 (0.1%) | 86 (0.4%) |
| REORDERING | 12,333 | 5,047 (40.9%) | 7,285 (59.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 20,333,504 | 0.20% | 0.23% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 0.25% | 1.44% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 4.98% | 5.26% |
| benign_degradation: DELAY | 89,990 | 0.00% | 95.38% |
| benign_degradation: DUPLICATION | 13,520 | 0.01% | 83.75% |
| benign_degradation: JITTER | 89,990 | 0.01% | 60.86% |
| benign_degradation: LINK_FLAP | 20,205 | 1.22% | 1.43% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 0.67% | 1.02% |
| benign_degradation: REORDERING | 12,333 | 0.01% | 59.08% |

> Highest attack_fpr: **CONGESTION_LOSS** (4.98%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

