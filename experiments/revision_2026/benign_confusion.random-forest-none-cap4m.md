# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 20:59:19 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `random-forest`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 20,325,210 | 35,609 | 4,411 | 1,629 | 8,490 | 10,575 | 20,385,924 |
| benign_degradation | 109,793 | 159,719 | 175 | 737 | 50 | 206 | 270,680 |
| SAG.DB | 16,865 | 156 | 15 | 5 | 17 | 36 | 17,094 |
| FRG | 20,638 | 719 | 3 | 30 | 11 | 35 | 21,436 |
| SAG.PB | 44,912 | 68 | 19 | 12 | 1,780 | 168 | 46,959 |
| SAG.PBM | 51,251 | 187 | 26 | 47 | 192 | 3,125 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 20,590 (96.1%) | 66 (0.3%) | 2 (0.0%) | 716 (3.3%) | 13 (0.1%) | 49 (0.2%) |
| DELAY | 89,990 | 4,404 (4.9%) | 85,586 (95.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,520 | 1,914 (14.2%) | 11,606 (85.8%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 34,814 (38.7%) | 55,176 (61.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| LINK_FLAP | 20,205 | 19,897 (98.5%) | 46 (0.2%) | 168 (0.8%) | 10 (0.0%) | 19 (0.1%) | 65 (0.3%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 23,008 (99.1%) | 72 (0.3%) | 5 (0.0%) | 11 (0.0%) | 18 (0.1%) | 92 (0.4%) |
| REORDERING | 12,333 | 5,166 (41.9%) | 7,167 (58.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 0.11% | 0.14% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 0.22% | 1.46% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 3.64% | 3.95% |
| benign_degradation: DELAY | 89,990 | 0.00% | 95.11% |
| benign_degradation: DUPLICATION | 13,520 | 0.00% | 85.84% |
| benign_degradation: JITTER | 89,990 | 0.00% | 61.31% |
| benign_degradation: LINK_FLAP | 20,205 | 1.30% | 1.52% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 0.54% | 0.85% |
| benign_degradation: REORDERING | 12,333 | 0.00% | 58.11% |

> Highest attack_fpr: **CONGESTION_LOSS** (3.64%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

