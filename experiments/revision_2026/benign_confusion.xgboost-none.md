# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 20:59:43 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `xgboost`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 20,362,020 | 23,904 | 0 | 0 | 0 | 0 | 20,385,924 |
| benign_degradation | 119,057 | 151,623 | 0 | 0 | 0 | 0 | 270,680 |
| SAG.DB | 17,094 | 0 | 0 | 0 | 0 | 0 | 17,094 |
| FRG | 21,436 | 0 | 0 | 0 | 0 | 0 | 21,436 |
| SAG.PB | 46,959 | 0 | 0 | 0 | 0 | 0 | 46,959 |
| SAG.PBM | 54,828 | 0 | 0 | 0 | 0 | 0 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 21,436 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DELAY | 89,990 | 4,453 (4.9%) | 85,537 (95.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,520 | 4,071 (30.1%) | 9,449 (69.9%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 43,163 (48.0%) | 46,827 (52.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| LINK_FLAP | 20,205 | 20,205 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 23,206 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| REORDERING | 12,333 | 2,523 (20.5%) | 9,810 (79.5%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 0.00% | 0.00% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 0.00% | 0.99% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 0.00% | 0.00% |
| benign_degradation: DELAY | 89,990 | 0.00% | 95.05% |
| benign_degradation: DUPLICATION | 13,520 | 0.00% | 69.89% |
| benign_degradation: JITTER | 89,990 | 0.00% | 52.04% |
| benign_degradation: LINK_FLAP | 20,205 | 0.00% | 0.00% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 0.00% | 0.00% |
| benign_degradation: REORDERING | 12,333 | 0.00% | 79.54% |

> Highest attack_fpr: **CONGESTION_LOSS** (0.00%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

