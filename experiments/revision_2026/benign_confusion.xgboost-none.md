# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-13 07:00:22 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `xgboost`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 22,708,965 | 30,086 | 254 | 0 | 0 | 0 | 22,739,305 |
| benign_degradation | 108,199 | 162,481 | 0 | 0 | 0 | 0 | 270,680 |
| SAG.DB | 50,781 | 0 | 1 | 0 | 0 | 0 | 50,782 |
| FRG | 63,972 | 0 | 4 | 0 | 0 | 0 | 63,976 |
| SAG.PB | 46,959 | 0 | 0 | 0 | 0 | 0 | 46,959 |
| SAG.PBM | 54,828 | 0 | 0 | 0 | 0 | 0 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 21,436 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DELAY | 89,990 | 3,001 (3.3%) | 86,989 (96.7%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,520 | 1,259 (9.3%) | 12,261 (90.7%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 36,626 (40.7%) | 53,364 (59.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| LINK_FLAP | 20,205 | 20,205 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 23,206 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| REORDERING | 12,333 | 2,466 (20.0%) | 9,867 (80.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 20,333,504 | 0.00% | 0.00% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 0.00% | 1.25% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 0.00% | 0.00% |
| benign_degradation: DELAY | 89,990 | 0.00% | 96.67% |
| benign_degradation: DUPLICATION | 13,520 | 0.00% | 90.69% |
| benign_degradation: JITTER | 89,990 | 0.00% | 59.30% |
| benign_degradation: LINK_FLAP | 20,205 | 0.00% | 0.00% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 0.00% | 0.00% |
| benign_degradation: REORDERING | 12,333 | 0.00% | 80.00% |

> Highest attack_fpr: **CONGESTION_LOSS** (0.00%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

