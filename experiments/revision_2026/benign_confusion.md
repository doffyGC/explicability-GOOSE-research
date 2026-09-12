# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 15:33:26 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 20,351,321 | 34,581 | 0 | 1 | 20 | 1 | 20,385,924 |
| benign_degradation | 112,193 | 158,487 | 0 | 0 | 0 | 0 | 270,680 |
| SAG.DB | 17,094 | 0 | 0 | 0 | 0 | 0 | 17,094 |
| FRG | 21,435 | 1 | 0 | 0 | 0 | 0 | 21,436 |
| SAG.PB | 46,959 | 0 | 0 | 0 | 0 | 0 | 46,959 |
| SAG.PBM | 54,828 | 0 | 0 | 0 | 0 | 0 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 21,436 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DELAY | 89,990 | 2,520 (2.8%) | 87,470 (97.2%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DUPLICATION | 13,520 | 3,272 (24.2%) | 10,248 (75.8%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 34,770 (38.6%) | 55,220 (61.4%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| LINK_FLAP | 20,205 | 20,205 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 23,206 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| REORDERING | 12,333 | 6,784 (55.0%) | 5,549 (45.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 0.00% | 0.00% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 0.00% | 1.44% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 0.00% | 0.00% |
| benign_degradation: DELAY | 89,990 | 0.00% | 97.20% |
| benign_degradation: DUPLICATION | 13,520 | 0.00% | 75.80% |
| benign_degradation: JITTER | 89,990 | 0.00% | 61.36% |
| benign_degradation: LINK_FLAP | 20,205 | 0.00% | 0.00% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 0.00% | 0.00% |
| benign_degradation: REORDERING | 12,333 | 0.00% | 44.99% |

> Highest attack_fpr: **CONGESTION_LOSS** (0.00%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

