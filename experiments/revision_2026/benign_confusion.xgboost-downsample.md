# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 20:53:16 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `xgboost`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 11,354,902 | 464,520 | 969,736 | 2,717,549 | 2,420,939 | 2,458,278 | 20,385,924 |
| benign_degradation | 12,599 | 185,223 | 9,103 | 50,382 | 1,738 | 11,635 | 270,680 |
| SAG.DB | 13 | 25 | 15,342 | 330 | 482 | 902 | 17,094 |
| FRG | 1,994 | 663 | 2,396 | 12,799 | 287 | 3,297 | 21,436 |
| SAG.PB | 1,229 | 53 | 8,818 | 1,350 | 31,231 | 4,278 | 46,959 |
| SAG.PBM | 1,263 | 40 | 12,422 | 3,487 | 6,695 | 30,921 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 1,332 (6.2%) | 263 (1.2%) | 2,347 (10.9%) | 14,216 (66.3%) | 228 (1.1%) | 3,050 (14.2%) |
| DELAY | 89,990 | 141 (0.2%) | 88,957 (98.9%) | 89 (0.1%) | 741 (0.8%) | 6 (0.0%) | 56 (0.1%) |
| DUPLICATION | 13,520 | 1 (0.0%) | 13,510 (99.9%) | 7 (0.1%) | 1 (0.0%) | 1 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 857 (1.0%) | 69,423 (77.1%) | 1,336 (1.5%) | 18,078 (20.1%) | 95 (0.1%) | 201 (0.2%) |
| LINK_FLAP | 20,205 | 5,186 (25.7%) | 517 (2.6%) | 2,412 (11.9%) | 7,566 (37.4%) | 764 (3.8%) | 3,760 (18.6%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 5,035 (21.7%) | 720 (3.1%) | 2,804 (12.1%) | 9,654 (41.6%) | 641 (2.8%) | 4,352 (18.8%) |
| REORDERING | 12,333 | 47 (0.4%) | 11,833 (95.9%) | 108 (0.9%) | 126 (1.0%) | 3 (0.0%) | 216 (1.8%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 38.69% | 40.04% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 66.92% | 76.16% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 92.56% | 93.79% |
| benign_degradation: DELAY | 89,990 | 0.99% | 99.84% |
| benign_degradation: DUPLICATION | 13,520 | 0.07% | 99.99% |
| benign_degradation: JITTER | 89,990 | 21.90% | 99.05% |
| benign_degradation: LINK_FLAP | 20,205 | 71.77% | 74.33% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 75.20% | 78.30% |
| benign_degradation: REORDERING | 12,333 | 3.67% | 99.62% |

> Highest attack_fpr: **CONGESTION_LOSS** (92.56%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

