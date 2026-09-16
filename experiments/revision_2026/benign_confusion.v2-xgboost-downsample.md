# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:55:39 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `xgboost`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 8,297,704 | 1,292,004 | 33,173 | 105,611 | 84,775 | 757,040 | 10,570,307 |
| benign_degradation | 2,319 | 233,232 | 2,242 | 21,781 | 3,011 | 8,057 | 270,642 |
| SAG.DB | 0 | 6 | 48,662 | 39 | 1,949 | 123 | 50,779 |
| FRG | 413 | 1,304 | 879 | 49,762 | 730 | 10,875 | 63,963 |
| SAG.PB | 193 | 31 | 6,746 | 61 | 39,159 | 769 | 46,959 |
| SAG.PBM | 175 | 227 | 2,119 | 4,803 | 2,561 | 44,943 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 96 (0.4%) | 284 (1.3%) | 263 (1.2%) | 16,342 (76.3%) | 287 (1.3%) | 4,156 (19.4%) |
| DELAY | 89,980 | 567 (0.6%) | 88,821 (98.7%) | 4 (0.0%) | 532 (0.6%) | 9 (0.0%) | 47 (0.1%) |
| DUPLICATION | 13,519 | 1 (0.0%) | 13,514 (100.0%) | 0 (0.0%) | 2 (0.0%) | 0 (0.0%) | 2 (0.0%) |
| JITTER | 89,980 | 1,324 (1.5%) | 87,746 (97.5%) | 27 (0.0%) | 739 (0.8%) | 7 (0.0%) | 137 (0.2%) |
| LINK_FLAP | 20,205 | 192 (1.0%) | 14,400 (71.3%) | 921 (4.6%) | 1,644 (8.1%) | 1,329 (6.6%) | 1,719 (8.5%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 135 (0.6%) | 16,409 (70.7%) | 956 (4.1%) | 2,502 (10.8%) | 1,377 (5.9%) | 1,821 (7.8%) |
| REORDERING | 12,330 | 4 (0.0%) | 12,058 (97.8%) | 71 (0.6%) | 20 (0.2%) | 2 (0.0%) | 175 (1.4%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 8.67% | 19.04% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 15.64% | 47.29% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 98.23% | 99.55% |
| benign_degradation: DELAY | 89,980 | 0.66% | 99.37% |
| benign_degradation: DUPLICATION | 13,519 | 0.03% | 99.99% |
| benign_degradation: JITTER | 89,980 | 1.01% | 98.53% |
| benign_degradation: LINK_FLAP | 20,205 | 27.78% | 99.05% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 28.69% | 99.42% |
| benign_degradation: REORDERING | 12,330 | 2.17% | 99.97% |

> Highest attack_fpr: **CONGESTION_LOSS** (98.23%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

