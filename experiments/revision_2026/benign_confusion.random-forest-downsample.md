# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-13 06:59:02 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `random-forest`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 12,803,273 | 1,013,646 | 857,763 | 3,184,501 | 2,579,145 | 2,300,977 | 22,739,305 |
| benign_degradation | 14,348 | 192,602 | 5,920 | 45,438 | 3,049 | 9,323 | 270,680 |
| SAG.DB | 1,170 | 673 | 33,549 | 3,752 | 4,822 | 6,816 | 50,782 |
| FRG | 6,493 | 5,415 | 4,808 | 39,774 | 1,489 | 5,997 | 63,976 |
| SAG.PB | 3,201 | 364 | 5,428 | 1,223 | 31,113 | 5,630 | 46,959 |
| SAG.PBM | 3,416 | 956 | 5,563 | 3,580 | 7,226 | 34,087 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 983 (4.6%) | 1,036 (4.8%) | 787 (3.7%) | 16,396 (76.5%) | 513 (2.4%) | 1,721 (8.0%) |
| DELAY | 89,990 | 17 (0.0%) | 89,791 (99.8%) | 19 (0.0%) | 153 (0.2%) | 2 (0.0%) | 8 (0.0%) |
| DUPLICATION | 13,520 | 323 (2.4%) | 12,550 (92.8%) | 152 (1.1%) | 168 (1.2%) | 134 (1.0%) | 193 (1.4%) |
| JITTER | 89,990 | 1,362 (1.5%) | 73,943 (82.2%) | 938 (1.0%) | 13,331 (14.8%) | 160 (0.2%) | 256 (0.3%) |
| LINK_FLAP | 20,205 | 5,771 (28.6%) | 1,585 (7.8%) | 1,957 (9.7%) | 6,566 (32.5%) | 1,093 (5.4%) | 3,233 (16.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 5,574 (24.0%) | 2,235 (9.6%) | 1,982 (8.5%) | 8,564 (36.9%) | 1,109 (4.8%) | 3,742 (16.1%) |
| REORDERING | 12,333 | 318 (2.6%) | 11,462 (92.9%) | 85 (0.7%) | 260 (2.1%) | 38 (0.3%) | 170 (1.4%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 20,333,504 | 37.21% | 40.63% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 56.37% | 69.58% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 90.58% | 95.41% |
| benign_degradation: DELAY | 89,990 | 0.20% | 99.98% |
| benign_degradation: DUPLICATION | 13,520 | 4.79% | 97.61% |
| benign_degradation: JITTER | 89,990 | 16.32% | 98.49% |
| benign_degradation: LINK_FLAP | 20,205 | 63.59% | 71.44% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 66.35% | 75.98% |
| benign_degradation: REORDERING | 12,333 | 4.48% | 97.42% |

> Highest attack_fpr: **CONGESTION_LOSS** (90.58%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

