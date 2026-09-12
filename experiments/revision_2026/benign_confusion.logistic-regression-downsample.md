# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-12 20:54:05 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `logistic-regression`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 9,559,132 | 1,699,912 | 1,485,715 | 2,102,219 | 3,566,090 | 1,972,856 | 20,385,924 |
| benign_degradation | 10,163 | 138,336 | 32,405 | 65,572 | 1,413 | 22,791 | 270,680 |
| SAG.DB | 763 | 848 | 11,759 | 1,035 | 1,346 | 1,343 | 17,094 |
| FRG | 1,382 | 6,578 | 3,030 | 7,245 | 150 | 3,051 | 21,436 |
| SAG.PB | 5,958 | 1,192 | 9,019 | 1,133 | 24,477 | 5,180 | 46,959 |
| SAG.PBM | 979 | 2,482 | 14,549 | 1,488 | 6,232 | 29,098 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 1,415 (6.6%) | 6,615 (30.9%) | 3,049 (14.2%) | 7,193 (33.6%) | 174 (0.8%) | 2,990 (13.9%) |
| DELAY | 89,990 | 0 (0.0%) | 68,842 (76.5%) | 6,760 (7.5%) | 10,710 (11.9%) | 167 (0.2%) | 3,511 (3.9%) |
| DUPLICATION | 13,520 | 1,276 (9.4%) | 6,941 (51.3%) | 1,821 (13.5%) | 1,271 (9.4%) | 167 (1.2%) | 2,044 (15.1%) |
| JITTER | 89,990 | 0 (0.0%) | 42,554 (47.3%) | 9,636 (10.7%) | 31,957 (35.5%) | 38 (0.0%) | 5,805 (6.5%) |
| LINK_FLAP | 20,205 | 4,104 (20.3%) | 3,574 (17.7%) | 3,830 (19.0%) | 4,934 (24.4%) | 523 (2.6%) | 3,240 (16.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 2,161 (9.3%) | 5,085 (21.9%) | 4,306 (18.6%) | 7,907 (34.1%) | 242 (1.0%) | 3,505 (15.1%) |
| REORDERING | 12,333 | 1,207 (9.8%) | 4,725 (38.3%) | 3,003 (24.3%) | 1,600 (13.0%) | 102 (0.8%) | 1,696 (13.8%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 17,980,123 | 42.21% | 48.82% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 63.91% | 85.16% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 62.54% | 93.40% |
| benign_degradation: DELAY | 89,990 | 23.50% | 100.00% |
| benign_degradation: DUPLICATION | 13,520 | 39.22% | 90.56% |
| benign_degradation: JITTER | 89,990 | 52.71% | 100.00% |
| benign_degradation: LINK_FLAP | 20,205 | 62.00% | 79.69% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 68.78% | 90.69% |
| benign_degradation: REORDERING | 12,333 | 51.90% | 90.21% |

> Highest attack_fpr: **QUEUE_OVERLOAD_BURST** (68.78%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

