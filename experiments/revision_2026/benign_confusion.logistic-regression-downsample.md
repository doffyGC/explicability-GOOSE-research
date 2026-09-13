# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-13 06:59:29 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `logistic-regression`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,050,939 | 2,143,251 | 1,811,813 | 2,590,298 | 3,885,028 | 2,257,976 | 22,739,305 |
| benign_degradation | 11,239 | 137,869 | 33,229 | 63,638 | 1,448 | 23,257 | 270,680 |
| SAG.DB | 2,184 | 2,325 | 37,414 | 2,686 | 3,126 | 3,047 | 50,782 |
| FRG | 4,443 | 19,689 | 9,114 | 21,251 | 485 | 8,994 | 63,976 |
| SAG.PB | 5,225 | 1,167 | 9,172 | 1,096 | 25,253 | 5,046 | 46,959 |
| SAG.PBM | 832 | 2,276 | 14,375 | 1,369 | 6,464 | 29,512 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 1,425 (6.6%) | 6,292 (29.4%) | 3,092 (14.4%) | 7,414 (34.6%) | 162 (0.8%) | 3,051 (14.2%) |
| DELAY | 89,990 | 0 (0.0%) | 69,234 (76.9%) | 7,119 (7.9%) | 9,945 (11.1%) | 77 (0.1%) | 3,615 (4.0%) |
| DUPLICATION | 13,520 | 1,418 (10.5%) | 6,809 (50.4%) | 1,748 (12.9%) | 1,246 (9.2%) | 179 (1.3%) | 2,120 (15.7%) |
| JITTER | 89,990 | 4 (0.0%) | 42,525 (47.3%) | 10,201 (11.3%) | 31,112 (34.6%) | 59 (0.1%) | 6,089 (6.8%) |
| LINK_FLAP | 20,205 | 4,533 (22.4%) | 3,511 (17.4%) | 3,691 (18.3%) | 4,651 (23.0%) | 602 (3.0%) | 3,217 (15.9%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 2,671 (11.5%) | 5,039 (21.7%) | 4,278 (18.4%) | 7,482 (32.2%) | 255 (1.1%) | 3,481 (15.0%) |
| REORDERING | 12,333 | 1,188 (9.6%) | 4,459 (36.2%) | 3,100 (25.1%) | 1,788 (14.5%) | 114 (0.9%) | 1,684 (13.7%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 20,333,504 | 44.44% | 52.52% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 62.68% | 83.55% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 64.00% | 93.35% |
| benign_degradation: DELAY | 89,990 | 23.06% | 100.00% |
| benign_degradation: DUPLICATION | 13,520 | 39.15% | 89.51% |
| benign_degradation: JITTER | 89,990 | 52.74% | 100.00% |
| benign_degradation: LINK_FLAP | 20,205 | 60.19% | 77.56% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 66.78% | 88.49% |
| benign_degradation: REORDERING | 12,333 | 54.21% | 90.37% |

> Highest attack_fpr: **QUEUE_OVERLOAD_BURST** (66.78%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

