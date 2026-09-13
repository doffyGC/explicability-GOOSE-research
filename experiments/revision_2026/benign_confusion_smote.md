# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-13 07:14:25 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 22,694,841 | 31,132 | 2,311 | 171 | 10,477 | 373 | 22,739,305 |
| benign_degradation | 129,251 | 138,879 | 1,943 | 547 | 0 | 60 | 270,680 |
| SAG.DB | 50,773 | 2 | 3 | 0 | 4 | 0 | 50,782 |
| FRG | 63,972 | 0 | 3 | 1 | 0 | 0 | 63,976 |
| SAG.PB | 46,903 | 0 | 0 | 0 | 56 | 0 | 46,959 |
| SAG.PBM | 54,822 | 0 | 1 | 0 | 5 | 0 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 21,432 (100.0%) | 0 (0.0%) | 3 (0.0%) | 1 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DELAY | 89,990 | 11,243 (12.5%) | 77,971 (86.6%) | 702 (0.8%) | 54 (0.1%) | 0 (0.0%) | 20 (0.0%) |
| DUPLICATION | 13,520 | 5,081 (37.6%) | 8,134 (60.2%) | 305 (2.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| JITTER | 89,990 | 39,620 (44.0%) | 49,057 (54.5%) | 929 (1.0%) | 345 (0.4%) | 0 (0.0%) | 39 (0.0%) |
| LINK_FLAP | 20,205 | 20,204 (100.0%) | 0 (0.0%) | 1 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 23,205 (100.0%) | 0 (0.0%) | 1 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| REORDERING | 12,333 | 8,466 (68.6%) | 3,717 (30.1%) | 2 (0.0%) | 147 (1.2%) | 0 (0.0%) | 1 (0.0%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 20,333,504 | 0.06% | 0.06% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 0.04% | 1.33% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 0.02% | 0.02% |
| benign_degradation: DELAY | 89,990 | 0.86% | 87.51% |
| benign_degradation: DUPLICATION | 13,520 | 2.26% | 62.42% |
| benign_degradation: JITTER | 89,990 | 1.46% | 55.97% |
| benign_degradation: LINK_FLAP | 20,205 | 0.00% | 0.00% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 0.00% | 0.00% |
| benign_degradation: REORDERING | 12,333 | 1.22% | 31.35% |

> Highest attack_fpr: **DUPLICATION** (2.26%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

