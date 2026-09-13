# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-13 06:58:34 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 11,849,885 | 347,697 | 1,248,414 | 2,982,156 | 3,284,242 | 3,026,911 | 22,739,305 |
| benign_degradation | 16,670 | 171,047 | 13,688 | 54,097 | 289 | 14,889 | 270,680 |
| SAG.DB | 7 | 59 | 46,496 | 530 | 1,756 | 1,934 | 50,782 |
| FRG | 7,501 | 1,686 | 7,229 | 36,579 | 157 | 10,824 | 63,976 |
| SAG.PB | 1,026 | 155 | 9,126 | 1,230 | 29,890 | 5,532 | 46,959 |
| SAG.PBM | 1,015 | 358 | 12,828 | 2,194 | 5,396 | 33,037 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,436 | 2,096 (9.8%) | 441 (2.1%) | 2,425 (11.3%) | 12,936 (60.3%) | 70 (0.3%) | 3,468 (16.2%) |
| DELAY | 89,990 | 25 (0.0%) | 83,726 (93.0%) | 1,984 (2.2%) | 4,166 (4.6%) | 0 (0.0%) | 89 (0.1%) |
| DUPLICATION | 13,520 | 11 (0.1%) | 11,873 (87.8%) | 450 (3.3%) | 755 (5.6%) | 5 (0.0%) | 426 (3.2%) |
| JITTER | 89,990 | 77 (0.1%) | 65,300 (72.6%) | 2,495 (2.8%) | 21,909 (24.3%) | 0 (0.0%) | 209 (0.2%) |
| LINK_FLAP | 20,205 | 7,228 (35.8%) | 218 (1.1%) | 2,487 (12.3%) | 5,698 (28.2%) | 117 (0.6%) | 4,457 (22.1%) |
| QUEUE_OVERLOAD_BURST | 23,206 | 6,997 (30.2%) | 468 (2.0%) | 2,914 (12.6%) | 7,865 (33.9%) | 97 (0.4%) | 4,865 (21.0%) |
| REORDERING | 12,333 | 236 (1.9%) | 9,021 (73.1%) | 933 (7.6%) | 768 (6.2%) | 0 (0.0%) | 1,375 (11.1%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 20,333,504 | 44.14% | 45.15% |
| normal (baseline messages inside a benign-impairment run) | 2,405,801 | 65.16% | 71.03% |
| benign_degradation: CONGESTION_LOSS | 21,436 | 88.16% | 90.22% |
| benign_degradation: DELAY | 89,990 | 6.93% | 99.97% |
| benign_degradation: DUPLICATION | 13,520 | 12.10% | 99.92% |
| benign_degradation: JITTER | 89,990 | 27.35% | 99.91% |
| benign_degradation: LINK_FLAP | 20,205 | 63.15% | 64.23% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,206 | 67.83% | 69.85% |
| benign_degradation: REORDERING | 12,333 | 24.94% | 98.09% |

> Highest attack_fpr: **CONGESTION_LOSS** (88.16%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

