# experiments/revision_2026

Scripts produced for the major revision of Gray-GOOSE paper.

## Current system (baseline, frozen for reference)

Main scripts of the pipeline that generated the submitted (rejected)
results, kept here as a reference point for the revision.

| Script | Description |
|---|---|
| `main.py` | Entry point: orchestrates the full pipeline — load dataset, preprocess, train (K-Fold CV), evaluate, save reports, run SHAP. |
| `config.py` | Global settings: seed, K-Fold splits, model type/params, discarded/delta feature columns, class names, dataset path, SHAP plot list. |
| `data/loader.py` | Loads the dataset from `.csv`/`.parquet` into a DataFrame. |
| `data/preprocess.py` | Splits the DataFrame into features (`X`) and target (`y`), label-encodes the class column. |
| `model/train.py` | Trains a classifier (XGBoost/RF/SVM/MLP/decision tree/logistic regression) with `StratifiedKFold` CV, plus a final model on all data. |
| `model/evaluate.py` | Computes per-class/global metrics with confidence intervals across CV folds, aggregates the confusion matrix, saves Markdown/log reports and `y_true`/`y_pred` arrays (for McNemar). |
| `model/matrix_confusion.py` | Plots and saves the confusion matrix (log-scale heatmap) as SVG. |
| `explainability/shap_analysis.py` | Runs SHAP (`TreeExplainer`) on the last CV fold's validation set and saves bar/beeswarm plots per class. |
| `notebooks/experiment.ipynb` | Same pipeline as `main.py`, runnable interactively section-by-section; used to reproduce and inspect results manually. |

> Known issues this revision must fix: `StratifiedKFold` splits at message level
> (no `run_id`/`trace_id` grouping → leakage risk), no benign-only packet-loss
> controls, SHAP computed on a CV fold rather than a proper held-out set, and
> McNemar/statistics run over pooled message-level predictions. See
> `prioridades_revisao_gray_goose.pdf` at the repo root for the full plan.

## New scripts for this revision

Each script added here should get one line below describing what it does
and which checklist item it covers. Keep entries short; details belong in
the script's own docstring/comments, not in this file.

| Script | Description | Checklist ref. |
|---|---|---|
| _(none yet)_ | | |

As scripts land (e.g. leakage checks, grouped-split protocol, benign
control generation, ablations, baselines, balancing experiments,
grouped statistics / held-out SHAP), add a row here in the same commit.
