# README — Models ML Evaluation with Explainability in IEC 61850

## System Objective

This project aims to evaluate the performance of machine learning models against grayhole DoS attacks in relation to the IEC 61850 standard, with a focus on the experiments reproducibility and results interpretability.
The system 
The system integrates data preprocessing, model training and evaluation, and explanation generation using XAI (SHAP library) techniques.The experiments are documented and executable via Jupyter Notebook, allowing researchers to fully reproduce the reported results.

---

## Directory Structure

```
.
├── data/               # Datasets used in experiments and data manipulation modules
    ├── CSV files/      # Location to add the CSV files
├── explainability/     # Modules and scripts for generating model explanations (SHAP)
├── model/              # Definition, training, and serialization of the evaluated models
├── notebooks/          # Jupyter Notebooks with step-by-step execution of experiments
├── config.py           # Global project settings (paths, hyperparameters, flags)
└── main.py             # Main entry point for executing the entire pipeline via CLI
```

---

## Pre-requisites

- [Git](https://git-scm.com/)
- Python 3.10+ *(to execute in venv)*
- [Docker](https://www.docker.com/) and Docker Compose *(to execute in container)*

Clone the repository before any of the options below:

```bash
git clone https://github.com/doffyGC/explicability-GOOSE-research.git
cd explicability-GOOSE-research
```

---

## Option 1 — Execute via Docker (Recommend)

This is the most robust way to ensure reproducibility, as Docker encapsulates the operating system, the Python version, and all dependencies in an environment identical to that used during development.

### 1.1 Build the image

```bash
docker build -t ml-evaluation .
```

### 1.2 Execute the complete pipeline 

```bash
docker run --rm ml-evaluation python main.py
```

### 1.3 Execute the Jupyter Notebook interactively

```bash
docker run --rm -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks ml-evaluation \
    jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

After executing the command, access the address displayed in the terminal in your browser, in the format:

```
http://127.0.0.1:8888/tree?token=<your-token>
```

Navigate to the `notebooks/` folder and open the desired notebook.

> **Tip:** The volume `-v $(pwd)/notebooks:/app/notebooks` ensures that changes made to notebooks within the container are saved locally.

---

## Opção 2 — Execute via Virtual Environment (venv)

This option is recommended for those who want to explore and modify the code without installing Docker, and assumes that Python 3.10+ **is already installed** on the machine.

### 2.1 Create and activate the virtual environment.


```bash
# Create the virtual environment.
python -m venv .venv

# Active — Linux/macOS
source .venv/bin/activate

# Active — Windows
.venv\Scripts\activate
```

### 2.2 Install the dependences

```bash
pip install -r requirements.txt
```

### 2.3 Execute the complete pipeline

```bash
python main.py
```

### 2.4 Execute the Jupyter notebook

```bash
jupyter notebook
```

Jupyter will open automatically in your browser. Navigate to the `notebooks/` folder and open the desired notebook.

---

## Executando o Notebook
 
The entire experimental pipeline is contained in a single file: `notebooks/experiment.ipynb`. The notebook is divided into numbered sections and must be run **from top to bottom, without skipping cells**, as each section depends on the state defined by the previous ones.
 
### Notebook Structure
 
| Section | Description |
|---|---|
| 1. Imports | Loading the necessary libraries |
| 2. Scenarios Definition | Centralized definition of parameters for both scenarios |
| 3. Active Scenario Selection | **Control cell** —switch here to change the active scenario |
| 4. Data Loader | Loading the configured dataset |
| 5. Pre-processing | Data transformation and preparation |
| 6. Training with K-Fold Validation | Model training with cross-validation |
| 7. Evaluation | Calculation of metrics and confusion matrix |
| 8. Final Results | Consolidated display of results |
| 9. Save Metrics Report | Exporting results to a file |
| 10. Explainability Analysis (SHAP) | SHAP analysis — **disabled by default** (see the note below) |
| 11. Execute all Scenarios | Automatic loop to execute all of the scenarios sequentially |
 
### To reproduce the results
 
1. Make sure the data in `data/CSV files/` is present before starting.
2. Open `notebooks/experiment.ipynb` in Jupyter.
3. In **Section 3**, Define the desired scenario in the `ACTIVE_SCENARIO` variable — or use **Section 11** to automatically run all scenarios in a loop.
4. Execute all cells in order via `Kernel > Restart & Run All`.
5. The results (metrics, reports, explanations) will be automatically saved to the directories configured for each scenario.
 
> **Note about SHAP:** The explainability analysis in Section 10 is disabled by default because it is computationally expensive. To enable it, change the value of the corresponding flag in that cell before running.
 
---

## Configuration

The `config.py` file centralizes the main project parameters, such as data paths, model hyperparameters, and execution flags. Review it before running experiments if you wish to adjust any parameters.

---

## Dependences

All dependencies are listed in `requirements.txt`. The main libraries used include:

| Library | purpose |
|---|---|
| `scikit-learn` | Training and evaluation of models |
| `pandas` / `numpy` | Data manipulation |
| `scipy`| Calculation of mean and confidence interval for performance evaluation metrics |
| `shap` | Explainability of models |
| `matplotlib`| Viewing results |
| `jupyter` | Interactive execution of experiments |
| `xgboost`| Gradient boosting-based classifier model used in the experiments |

---

## Reproducibility

To ensure that the results are identical each time they are run:

- The random seed is defined in `config.py` and is applied globally.
- The dataset used in the experiment should be added to the `data/CSV files` folder.
- The Docker environment ensures complete isolation of the operating system and dependencies.
