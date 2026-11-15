# Task 4 — Random Forests & Ensembles

**Course:** MOP 3231, Machine Learning (Python)  
**Student:** Nurdaulet Zhussip

This repository contains my Task 4 spam-classification notebook and the supplied Spambase dataset. The notebook is `zhussip_task4.ipynb`; it covers bagging and out-of-bag evaluation, model evaluation and threshold selection, voting, stacking, AdaBoost, label noise, and the final recommendation.

## Project files

| File | Description |
|---|---|
| `zhussip_task4.ipynb` | Executable solution for Parts 1–6 |
| `data/spambase.csv` | 4,204 email examples, 57 numeric features, and the binary `spam` label |
| `requirements.txt` | Pinned dependencies supplied for the task; do not modify |

## Run the notebook

The notebook uses Python 3.11–3.14 and the pinned dependencies. From the repository root, create and activate the virtual environment and install them:

```powershell
python -m venv .lastname_ML_task_4
.\.lastname_ML_task_4\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `zhussip_task4.ipynb` in Jupyter and use **Restart Kernel and Run All**. The notebook uses `SEED = 2729`, a stratified 75/25 train/test split, and one shared stratified five-fold CV object. The test split is reserved for reporting final results and is not used to select the forest threshold.

The environment and `.ipynb_checkpoints` are excluded from Git.

## Dataset and attribution

`data/spambase.csv` is derived from the [Spambase dataset](https://doi.org/10.24432/C53G6X) by Hopkins, Reeber, Forman, and Suermondt (1999), from the UCI Machine Learning Repository. The source dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The supplied CSV contains 4,204 rows: 1,676 spam and 2,528 legitimate emails. Its 57 feature columns are named using the source feature descriptions; six names containing punctuation were replaced with descriptive identifiers. Exact duplicate rows and six rows with identical feature vectors but conflicting labels were removed. The remaining rows were shuffled with seed 3231 during dataset preparation. Feature values are otherwise raw—not scaled, clipped, or rounded.

This historical dataset is for coursework and should not be treated as evidence about present-day spam.
