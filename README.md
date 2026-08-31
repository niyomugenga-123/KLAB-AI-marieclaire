# KLAB AI Bootcamp — Marie Claire Niyomugenga

Project repository for the tekHer AI Cohort 2 program at kLab, Rwanda.

## Setup Instructions

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You'll know it worked when your terminal prompt starts with `(.venv)`.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the smoke test

Open `notebooks/python_basics.ipynb` (or your assignment notebook) in VS Code or Jupyter, and run this cell:

```python
import numpy, pandas, sklearn, matplotlib
print("all good")
```

If it prints `all good`, your environment is set up correctly.

### 4. Open the assignment notebook

- Open the `klab-ai-marieclaire` folder in VS Code
- Go to the `notebooks/` folder and open the assignment notebook
- Click **Select Kernel** (top-right) → **Python Environments** → choose `.venv (3.14.7)`
- Run cells with `Shift+Enter`

### 5. Project structure

```
klab-ai-marieclaire/
├── notebooks/       # Jupyter notebooks
├── src/             # reusable Python functions
├── data/raw/        # raw, unprocessed data
├── data/processed/  # cleaned/processed data
├── reports/         # charts and reflection write-ups
├── requirements.txt # exact package versions
├── .env.example     # environment variable names (no real values)
└── .gitignore       # files/folders excluded from version control
```

## Dataset (Assignment 2)

- **Name:** Carvana - Predict Car Prices
- **Source:** Kaggle — https://www.kaggle.com/datasets/ravishah1/carvana-predict-car-prices (by Ravi Shah)
- **License:** CC0: Public Domain
- **Rows/columns:** 22,000 rows, 4 columns (`Name`, `Year`, `Miles`, `Price`) before
  cleaning; 9,317 rows after removing exact duplicates.
- **Why chosen:** a realistic, moderately messy dataset with one categorical and three
  numeric columns — real data-quality problems (corrupted `Year` values, many duplicate
  rows) to practice deliberate cleaning decisions on, rather than a dataset that's
  already clean.

  
## Day 5 — Linear Regression & Random Forest (Car Price Prediction)

Notebook: `notebooks/day05_ml_models.ipynb`

Builds two models to predict used car `Price` from `Year` and `Miles`, using
`data/raw/carvana.csv`, then compares them.

**Results:**

| Model | MAE | R² |
|---|---|---|
| Linear Regression | $4,131.52 | 0.368 |
| Random Forest | $4,508.57 | 0.168 |

**Finding:** Linear Regression outperformed Random Forest here. With only two
fairly linear features (`Year`, `Miles`) and noisy labels (identical Year/Miles
rows with different prices), Random Forest's added flexibility overfits to noise
rather than capturing genuine complexity — so the simpler model generalizes
better. Full reasoning is in the notebook's conclusion section.

## Week 2: Classification & Metrics

### Day 3: Classification Metrics Assignment ✅
- **Task:** Build a classifier to predict premium cars (≥$30,000) vs regular cars
- **Dataset:** Carvana listings (21,000 samples, 9.8% premium)
- **Model:** Logistic Regression
- **Features:** Age, Miles
- **Results:**
  - Accuracy: 94.2% (misleading on imbalanced data)
  - Precision: 80% (when it says premium, it's correct 4/5 times)
  - Recall: 1.5% (finds only 4 out of 260 actual premium cars)
  - F1: 3%
  - ROC-AUC: 80.4%
- **Key Insight:** Recall prioritized as business metric — missing premium listings costs more than false alarms
- **Files:** 
  - `notebooks/Metrics Assignment.ipynb` — full implementation
  - `reports/Classification_Metrics_Defense.md` — 1-page defense essay