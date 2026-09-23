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

  ## Week 3: Unsupervised Learning & Clustering Evaluation

### Assignment: Evaluating Unsupervised Models (Overfitting, Underfitting, and How to Fix Them)

**Challenge:** How do you evaluate clustering models without labels? No "accuracy" available, yet models can still overfit/underfit.

**5-Part Assignment Completed:**

#### Part 1: Overlapping Clusters
- Generated data with increasing cluster overlap (cluster_std=1.15 → 2.5)
- Observed silhouette score peak flatten as clusters overlap
- **Finding:** Silhouette loses confidence on ambiguous data

#### Part 2: Unequal Cluster Sizes
- Tested K-Means vs DBSCAN with unequal sizes [0.5, 0.5, 3.0, 0.5]
- **Finding:** K-Means splits large clusters; DBSCAN respects actual cluster sizes

#### Part 3: Pure Noise Baseline
- Ran clustering on 1200 random points (no real structure)
- K-Means still returned clusters with silhouette ≈ 0.33
- **Finding:** Silhouette alone gives false positives; need threshold (~0.50)

#### Part 4: Automatic Elbow Detection
- Implemented function to find elbow (furthest point from reference line)
- Tested on original (std=1.15) and hard data (std=2.5)
- **Finding:** Elbow robust on synthetic data but misled by smooth inertia curves

#### Part 5: Real Data (Wine Dataset)
- Applied metrics to 178-sample wine dataset (13 features, 3 true classes)
- **Silhouette predicted:** k=3 ✅ (CORRECT)
- **Elbow predicted:** k=4 ❌ (incorrect)
- **Key Insight:** Silhouette is more reliable than elbow on real data

### Key Learnings

| Metric | Strengths | Weaknesses | When to Use |
|--------|-----------|-----------|------------|
| **Silhouette** | Measures cluster cohesion + separation | Low scores on all real data | Primary metric; best for real data |
| **Elbow** | Clear geometric interpretation | Misled by smooth curves | Confirmation; synthetic data |
| **Stability** | Real structure survives resampling | Computationally expensive | Verify final answer |
| **Inertia** | Always available | Always decreases; can't choose k | Supporting info only |

### Critical Rules Learned

1. **Silhouette score threshold:** < 0.50 = probably noise, > 0.60 = likely real
2. **When methods disagree:** Trust silhouette (measures what we care about)
3. **Labels ≠ ground truth:** Wine has 3 classes, but that's just one clustering perspective
4. **Use multiple metrics:** No single metric is perfect; combine for robust decisions

### Files

-
## Chapter 1 — Image Classifier (Bird / Forest / Mountain / River)

Notebook: `notebooks/Chapter1/Chapter1_Image_Classifier.ipynb`
Model: `models/image_classifier_model.pkl`
Report: `reports/Chapter1_Report.md`

Trained a ResNet18 image classifier (transfer learning, fastai) on ~30 custom
photos per category across 4 classes: bird, forest, mountain, river.

**Results:**
- Final validation accuracy: **83.3%**
- Differentiation test: correctly classified **4/4** held-out sample images,
  one from each category, confirming the model distinguishes between all
  classes rather than favoring one

**Key steps:**
- Resized all images to a consistent 224x224 before batching
  (`item_tfms=Resize(224)`), required since raw photos varied in dimensions
- Used `fine_tune()` for transfer learning: 1 frozen warm-up epoch, then 4
  epochs training the full network
- Exported the full inference pipeline (model + preprocessing) with
  `learn.export()` for reuse