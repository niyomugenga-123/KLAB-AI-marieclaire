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