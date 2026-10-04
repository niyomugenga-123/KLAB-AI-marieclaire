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


 # Irony and Sarcasm Detection in Tweets

A text classification project that labels tweets as `figurative`, `irony`, `sarcasm` or `regular`.

- **Part 1:** a classical NLP pipeline (cleaning, TF-IDF features, linear models).
- **Part 2:** the same project improved with pre-trained word embeddings (GloVe-Twitter), compared against Part 1 on the same metrics.

Blog post on word embeddings (Dev.to): [ADD LINK HERE]

## Results at a glance

| Version | Features | Model | Test accuracy |
|---|---|---|---|
| Part 1 (before) | TF-IDF, 5,000 word 1-2-grams | Linear SVM (C=0.01), calibrated | 0.584 |
| Part 2, embeddings only | GloVe-Twitter-100 (mean + max pooling) | Logistic Regression | 0.500 |
| Part 2 (after) | TF-IDF + GloVe-Twitter-100 | Linear SVM (C=0.03) | **0.591** |

Embeddings alone did not beat TF-IDF. Adding them to TF-IDF gave a small gain (+0.007), mostly on `irony` and `sarcasm`. Details are in the Part 2 section.

## Repository structure

```
data/raw/            train.csv, test.csv (not modified)
models/              saved models (Part 1 and Part 2)
models/embeddings/   downloaded GloVe file (not committed, see .gitignore)
notebooks/           nlp_tweets.ipynb  (all code: Part 1 and Part 2)
README.md
```

The notebook reads data with relative paths (`../data/raw/...`), so it is run from inside its folder, one level below the repository root.

## How to run

1. Put `train.csv` and `test.csv` in `data/raw/`.
2. Install the dependencies:
   ```
   pip install pandas numpy scikit-learn scipy matplotlib joblib vaderSentiment
   ```
3. Open `nlp_tweets.ipynb` and run it from top to bottom.
   - The cells up to the saved model are Part 1.
   - The embeddings cells are Part 2. On the first run they download GloVe-Twitter-25 (about 104 MB) and GloVe-Twitter-100 (about 390 MB) into `models/embeddings/`. `gensim` is not needed; the files are parsed directly.
4. Saved models:
   - `models/tweet_classifier.joblib`: Part 1 model, a dict with keys `vectorizer` and `model`.
   - `models/tweet_classifier_glove.joblib` and `models/glove100_subset.npz`: Part 2 model (TF-IDF vectorizer, scaler, classifier) and the GloVe vectors for the words in this dataset.

Using the Part 1 model on a new tweet:

```python
import re, joblib

leak = re.compile(r"#(?!ironing\b)\w*(?:sarcas|iron[iy])\w*", flags=re.IGNORECASE)

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", " and ")
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

bundle = joblib.load("models/tweet_classifier.joblib")
vec, model = bundle["vectorizer"], bundle["model"]

tweet = "Oh great, another Monday. Just what I needed."
probs = model.predict_proba(vec.transform([clean_tweet(leak.sub("", tweet))]))[0]
print(dict(zip(model.classes_, probs.round(2))))
```

---

# Part 1: TF-IDF baseline

## Problem

Given the text of a tweet, predict one of four classes:

| Class | Train | Test |
|---|---|---|
| figurative | 21,238 | 2,044 |
| irony | 20,894 | 2,111 |
| sarcasm | 20,681 | 2,105 |
| regular | 18,595 | 1,859 |

The raw data has 81,408 training and 8,128 test rows with two columns, `tweets` and `class`. The classes are roughly balanced.

## Data cleaning

1. **Missing values and duplicates.** Rows with missing values and duplicate tweets were dropped (train: 49 duplicates; test: 2 missing tweets, 9 missing labels, 1 duplicate).
2. **Label leakage.** Many tweets contain the label itself as a hashtag (`#irony`, `#ironic`, `#sarcasm`, `#sarcastic`). A model could score well just by reading the hashtag, so these hashtags (and variants such as `#sarcasmo` or `#ironictiming`) are removed with a regular expression before anything else. `#ironing` is kept because it is a different word.
3. **Empty tweets.** Tweets that were only a label hashtag became empty after step 2 and were removed.
4. **Text cleaning.** Lowercase; remove links and @mentions; replace `&amp;` with "and"; drop the `#` sign but keep the hashtag word; keep only letters and apostrophes; collapse extra spaces.

After cleaning: **81,342 training tweets and 8,117 test tweets**.

## Findings from exploring the data

- **The `figurative` class is hard to separate.** Before label hashtags were removed, `figurative` tweets carried either an irony tag (10,610) or a sarcasm tag (10,614) in about equal numbers, while `irony` tweets carried only irony tags, `sarcasm` tweets only sarcasm tags, and `regular` tweets none. So `figurative` overlaps with the other two classes, and once the hashtags are removed there is little left to tell them apart.
- **`regular` is easy.** These tweets are very different in content (links, topic hashtags, news-like text), and every model reaches about 0.96 F1 on this class.
- **Style features did not help.** Ten simple features (word and character counts, number of `!`, `?`, `...`, capitalised words, hashtags, mentions, links and non-ASCII characters) were added to TF-IDF. Validation accuracy stayed at 0.526, so they were not used in the final model.

## Experiments

All numbers are validation accuracy on a fixed 80/20 stratified split of the training data (`random_state=42`).

| Experiment | Validation accuracy |
|---|---|
| TF-IDF (50,000 features, 1-2-grams) + Logistic Regression (baseline) | 0.526 |
| Baseline + 10 style features | 0.526 |
| Same features, Linear SVM (default C) | 0.482 |
| Same features, Complement Naive Bayes | 0.508 |
| Logistic Regression, stronger regularisation (C=0.1) | 0.559 |
| Logistic Regression, C=0.1, 5,000 TF-IDF features | 0.570 |
| Word + character TF-IDF + style features, Logistic Regression | 0.569 |
| Word + character TF-IDF, Logistic Regression (C=0.05) | 0.572 |
| 5,000 word features + VADER sentiment, Logistic Regression | 0.569 |
| 5,000 word features, **Linear SVM (C=0.01)** | **0.579** |

Main lesson: the baseline overfits. With C=10 the training accuracy was 0.845 while validation accuracy fell to 0.470, and as C was lowered, validation accuracy rose. Fewer features and stronger regularisation worked better than a bigger feature set. Character n-grams, style features and sentiment scores added nothing beyond noise.

## Final Part 1 model

- Features: TF-IDF on word 1-2-grams, `min_df=3`, 5,000 features.
- Classifier: Linear SVM (C=0.01), wrapped in `CalibratedClassifierCV` so it outputs class probabilities.
- Trained on all of the training data, evaluated once on the test set.

**Test accuracy: 0.584**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| figurative | 0.29 | 0.14 | 0.19 | 2,043 |
| irony | 0.50 | 0.64 | 0.56 | 2,110 |
| regular | 0.92 | 0.99 | 0.96 | 1,859 |
| sarcasm | 0.51 | 0.60 | 0.55 | 2,105 |
| macro avg | 0.56 | 0.59 | 0.57 | 8,117 |
| weighted avg | 0.55 | 0.58 | 0.56 | 8,117 |

## Example outputs (Part 1 model)

| Tweet | True label | Model | Confidence |
|---|---|---|---|
| Oh great, another Monday. Just what I needed. | sarcasm | sarcasm | 72% |
| Love waiting two hours for a delayed flight | sarcasm | sarcasm | 62% |
| Says you hiding behind a fake name | irony | irony | 63% |
| Reading the latest news on the economy http://t.co/abc #politics | regular | regular | 95% |

---

# Part 2: Improving the project with word embeddings

## What changed

TF-IDF treats every word as an unrelated column. Word embeddings give each word a dense vector learned from large amounts of text, so words used in similar contexts get similar vectors. This part tests whether that helps the tweet classifier. The data, cleaning, split and metrics are the same as in Part 1.

## Embeddings used

**GloVe-Twitter-100** (pre-trained on Twitter text, 100 dimensions).

Why this choice: the data is tweets, so embeddings trained on tweets cover slang, abbreviations and informal spelling better than embeddings trained on news or Wikipedia. I first tried the 25-dimension version and then moved to 100 dimensions, which gave a large gain on its own.

How each tweet becomes a vector: each cleaned tweet is split into words, each word is looked up in GloVe, and the tweet is represented by the **mean and max** of its word vectors (200 features), standardised with `StandardScaler`. Words not in GloVe are skipped. GloVe covered 40,419 of 63,369 vocabulary words (about 64%) in the training data.

Two ways of using the embeddings were tested:

1. **Replace** TF-IDF with embeddings.
2. **Extend** TF-IDF by appending the embedding features (weight 0.1) to the 5,000 TF-IDF features, then Linear SVM (C=0.03).

## Results

Settings were chosen on the validation split; the test data was used once at the end.

**Validation accuracy (while tuning)**

| Setup | Validation accuracy |
|---|---|
| TF-IDF + Linear SVM (Part 1) | 0.579 |
| GloVe-25 alone | 0.438 |
| GloVe-100 alone (scaled, mean + max pooling) | 0.506 |
| TF-IDF + GloVe-25 | 0.582 |
| TF-IDF + GloVe-100 | 0.581 |

**Final test accuracy**

| Setup | Test accuracy |
|---|---|
| Before: TF-IDF + Linear SVM | 0.584 |
| GloVe-100 alone | 0.500 |
| After: TF-IDF + GloVe-100 | **0.591** |

**Per-class results on the test set, before and after**

| Class | Precision (before / after) | Recall (before / after) | F1 (before / after) |
|---|---|---|---|
| figurative | 0.29 / 0.26 | 0.14 / 0.10 | 0.19 / 0.15 |
| irony | 0.50 / 0.51 | 0.64 / 0.66 | 0.56 / 0.58 |
| regular | 0.92 / 0.91 | 0.99 / 1.00 | 0.96 / 0.95 |
| sarcasm | 0.51 / 0.52 | 0.60 / 0.64 | 0.55 / 0.57 |

Weighted average F1: 0.56 before, 0.55 after.

**Example outputs, before and after**

| Tweet | True label | Before | After |
|---|---|---|---|
| Oh great, another Monday. Just what I needed. | sarcasm | sarcasm | sarcasm |
| Love waiting two hours for a delayed flight | sarcasm | sarcasm | sarcasm |
| Says you hiding behind a fake name | irony | irony | irony |
| Reading the latest news on the economy http://t.co/abc #politics | regular | regular | regular |

On clear-cut examples both versions agree. The difference between them shows up on the 686 test tweets (8.5%) where exactly one of the two versions is correct.

**Where the two versions disagree (test set, 8,117 tweets)**

| | Count |
|---|---|
| After correct, before wrong | 369 |
| Before correct, after wrong | 317 |
| Net gain | +52 tweets (about +0.6 percentage points) |

Examples where the new version is right and the old one is wrong (label hashtags are removed before the model sees the text):

| Tweet | True label | Before | After |
|---|---|---|---|
| When one of your tweets gets more favorites than you have followers... #irony | figurative | irony | figurative |
| Terrelle Pryor and Braxton Miller Both Wide Receivers Now! #Ironic | figurative | irony | figurative |

Examples where the old version is right and the new one is wrong:

| Tweet | True label | Before | After |
|---|---|---|---|
| Quick rt that throwing money vine I've not seen it 30 times in the last 30 seconds #sarcasm | figurative | figurative | sarcasm |
| @theandyboyd You say, via twitter. #irony | figurative | figurative | irony |

Reading these examples: the disagreements are mostly on tweets that read as irony or sarcasm but carry the `figurative` label, which is the ambiguous class described in Part 1. The examples are the first few in file order and the test file appears to be sorted by class, so they are not a random sample. A McNemar-style comparison of the two counts (369 vs 317) gives z of about 2.0, a borderline result: weak evidence that the improvement is real, not strong evidence.

## What improved, what did not, and why

- **Improved:** irony and sarcasm F1 each rose by about 0.02, and overall test accuracy rose from 0.584 to 0.591. The gain is small, and all validation scores for the combined models were within about 0.003 of TF-IDF alone, so it should be read as a slight improvement, not a clear one. On the test set the new version fixed 369 predictions and broke 317, a net gain of 52 tweets, which is borderline statistically.
- **Embeddings alone were worse than TF-IDF** (0.500 vs 0.584). Averaging word vectors loses word order and rare words, and the signal for this task sits in specific words, hashtags and phrases. With embeddings alone, `regular` tweets dropped from 0.96 to 0.76 F1, because TF-IDF can use distinctive tokens that averaged vectors blur.
- **Higher dimensions helped:** going from 25 to 100 dimensions, with scaling and mean + max pooling, raised embeddings-alone validation accuracy from 0.438 to 0.506.
- **Did not improve:** `figurative` got worse (F1 0.19 to 0.15). This class overlaps heavily with irony and sarcasm, and general-meaning embeddings do not separate them.
- **Why the gain is limited:** GloVe encodes general word meaning, not whether a tweet is ironic or sarcastic, and about 36% of the vocabulary (slang, typos, glued hashtags) has no vector.

## Other embeddings compared (validation set)

After the GloVe experiments, Word2Vec and FastText were also tested, both trained on the tweets themselves and as pre-trained vectors. This was run in Google Colab because `gensim` could not be installed on the local Python 3.14 environment (`notebooks/embeddings_comparison_colab.ipynb`). It uses the same cleaning and the same validation split (16,269 tweets). Each embedding is scored alone (Logistic Regression) and combined with the 5,000 TF-IDF features (Linear SVM, C=0.03) at two embedding weights.

| Embedding | Alone | + TF-IDF (weight 0.05) | + TF-IDF (weight 0.1) |
|---|---|---|---|
| Word2Vec trained on the tweets (100d) | 0.569 | **0.586** | 0.584 |
| FastText trained on the tweets (100d) | 0.555 | 0.580 | 0.580 |
| GloVe-Twitter-100 (pre-trained) | 0.503 | 0.580 | 0.581 |
| FastText wiki-news (pre-trained, 300d) | 0.531 | 0.581 | 0.580 |
| Word2Vec Google News (pre-trained, 300d) | 0.533 | 0.581 | 0.580 |
| TF-IDF alone (reference) | - | 0.579 | 0.579 |

- **Embeddings trained on the tweets beat pre-trained ones when used alone** (0.569 and 0.555 vs 0.503 to 0.533), even though they learn from only about 81,000 tweets. A likely explanation is coverage: the trained models have a vector for every word seen at least twice (32,624 words), while the pre-trained ones cover only 49% to 64% of the tweet vocabulary (GloVe 64%, FastText wiki-news 53%, Google News 49%), and the trained ones also learn word associations specific to this data. This explanation was not tested directly.
- **Combined with TF-IDF, only Word2Vec trained on the tweets stands out:** 0.586, which is +0.007 over TF-IDF alone. The other combinations are within 0.002 of TF-IDF alone, which is noise.
- **FastText trained on the tweets did not beat Word2Vec**, even though it can build vectors for unseen words from subword pieces.
- **Caveats:** the embeddings were trained on the text of all training tweets, including those in the validation split (no labels were used). The best weight was picked on the validation set. The +0.007 is therefore a validation result and still needs one clean check on the test set before it is reported as an improvement over the GloVe model.

## Limitations and next steps

- The labels come from hashtags written by the tweet authors, so they are noisy, and `figurative` is not clearly distinct from `irony` and `sarcasm`. This puts a ceiling on accuracy for any model that only sees the text.
- Averaging word vectors ignores word order. Sentence-level models (for example a small neural network or transformer embeddings) could capture tone better.
- The best validation result came from Word2Vec trained on the tweets themselves (0.586 combined with TF-IDF). Its test-set result is still to be confirmed with one final run before it replaces the GloVe model's 0.591.