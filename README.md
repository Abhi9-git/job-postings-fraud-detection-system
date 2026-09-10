# 🛡️ FraudGuard — Fake Job Posting Detection

> *A data publication that sniffs out fake job postings.*
> Machine learning (TF-IDF + scikit-learn classifiers) served through a pudding.cool-style editorial dashboard.

Predict whether a job posting is **REAL** or **FAKE** — type one in by hand, or paste an **Internshala** URL and let the live scraper + classifier deliver a verdict with confidence.

---

## ✨ Features

| | |
|---|---|
| 📊 **Dashboard** | Class balance, gmail/logo rates, text length — plus live **ROC curves & confusion matrices** per model |
| ✍️ **Manual predictor** | Full posting form with real/fake templates |
| 🕷️ **Internshala scraper** | Pastes a listing URL → scrapes JSON-LD/HTML → auto-maps fields → verdict |
| 🧠 **4 classifiers** | Logistic Regression, Decision Tree (shared 411-feature pipeline) + GitHub Random Forest & Decision Tree (text-only) |

---

## 🚀 Quickstart

```bash
# 1. Activate the environment
source .venv/bin/activate        # (or venv/ on some setups)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (pickles are gitignored, so a fresh clone needs this)
python train_github_models.py
python LogisticRegression_03.py --skip-predict
python dtrees.py

# 4. Serve the dashboard
python app.py
```

Open 👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Project Structure

```text
.
├── app.py                        # Flask server: API routes, feature builder, scraper
├── pipeline_config.py            # ⭐ Shared pipeline constants (see note below)
├── templates/index.html          # Entire frontend (editorial UI + all JS)
├── dtrees.py                     # Trains the decision tree (shared pipeline)
├── LogisticRegression_03.py      # Trains calibrated LR (+ interactive CLI)
├── train_github_models.py        # Trains the GitHub text-only pair
├── model_metrics.json            # Training log (metrics + CV + top features)
├── CleanData.csv                 # Cleaned training data (3000 × 20)
├── fake_real_job_postings_3000x25.csv  # Raw dataset (3000 × 25)
├── github_fakejobs.csv           # External text+label corpus
├── DataWrangling_01.ipynb        # Cleaning notebook (raw → CleanData)
├── EDA_02.ipynb                  # Exploratory analysis notebook
├── requirements.txt              # Pinned dependencies
├── documentation.md              # 📖 Full documentation (start here for details)
└── *.pkl                         # Trained artifacts (gitignored — retrain to create)
```

> **Pipeline rule:** `pipeline_config.py` is the single source of truth for text columns, categorical columns, structured-feature order, TF-IDF params and the train/test split. `app.py`, `dtrees.py` and `LogisticRegression_03.py` all import from it — change the pipeline **once, there**, then retrain. Never hardcode a second copy.

---

## ⚙️ How Detection Works

1. **NLP text vectorization** — each training row's text is a label-matched real posting grafted from `github_fakejobs.csv` (see `load_main_frame()`), transformed with **TF-IDF** (top 2000 unigrams & bigrams) → 2000 terms.
2. **Structured encoding** — experience, openings, telecommuting + 6 LabelEncoded categoricals (`industry`, `employment_type`, `salary_range`, `education_level`, `department`, `job_function`), StandardScaled → 9 numbers. Leakage-free by design: excludes `text_length`, gmail/logo signals.
3. **Stacked matrix** — `scipy.sparse.hstack([structured, tfidf])` → **2009 features** per posting.
4. **Classification** — the selected model returns REAL/FAKE plus calibrated confidence; every prediction is width-guarded against vectorizer/model drift.

---

## 📊 Models

| Model | Pipeline | Notes |
|---|---|---|
| Logistic Regression | Shared 2009-feature | Calibrated (`C=0.5`, 5-fold), linear + interpretable coefficients |
| Decision Tree | Shared 2009-feature | Balanced, `max_depth=12`, fast rule-based verdicts |
| GitHub Random Forest | Text-only CountVectorizer | 200-tree ensemble on the external corpus |
| GitHub Decision Tree | Text-only CountVectorizer | Lightweight tree on the external corpus |

Live ROC curves, confusion matrices and accuracy/precision/recall/F1/AUC for all four are computed on the held-out split and shown on the dashboard (`GET /api/evaluation`).

---

## 🔌 API

| Method & route | Purpose |
|---|---|
| `GET /` | Dashboard UI |
| `GET /api/options` | Dropdown categories for the form |
| `GET /api/dataset-stats` | Corpus stats for the charts |
| `GET /api/samples` | Real/fake form templates |
| `GET /api/evaluation` | Confusion matrices + ROC points per model |
| `POST /api/predict` | Single-posting verdict `{prediction, probability}` |
| `POST /api/scrape-internshala` | Scrape an Internshala URL → verdict + summary |

---

## 📖 Further Reading

- **`documentation.md`** — full documentation: architecture, every file, deep dives, known quirks.
- *Fixed quirk:* `CleanData` text was synthetic template salad (`global` = every fake, `company` = every real), which collapsed the decision tree to always-REAL @ 1.0 on real input. The main pipeline now trains on grafted real-world text (`load_main_frame()` in `pipeline_config.py`) — DT ~0.79 / LR ~0.89, honest (see `documentation.md`).
