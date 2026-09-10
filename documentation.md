# 🛡️ Job Postings Fraud Detection System

> **FraudGuard** — *a data publication that sniffs out fake job postings.*
> Flask + scikit-learn + TF-IDF, served through a pudding.cool-style editorial dashboard.

---

## 📑 Contents

1. [Overview](#1--overview)
2. [How It Works](#2--how-it-works)
3. [Repository Map](#3--repository-map)
4. [Deep Dive](#4--deep-dive)
   - [pipeline\_config.py](#41-pipeline_configpy---read-this-first)
   - [app.py](#42-apppy--the-backend)
   - [Training scripts](#43-training-scripts)
   - [templates/index.html](#44-templatesindexhtml--the-frontend)
   - [Data](#45-data)
   - [Artifacts & config](#46-artifacts--config)
5. [Run & Retrain](#5--run--retrain)
6. [Known Quirks](#6--known-quirks)

---

## 1. 🧭 Overview

Detect fraudulent (**fake**) job postings by combining **natural-language signals** (job description text) with **structured metadata** (experience, industry, employment type, …). Several classifiers are trained, compared with live ROC / confusion-matrix diagnostics, and served for real-time prediction — typed in by hand or scraped live from Internshala.

**Pipeline at a glance:**

```
raw CSV ──(wrangling)──▶ CleanData.csv ──(TF-IDF + encoding + scaling)──▶ 411 features
                                                                                │
                                    ┌───────────────┬───────────────┬───────────┘
                                    ▼               ▼               ▼
                              logistic reg.   decision tree   github RF/DT (text-only)
                                    │               │               │
                                    └───────┬───────┴───────┬───────┘
                                            ▼               ▼
                                   Flask API (app.py)  ◀──  pudding-style UI
```

**Live user flows:**

| Flow | Path |
|---|---|
| 📊 Dashboard | Charts + per-model ROC curves & confusion matrices |
| ✍️ Manual prediction | Fill form / load template → verdict + confidence |
| 🕷️ Scraper prediction | Paste Internshala URL → scraped summary + verdict |

---

## 2. ⚙️ How It Works

### 2.1 Feature engineering — one shared pipeline (`pipeline_config.py`)

Main models (logistic regression + decision tree) share **one** pipeline so inference always matches training:

| Step | Detail |
|---|---|
| 📝 Text | Join `job_description + requirements + benefits + company_profile` (`TEXT_COLS` order) → TF-IDF (`500` terms, english stops, `(1,2)`-grams, `min_df=5`, `max_df=0.95`) → **402 terms** |
| 🧱 Structured | 3 numerics + 6 LabelEncoded categoricals (`STRUCTURED_COLS` order), StandardScaled → **9 numbers**. Leakage-free by design: excludes `text_length`, `is_gmail`, `has_logo` |
| 🧩 Stack | `scipy.sparse.hstack([structured, tfidf])` → **9 + 402 = 411 columns** |

GitHub models use a **separate text-only** pipeline: `CountVectorizer((1,2), min_df=2, max_df=0.95)` over a single `text` column. No structured features; its constants intentionally live in `train_github_models.py`.

### 2.2 Training — 80/20 stratified split (`TEST_SIZE` / `RANDOM_STATE`)

| Script | Produces |
|---|---|
| `LogisticRegression_03.py` | Calibrated logistic regression → `lr_model.pkl` |
| `dtrees.py` | Balanced decision tree (`max_depth=8`) → `dtres_model.pkl` |
| `train_github_models.py` | Balanced RF (200 trees) + DT on external corpus → `github_*.pkl` |

All three merge accuracy / precision / recall / F1 / ROC-AUC / CV scores + top features into `model_metrics.json` (kept as a **training log**). ROC curves and confusion matrices are rendered **live** in the dashboard — no static images.

### 2.3 Serving — `app.py` (Flask, port `5000`)

- Loads all pickles once at startup (a missing file just means that model is absent).
- `_build_feature_vector()` is the **single inference path** for manual + scraper predictions.
- `_predict_with_model()` guards every prediction with a feature-width check and a clear retraining hint.
- `GET /api/evaluation` rebuilds the training split with the **loaded** transforms and scores every model (cached in memory).

### 2.4 What was removed

Dead code from an earlier bulk-CSV / comparison-tab era was stripped: `/api/predict-csv`, `/api/download-predictions`, `/api/metrics`, the frontend bulk + comparison helpers, `temp_predictions/`, and the static `confusion_matrix_lr.png` / `roc_curve_lr.png` (plus the matplotlib code that made them). Every remaining `fetch()` in the UI maps to a live route.

---

## 3. 🗂️ Repository Map

| File | What it is |
|---|---|
| `app.py` | Flask server: model loading, live `/api/*` routes, shared feature builder, guarded predictions, Internshala scraper, evaluation |
| `pipeline_config.py` | ⭐ Single source of truth: column lists, TF-IDF params, split constants. Imported by `app.py` + both main training scripts |
| `templates/index.html` | Entire frontend (pudding-style UI + all JS). No dead screens |
| `dtrees.py` | Trains the decision tree on the shared 411-feature pipeline |
| `LogisticRegression_03.py` | Trains calibrated LR; prints report/metrics; interactive CLI predictor. No plot output |
| `train_github_models.py` | Trains `github_rf` + `github_dt` (text-only, own vectorizer) |
| `model_metrics.json` | Training log per model (metrics + CV + top features) |
| `CleanData.csv` | Cleaned training data (**3000 × 20**) |
| `fake_real_job_postings_3000x25.csv` | Raw dataset (**3000 × 25**, provenance for the cleaning notebook) |
| `github_fakejobs.csv` | External text+label corpus for the GitHub pair |
| `DataWrangling_01.ipynb` | Cleaning notebook (raw → `CleanData.csv`) |
| `EDA_02.ipynb` | Exploratory analysis notebook |
| `label_encoders.pkl` | 6 `LabelEncoder`s 🔒 gitignored — regenerate by training |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF (402 terms) 🔒 gitignored |
| `scaler.pkl` | `StandardScaler` for the 9 structured features 🔒 gitignored |
| `lr_model.pkl` | Calibrated LR (expects 411 features) 🔒 gitignored |
| `dtres_model.pkl` | Decision tree (expects 411 features) 🔒 gitignored |
| `github_vectorizer.pkl` | `CountVectorizer` for GitHub models 🔒 gitignored |
| `github_rf_model.pkl` / `github_dt_model.pkl` | GitHub models 🔒 gitignored |
| `requirements.txt` | Pinned Python dependencies |
| `README.md` | Older overview (partially outdated — this file is current) |
| `.gitignore` | `venv/`, `.venv/`, `.vscode/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `*.pkl` |
| `documentation.md` | This file |

---

## 4. 🔬 Deep Dive

### 4.1 `pipeline_config.py` — ⭐ read this first

Pure-Python constants, zero third-party imports:

```python
TEXT_COLS        = ["job_description", "requirements", "benefits", "company_profile"]
CATEGORICAL_COLS = ["industry", "employment_type", "salary_range",
                    "education_level", "department", "job_function"]
STRUCTURED_COLS  = ["required_experience_years", "num_open_positions",
                    "telecommuting", "industry_enc", ...]   # 9 total
TFIDF_PARAMS     = {"max_features": 500, "stop_words": "english",
                    "ngram_range": (1, 2), "min_df": 5, "max_df": 0.95}
TEST_SIZE, RANDOM_STATE = 0.20, 42
EXPECTED_MAIN_FEATURES  = 411  # 9 + 402
```

> **Rule:** change the pipeline **once, here** — then retrain. Never hardcode a second copy of these values (a past hardcoded drift, 336 vs 402 TF-IDF terms, caused the *"X has 411 features, expecting 345"* outage).

### 4.2 `app.py` — the backend

**Startup.** Loads encoders, TF-IDF, scaler (optional), the 4 model pickles and the GitHub vectorizer. `github_vectorizer` defaults to `None` even on failure, so later `is None` checks never raise `NameError`. `_eval_cache` memoizes evaluation.

**`_build_feature_vector(data, model_name)`.** Builds joined text in `TEXT_COLS` order; GitHub models take the `CountVectorizer` path, main models take TF-IDF + safe-encoded `CATEGORICAL_COLS` (unknown → `0`) + scaled 9-vector (asserted against `STRUCTURED_COLS`) → `(1, 411)`.

**`_predict_with_model(model, matrix)`.** Compares `matrix.shape[1]` to `model.n_features_in_` and raises a *"retrain with the shared pipeline (see `pipeline_config.py`)"* hint on drift. Used by **both** prediction routes.

**Live routes** (each is called by the frontend):

| Method & route | Purpose |
|---|---|
| `GET /` | Renders `index.html` |
| `GET /api/options` | Encoder classes for the form dropdowns |
| `POST /api/predict` | `{model, texts, structured, categoricals}` → `{prediction 0/1, probability {real, fake}}`. Tolerates empty/missing JSON |
| `POST /api/scrape-internshala` | `{url, model}` → scrapes JSON-LD/HTML, maps stipend/industry/employment/experience/education/department/function into training categories, sniffs work-from-home, returns verdict + `scraped_data` summary |
| `GET /api/dataset-stats` | Counts, gmail/logo ratios by class, avg text length, top industries → the 4 dashboard charts |
| `GET /api/samples` | First real + first fake row → form templates |
| `GET /api/evaluation` | Rebuilds the shared split with loaded transforms; per model returns `confusion_matrix {labels, matrix [[TN,FP],[FN,TP]]}`, `roc {fpr[], tpr[] (≤120 pts), auc}`, accuracy/precision/recall/F1, `n_test` |

**Scraper helpers** (all live): `SCRAPER_HEADERS`, `_map_stipend_to_salary_range`, `_map_employment_type`, `_map_industry`, `_infer_required_experience_years`, `_infer_education_level`, `_infer_department`, `_infer_job_function`, `_scrape_internshala`.

### 4.3 Training scripts

**`dtrees.py`.** Encodes `CATEGORICAL_COLS`, joins `TEXT_COLS`, scales `STRUCTURED_COLS`, TF-IDFs with shared `TFIDF_PARAMS`, asserts stacked width, splits with shared `TEST_SIZE`/`RANDOM_STATE`, trains `DecisionTreeClassifier(balanced, max_depth=8)`, reports metrics + 5-fold CV + top-10 importances, saves `dtres_model.pkl` and refreshes the shared pickles (byte-identical when configs match), merges metrics.

**`LogisticRegression_03.py`.** Same shared constants; `CalibratedClassifierCV(LogisticRegression(C=0.5), cv=5)`; prints metrics, classification report, top-20 coefficients; saves `lr_model.pkl` + shared pickles; merges metrics. Ships an interactive CLI (`--skip-predict` to skip).

**`train_github_models.py`** *(untouched by the cleanup).* Own `CountVectorizer` pipeline on `github_fakejobs.csv`; balanced RF + DT; 5-fold CV; saves the three GitHub pickles; merges metrics.

### 4.4 `templates/index.html` — the frontend

Single-file app (`<style>` + HTML + `<script>`, Chart.js CDN) in a pudding.cool editorial theme: cream paper + dot grid, ink-black 2px borders, hard 6px shadows, rotated sticker pills, lowercase Source-Serif headlines, Inter body, Plex-Mono kickers.

| Area | Contents |
|---|---|
| Header | Black ticker bar · tagline · wobbly `fraud*guard*` wordmark · Dashboard/Predictor sticker tabs |
| Dashboard | Intro (№ 01) · stat cards · charts fig.1–4 · diagnostics intro (№ 03) with model pills · fig.5 ROC canvas + badges · fig.6 confusion-matrix grid |
| Predictor | Manual/Scraper pills · manual form (every input ID maps to `/api/predict` fields) · template buttons · verdict shield + confidence bar · scraper form + spinner + scraped summary + “fill form” transfer |

JS contains **only live code**: `switchTab`, `switchPredictorMode`, `fetchDropdownOptions`, `fetchSamples` / `loadSample` / `resetResultPanel`, `submitPrediction`, `submitScraperPrediction` / `transferScrapedToManual`, `loadStats` + `loadEvaluation` / `renderEvaluation`.

### 4.5 Data

- `CleanData.csv` — **3000 × 20**: `job_title`, `job_description`, `requirements`, `benefits`, `company_profile`, `industry`, `employment_type`, `salary_range`, `required_experience_years`, `education_level`, `department`, `posting_date`, `contact_email`, `has_logo`, `num_open_positions`, `job_function`, `telecommuting`, `text_length`, **`is_fake`** (`0` = REAL, `1` = FAKE; ≈1528 / ≈1472), `application_deadline`.
- `github_fakejobs.csv` — external text/label corpus (held-out `n=347` in `/api/evaluation`).
- Notebooks are kept as pipeline provenance; the `.py` scripts are the runnable truth.

### 4.6 Artifacts & config

- `*.pkl` are **gitignored**: a fresh clone has no models — run the three training scripts first.
- `model_metrics.json` is a write-only training log (CV stats + importances the live endpoint doesn't compute).
- `requirements.txt`: Flask 3.1, scikit-learn 1.9, scipy, pandas 3, beautifulsoup4, requests, plus matplotlib/seaborn (EDA notebook only — no training script plots anymore).

---

## 5. 🚀 Run & Retrain

```bash
python app.py                                  # serve → http://127.0.0.1:5000
python dtrees.py                               # retrain decision tree
python LogisticRegression_03.py --skip-predict # retrain logistic regression
python train_github_models.py                  # retrain GitHub pair
```

---

## 6. ⚠️ Known Quirks

1. **LR scale asymmetry — FIXED.** LR used to train on *unscaled* structured values while `app.py` scaled at inference. Both scripts now fit the same `StandardScaler` on train rows and persist `scaler.pkl`, so train == serve.
2. **Synthetic-text leakage — FIXED via real-text grafting.** `CleanData` text was template salad keyed by artifacts (`global` opened every fake profile, `company` every real one), so the tree collapsed to a depth-2 stump predicting REAL @ 1.0 on any real-world input. The main pipeline now trains on `load_main_frame()` (`pipeline_config.py`): CleanData structured fields + label-matched real postings from `github_fakejobs.csv` (truncated to wild-like length), with leakage tokens stripped and a group-aware split (shared by both training scripts and `/api/evaluation`). Retraining changed expectations honestly: DT ~0.79 / LR ~0.89 accuracy instead of fake 1.0s. De-leakage still must be applied to **both** main models together, never just one.
