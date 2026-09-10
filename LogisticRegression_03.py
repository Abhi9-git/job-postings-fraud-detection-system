"""
Fake Job Posting Detection — Logistic Regression with NLP
==========================================================
Uses TF-IDF vectorization on text columns (job_description,
requirements, benefits, company_profile) combined with structured
features to detect fraudulent job postings.
"""

# ── Imports ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from pipeline_config import (
    CATEGORICAL_COLS,
    STRUCTURED_COLS,
    TFIDF_PARAMS,
    grafted_train_test_indices,
    load_main_frame,
)
import os
import sys
import warnings, pickle

warnings.filterwarnings("ignore")

# ── 1. Load the grafted frame ───────────────────────────────────────
# CleanData structured fields + label-matched real-world text (see
# pipeline_config). Never train on CleanData's raw text columns: they are
# synthetic template salad that makes TF-IDF models constant on real input.
print("=" * 60)
print("  STEP 1 : Loading grafted training frame")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = load_main_frame(BASE_DIR)
print(f"Shape  : {df.shape}")
print(f"Target : \n{df['is_fake'].value_counts()}\n")

# ── 2. Feature Engineering ──────────────────────────────────────────
print("=" * 60)
print("  STEP 2 : Feature Engineering (Structured + NLP)")
print("=" * 60)

# ─── 2a. Structured features ────────────────────────────────────────
# Keep the feature schema aligned with app.py so predictions use the same
# live feature set that was actually trained. This avoids leakage-like
# dataset-only signals such as text_length and email/logo indicators.
# Encode categorical columns with LabelEncoder (shared with app.py)
label_encoders = {}
categorical_cols = CATEGORICAL_COLS

for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  Encoded  {col:25s}  →  {list(le.classes_)}")

structured_feature_cols = STRUCTURED_COLS

# ─── 2b. NLP features — TF-IDF on grafted real-world text ───────────
# The frame's combined_text is already stripped/truncated (see
# pipeline_config). The vectorizer is fit on TRAIN rows only (after the
# group-aware split below) so test text never leaks into the vocabulary.
print(f"\n  Text source: grafted real-world postings ({len(df)} rows)")
print(f"  Structured features  : {len(structured_feature_cols)}\n")

# ─── 2c. Group-aware train/test split + fit transforms ──────────────
# Rows sharing grafted text never span train/test (honest metrics).
print("=" * 60)
print("  STEP 3 : Group-aware Train / Test Split (80-20 by text group)")
print("=" * 60)

y = df["is_fake"].astype(int).values
idx_train, idx_test = grafted_train_test_indices(y, df["text_group"].values)

tfidf = TfidfVectorizer(**TFIDF_PARAMS)
tfidf_train = tfidf.fit_transform(df["combined_text"].iloc[idx_train])
tfidf_test = tfidf.transform(df["combined_text"].iloc[idx_test])
print(f"  TF-IDF matrix shape (train) : {tfidf_train.shape}")
print(f"  TF-IDF vocab size           : {len(tfidf.vocabulary_)}")

# StandardScale the structured block: app.py scales structured inputs at
# inference time, so training must use the same transform (previously LR
# trained unscaled while being served scaled — a train/serve skew).
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_struct_train = scaler.fit_transform(df[structured_feature_cols].values[idx_train])
X_struct_test = scaler.transform(df[structured_feature_cols].values[idx_test])
from scipy.sparse import csr_matrix

X_train = hstack([csr_matrix(X_struct_train), tfidf_train])
X_test = hstack([csr_matrix(X_struct_test), tfidf_test])
y_train, y_test = y[idx_train], y[idx_test]

print(f"  Combined X train shape : {X_train.shape}")
print(f"  Combined X test shape  : {X_test.shape}")
print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples\n")

# ── 4. Fit Logistic Regression ───────────────────────────────────────
print("=" * 60)
print("  STEP 4 : Fitting Logistic Regression Model (with NLP)")
print("=" * 60)

base_lr = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
model = CalibratedClassifierCV(
    estimator=base_lr,
    method="sigmoid",
    cv=5,
)
model.fit(X_train, y_train)
print("  Calibrated logistic regression model trained successfully ✓\n")

# ── 5. Evaluation ───────────────────────────────────────────────────
print("=" * 60)
print("  STEP 5 : Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Real (0)", "Fake (1)"]))

# ── 5a. Top NLP Feature Importance ──────────────────────────────────
feature_names = structured_feature_cols + tfidf.get_feature_names_out().tolist()
if hasattr(model, "coef_"):
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", key=abs, ascending=False)

    print("Top 20 Most Important Features (by |coefficient|):")
    print(coef_df.head(20).to_string(index=False))
    print()
else:
    print("Top feature importance is not available for calibrated classifiers; probability calibration is in use.\n")

# ── 5b. ROC summary (curves are rendered live in the dashboard) ──────
print(f"  ROC-AUC on held-out split : {roc_auc:.4f}\n")

# ── 5c. Persist metrics to the shared training log ─────────────────────
import json as _json
try:
    with open("model_metrics.json", "r") as f:
        _existing_metrics = _json.load(f)
except FileNotFoundError:
    _existing_metrics = {}
_existing_metrics["logisticregression"] = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "n_test": int(len(y_test)),
    "text_source": "grafted-real",
}
with open("model_metrics.json", "w") as f:
    _json.dump(_existing_metrics, f, indent=4)
print("  Metrics merged → model_metrics.json\n")

# ── 6. Save model + encoders + tfidf ────────────────────────────────
print("=" * 60)
print("  STEP 6 : Saving Model, Encoders & TF-IDF Vectorizer")
print("=" * 60)

with open("lr_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  Saved → lr_model.pkl, label_encoders.pkl, tfidf_vectorizer.pkl, scaler.pkl\n")


# ══════════════════════════════════════════════════════════════════════
#                    PREDICTION SYSTEM
# ══════════════════════════════════════════════════════════════════════

def predict_job_posting():
    """Interactive NLP-powered prediction system — takes user input and predicts."""

    print("\n" + "=" * 60)
    print("  🔍 FAKE JOB POSTING PREDICTION SYSTEM (NLP-Powered)")
    print("=" * 60)
    print("Enter the details of the job posting below.\n")

    # ── Collect TEXT inputs ──────────────────────────────────────────
    print("── TEXT FIELDS (NLP features) ──")
    job_desc     = input("Job Description     : ").strip()
    requirements = input("Requirements        : ").strip()
    benefits     = input("Benefits            : ").strip()
    company_prof = input("Company Profile     : ").strip()

    combined_text = f"{job_desc} {requirements} {benefits} {company_prof}"

    # ── Collect STRUCTURED inputs ────────────────────────────────────
    print("\n── STRUCTURED FIELDS ──")
    experience = int(input("Required experience (years, e.g. 5) : "))
    open_pos   = int(input("Number of open positions (e.g. 3)   : "))
    telecomm   = int(input("Telecommuting?     (1=Yes, 0=No)    : "))

    # Categorical inputs
    def ask_categorical(col_name, le):
        options = list(le.classes_)
        print(f"  Options for {col_name}: {options}")
        val = input(f"  Enter {col_name}: ").strip()
        return le.transform([val])[0]

    print()
    industry_enc     = ask_categorical("industry", label_encoders["industry"])
    employment_enc   = ask_categorical("employment_type", label_encoders["employment_type"])
    salary_enc       = ask_categorical("salary_range", label_encoders["salary_range"])
    education_enc    = ask_categorical("education_level", label_encoders["education_level"])
    department_enc   = ask_categorical("department", label_encoders["department"])
    job_function_enc = ask_categorical("job_function", label_encoders["job_function"])

    # ── Build combined feature vector ────────────────────────────────
    # Structured part must match the live feature schema used in app.py
    structured_input = scaler.transform(np.array([[
        experience, open_pos, telecomm,
        industry_enc, employment_enc, salary_enc,
        education_enc, department_enc, job_function_enc,
    ]]))

    # NLP part — transform text with the fitted TF-IDF vectorizer
    tfidf_input = tfidf.transform([combined_text])

    # Combine
    input_combined = hstack([csr_matrix(structured_input), tfidf_input])

    # ── Predict ──────────────────────────────────────────────────────
    prediction  = model.predict(input_combined)[0]
    probability = model.predict_proba(input_combined)[0]

    print("\n" + "-" * 50)
    print("📊  PREDICTION RESULT")
    print("-" * 50)
    if prediction == 1:
        print("⚠️   This job posting is likely  ** FAKE **")
    else:
        print("✅   This job posting appears to be  ** REAL **")
    print(f"     Confidence — Real: {probability[0]*100:.1f}%  |  Fake: {probability[1]*100:.1f}%")
    print("-" * 50)


# ── 7. Run Prediction System ────────────────────────────────────────
if __name__ == "__main__":
    if "--skip-predict" in sys.argv:
        print("\nTraining complete. Skipped interactive prediction mode.\n")
    else:
        while True:
            predict_job_posting()
            again = input("\nPredict another? (y/n): ").strip().lower()
            if again != "y":
                print("\nGoodbye! 👋")
                break
