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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import warnings, pickle

warnings.filterwarnings("ignore")

# ── 1. Load the cleaned data ────────────────────────────────────────
print("=" * 60)
print("  STEP 1 : Loading CleanData.csv")
print("=" * 60)

df = pd.read_csv("CleanData.csv")
print(f"Shape  : {df.shape}")
print(f"Target : \n{df['is_fake'].value_counts()}\n")

# ── 2. Feature Engineering ──────────────────────────────────────────
print("=" * 60)
print("  STEP 2 : Feature Engineering (Structured + NLP)")
print("=" * 60)

# ─── 2a. Structured features ────────────────────────────────────────
# Email domain — gmail vs company  (strong signal from EDA)
df["is_gmail"] = df["contact_email"].apply(
    lambda x: 1 if "gmail" in str(x).lower() else 0
)

# Encode categorical columns with LabelEncoder
label_encoders = {}
categorical_cols = [
    "industry",
    "employment_type",
    "salary_range",
    "education_level",
    "department",
    "job_function",
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  Encoded  {col:25s}  →  {list(le.classes_)}")

structured_feature_cols = [
    "required_experience_years",
    "has_logo",
    "num_open_positions",
    "telecommuting",
    "text_length",
    "is_gmail",
    "industry_enc",
    "employment_type_enc",
    "salary_range_enc",
    "education_level_enc",
    "department_enc",
    "job_function_enc",
]

# ─── 2b. NLP features — TF-IDF on text columns ─────────────────────
text_cols = ["job_description", "requirements", "benefits", "company_profile"]

# Combine all text columns into a single text feature
df["combined_text"] = df[text_cols].apply(
    lambda row: " ".join(row.values.astype(str)), axis=1
)

print(f"\n  Text columns combined: {text_cols}")
print(f"  Sample combined text (first 120 chars):")
print(f"    \"{df['combined_text'].iloc[0][:120]}...\"\n")

# Fit TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=500,       # top 500 terms to keep it manageable
    stop_words="english",   # remove common English stop words
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=5,               # term must appear in at least 5 docs
    max_df=0.95,            # ignore terms in > 95% of docs
)

tfidf_matrix = tfidf.fit_transform(df["combined_text"])
print(f"  TF-IDF matrix shape  : {tfidf_matrix.shape}")
print(f"  Top 20 TF-IDF terms  : {tfidf.get_feature_names_out()[:20].tolist()}")
print(f"  Structured features  : {len(structured_feature_cols)}")
print(f"  Total features       : {len(structured_feature_cols)} + {tfidf_matrix.shape[1]} = {len(structured_feature_cols) + tfidf_matrix.shape[1]}\n")

# ─── 2c. Combine structured + TF-IDF features ───────────────────────
X_structured = df[structured_feature_cols].values
from scipy.sparse import csr_matrix

X_combined = hstack([csr_matrix(X_structured), tfidf_matrix])
y = df["is_fake"]

print(f"  Combined X shape : {X_combined.shape}")
print(f"  y shape          : {y.shape}\n")

# ── 3. Train / Test Split ───────────────────────────────────────────
print("=" * 60)
print("  STEP 3 : Train / Test Split  (80-20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples\n")

# ── 4. Fit Logistic Regression ───────────────────────────────────────
print("=" * 60)
print("  STEP 4 : Fitting Logistic Regression Model (with NLP)")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_train, y_train)
print("  Model trained successfully ✓\n")

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
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0],
}).sort_values("Coefficient", key=abs, ascending=False)

print("Top 20 Most Important Features (by |coefficient|):")
print(coef_df.head(20).to_string(index=False))
print()

# ── 5b. Confusion Matrix Plot ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Logistic Regression (NLP)")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png", dpi=150)
plt.show()
print("  Confusion matrix saved → confusion_matrix_lr.png\n")

# ── 5c. ROC Curve Plot ──────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression (NLP)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_lr.png", dpi=150)
plt.show()
print("  ROC curve saved → roc_curve_lr.png\n")

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
print("  Saved → lr_model.pkl, label_encoders.pkl, tfidf_vectorizer.pkl\n")


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
    has_logo   = int(input("Company has logo?  (1=Yes, 0=No)    : "))
    open_pos   = int(input("Number of open positions (e.g. 3)   : "))
    telecomm   = int(input("Telecommuting?     (1=Yes, 0=No)    : "))
    text_len   = int(input("Text length of posting  (e.g. 89)   : "))

    email_type = input("Contact email domain (gmail / company) : ").strip().lower()
    is_gmail   = 1 if email_type == "gmail" else 0

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
    # Structured part
    structured_input = np.array([[
        experience, has_logo, open_pos, telecomm, text_len, is_gmail,
        industry_enc, employment_enc, salary_enc,
        education_enc, department_enc, job_function_enc,
    ]])

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
    while True:
        predict_job_posting()
        again = input("\nPredict another? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye! 👋")
            break
