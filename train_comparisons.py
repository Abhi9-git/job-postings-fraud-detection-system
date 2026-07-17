import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from scipy.sparse import hstack, csr_matrix

print("Starting training of both Baseline and Robust (Leakage-Free) models...")

# 1. Load the cleaned data
df = pd.read_csv("CleanData.csv")
df["is_gmail"] = df["contact_email"].apply(lambda x: 1 if "gmail" in str(x).lower() else 0)

# Label Encoders
label_encoders = {}
categorical_cols = ["industry", "employment_type", "salary_range", "education_level", "department", "job_function"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Text setup
text_cols = ["job_description", "requirements", "benefits", "company_profile"]
df["combined_text"] = df[text_cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

y = df["is_fake"]

# Define Feature Sets
# Baseline structured features (includes leakages)
baseline_structured_cols = [
    "required_experience_years", "has_logo", "num_open_positions", "telecommuting",
    "text_length", "is_gmail", "industry_enc", "employment_type_enc",
    "salary_range_enc", "education_level_enc", "department_enc", "job_function_enc"
]

# Robust structured features (drops leakages: text_length and is_gmail)
robust_structured_cols = [
    "required_experience_years", "has_logo", "num_open_positions", "telecommuting",
    "industry_enc", "employment_type_enc", "salary_range_enc",
    "education_level_enc", "department_enc", "job_function_enc"
]

# ==========================================
# 2. Train BASELINE Models (with leakage)
# ==========================================
print("\n--- Training Baseline Models (with Leakage) ---")
tfidf_baseline = TfidfVectorizer(
    max_features=500, stop_words="english", ngram_range=(1, 2), min_df=5, max_df=0.95
)
tfidf_baseline_matrix = tfidf_baseline.fit_transform(df["combined_text"])

X_struct_base = df[baseline_structured_cols].values
X_combined_base = hstack([csr_matrix(X_struct_base), tfidf_baseline_matrix])

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_combined_base, y, test_size=0.20, random_state=42, stratify=y
)

baseline_classifiers = {
    "logisticregression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12),
    "decisiontree": DecisionTreeClassifier(random_state=42, max_depth=8)
}

baseline_metrics = {}
for name, clf in baseline_classifiers.items():
    print(f"Training Baseline {name}...")
    clf.fit(X_train_b, y_train)
    
    y_pred = clf.predict(X_test_b)
    y_prob = clf.predict_proba(X_test_b)[:, 1]
    
    baseline_metrics[name] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    # Save baseline model
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(clf, f)

# ==========================================
# 3. Train ROBUST Models (Leakage-Free)
# ==========================================
print("\n--- Training Robust Models (Leakage-Free) ---")
# Use custom stop words to filter out target leakage terms "global" and "company"
custom_stops = ["global", "company"]
robust_stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stops))

tfidf_robust = TfidfVectorizer(
    max_features=500, stop_words=robust_stop_words, ngram_range=(1, 2), min_df=5, max_df=0.95
)
tfidf_robust_matrix = tfidf_robust.fit_transform(df["combined_text"])

X_struct_rob = df[robust_structured_cols].values
X_combined_rob = hstack([csr_matrix(X_struct_rob), tfidf_robust_matrix])

X_train_r, X_test_r, _, _ = train_test_split(
    X_combined_rob, y, test_size=0.20, random_state=42, stratify=y
)

robust_classifiers = {
    "logisticregression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12),
    "decisiontree": DecisionTreeClassifier(random_state=42, max_depth=8)
}

robust_metrics = {}
for name, clf in robust_classifiers.items():
    print(f"Training Robust {name}...")
    clf.fit(X_train_r, y_train)
    
    y_pred = clf.predict(X_test_r)
    y_prob = clf.predict_proba(X_test_r)[:, 1]
    
    robust_metrics[name] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    # Save robust model
    with open(f"{name}_safe_model.pkl", "wb") as f:
        pickle.dump(clf, f)

# ==========================================
# 4. Save Pipeline Components & Metrics
# ==========================================
# Save global encoders and vectorizers
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_baseline, f)
with open("tfidf_vectorizer_safe.pkl", "wb") as f:
    pickle.dump(tfidf_robust, f)

# Save combined metrics JSON
metrics_output = {
    "baseline": baseline_metrics,
    "robust": robust_metrics
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics_output, f, indent=4)

print("\nModel metrics saved to model_metrics.json!")
print("Baseline and Robust pipelines successfully compiled and saved!")
