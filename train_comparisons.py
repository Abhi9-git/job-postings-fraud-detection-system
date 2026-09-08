import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

print("Starting training of Leakage-Free ML Model Pipeline...")

# 1. Load the cleaned data
df = pd.read_csv("CleanData.csv")

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

# Leakage-Free structured features:
# Excludes text_length, is_gmail, and has_logo to eliminate synthetic target leakage
structured_cols = [
    "required_experience_years", "num_open_positions", "telecommuting",
    "industry_enc", "employment_type_enc", "salary_range_enc",
    "education_level_enc", "department_enc", "job_function_enc"
]

scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(df[structured_cols])

# Expanded stop words: filter out synthetic text pattern triggers
custom_stops = [
    "global", "company", "our company", "we are global",
    "our", "we are", "are global"
]
stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stops))

tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words=stop_words,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True
)
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

X_combined = hstack([csr_matrix(X_struct_scaled), tfidf_matrix])

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.20, random_state=42, stratify=y
)

# Regularized classifiers
classifiers = {
    "logisticregression": LogisticRegression(
        max_iter=1000, random_state=42,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs"
    ),
    "randomforest": RandomForestClassifier(
        n_estimators=80,
        random_state=42,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt"
    ),
    "decisiontree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
        min_samples_leaf=15,
        min_samples_split=30
    )
}

# 5-fold Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    cv_scores = cross_val_score(clf, X_combined, y, cv=cv, scoring="accuracy")
    
    metrics[name] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std())
    }
    
    # Feature importance transparency
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        struct_names = structured_cols
        tfidf_names = tfidf.get_feature_names_out().tolist()
        all_names = struct_names + tfidf_names
        top_indices = np.argsort(importances)[-10:][::-1]
        metrics[name]["top_features"] = [
            {"feature": all_names[i], "importance": float(importances[i])}
            for i in top_indices
        ]
    elif hasattr(clf, "coef_"):
        coefs = np.abs(clf.coef_[0])
        struct_names = structured_cols
        tfidf_names = tfidf.get_feature_names_out().tolist()
        all_names = struct_names + tfidf_names
        top_indices = np.argsort(coefs)[-10:][::-1]
        metrics[name]["top_features"] = [
            {"feature": all_names[i], "importance": float(coefs[i])}
            for i in top_indices
        ]
    
    # Save model file
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(clf, f)

# Save Pipeline Encoders, Scaler, and Vectorizers
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save metrics JSON
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n========== RESULTS ==========")
for name, m in metrics.items():
    print(f"  {name}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, CV={m['cv_accuracy_mean']:.4f}±{m['cv_accuracy_std']:.4f}")

print("\nModel metrics saved to model_metrics.json!")
print("Model pipeline successfully compiled and saved!")
