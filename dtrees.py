import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from scipy.sparse import hstack, csr_matrix
from pipeline_config import (
    CATEGORICAL_COLS,
    RANDOM_STATE,
    STRUCTURED_COLS,
    TEST_SIZE,
    TEXT_COLS,
    TFIDF_PARAMS,
)

print("Starting training of Decision Tree Model Pipeline...")

# 1. Load the cleaned data

df = pd.read_csv("CleanData.csv")

# Label encoders (columns shared with app.py via pipeline_config)
label_encoders = {}
categorical_cols = CATEGORICAL_COLS
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Text setup
text_cols = TEXT_COLS
df["combined_text"] = df[text_cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

y = df["is_fake"]

# Leakage-free structured features (order shared with app.py)
structured_cols = STRUCTURED_COLS

scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(df[structured_cols])

# Shared TF-IDF config (see pipeline_config). Do NOT hardcode a different
# config here: it overwrites the shared tfidf_vectorizer.pkl with an
# incompatible vocab and breaks inference with "X has N features, but
# DecisionTreeClassifier is expecting M".
tfidf = TfidfVectorizer(**TFIDF_PARAMS)
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

X_combined = hstack([csr_matrix(X_struct_scaled), tfidf_matrix])
assert X_combined.shape[1] == 9 + len(tfidf.vocabulary_), (
    f"Unexpected feature width {X_combined.shape[1]}"
)

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=10,
    class_weight="balanced"
)

print("Training decision tree...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_combined, y, cv=cv, scoring="accuracy")

metrics = {
    "decisiontree": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std())
    }
}

# Feature importance transparency
importances = clf.feature_importances_
struct_names = structured_cols
tfidf_names = tfidf.get_feature_names_out().tolist()
all_names = struct_names + tfidf_names
top_indices = np.argsort(importances)[-10:][::-1]
metrics["decisiontree"]["top_features"] = [
    {"feature": all_names[i], "importance": float(importances[i])}
    for i in top_indices
]

# Save model and supporting pipeline artifacts
with open("dtres_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Merge with existing metrics if present
try:
    with open("model_metrics.json", "r") as f:
        existing_metrics = json.load(f)
except FileNotFoundError:
    existing_metrics = {}

existing_metrics["decisiontree"] = metrics["decisiontree"]

with open("model_metrics.json", "w") as f:
    json.dump(existing_metrics, f, indent=4)

print("\nDecision tree retrained and saved successfully.")
print(f"Saved model: dtres_model.pkl")
print(f"Accuracy: {metrics['decisiontree']['accuracy']:.4f}")
print(f"F1: {metrics['decisiontree']['f1']:.4f}")
print(f"CV Accuracy: {metrics['decisiontree']['cv_accuracy_mean']:.4f} ± {metrics['decisiontree']['cv_accuracy_std']:.4f}")
