import pickle
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
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
    STRUCTURED_COLS,
    TFIDF_PARAMS,
    grafted_train_test_indices,
    load_main_frame,
)

print("Starting training of Decision Tree Model Pipeline...")

# 1. Load the grafted frame: CleanData structured fields + label-matched
# real-world text (see pipeline_config). Never read CleanData text directly:
# it is synthetic template salad that makes the tree a constant function.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = load_main_frame(BASE_DIR)

# Label encoders (columns shared with app.py via pipeline_config)
label_encoders = {}
categorical_cols = CATEGORICAL_COLS
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

y = df["is_fake"].astype(int).values
groups = df["text_group"].values

# Group-aware split: identical grafted texts never span train/test.
idx_train, idx_test = grafted_train_test_indices(y, groups)

# Leakage-free structured features (order shared with app.py)
structured_cols = STRUCTURED_COLS

scaler = StandardScaler()
X_struct_train = scaler.fit_transform(df[structured_cols].values[idx_train])
X_struct_test = scaler.transform(df[structured_cols].values[idx_test])

# Shared TF-IDF config (see pipeline_config), fit on TRAIN rows only so
# test text never leaks into the vocabulary/idf. Do NOT hardcode a different
# config here: it overwrites the shared tfidf_vectorizer.pkl with an
# incompatible vocab and breaks inference with "X has N features, but
# DecisionTreeClassifier is expecting M".
tfidf = TfidfVectorizer(**TFIDF_PARAMS)
tfidf_train = tfidf.fit_transform(df["combined_text"].iloc[idx_train])
tfidf_test = tfidf.transform(df["combined_text"].iloc[idx_test])

X_train = hstack([csr_matrix(X_struct_train), tfidf_train])
X_test = hstack([csr_matrix(X_struct_test), tfidf_test])
y_train, y_test = y[idx_train], y[idx_test]
assert X_train.shape[1] == 9 + len(tfidf.vocabulary_), (
    f"Unexpected feature width {X_train.shape[1]}"
)

clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=12,
    min_samples_leaf=3,
    min_samples_split=6,
    class_weight="balanced"
)

print("Training decision tree...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Honest 5-fold CV: fresh vectorizer+scaler per fold, folds split by
# grafted-text group so duplicates never leak across folds.
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold_train, fold_test in cv.split(df["combined_text"], y, groups):
    fold_tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    fold_text_train = fold_tfidf.fit_transform(df["combined_text"].iloc[fold_train])
    fold_text_test = fold_tfidf.transform(df["combined_text"].iloc[fold_test])
    fold_scaler = StandardScaler()
    fold_struct_train = fold_scaler.fit_transform(df[structured_cols].values[fold_train])
    fold_struct_test = fold_scaler.transform(df[structured_cols].values[fold_test])
    fold_clf = DecisionTreeClassifier(
        random_state=42, max_depth=12, min_samples_leaf=3,
        min_samples_split=6, class_weight="balanced",
    )
    fold_clf.fit(
        hstack([csr_matrix(fold_struct_train), fold_text_train]), y[fold_train]
    )
    cv_scores.append(fold_clf.score(
        hstack([csr_matrix(fold_struct_test), fold_text_test]), y[fold_test]
    ))
cv_scores = np.array(cv_scores)

metrics = {
    "decisiontree": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_test": int(len(y_test)),
        "text_source": "grafted-real",
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
