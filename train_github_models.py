import json
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = "github_fakejobs.csv"
MODEL_METRICS_FILE = "model_metrics.json"


def save_metrics(metrics_dict):
    try:
        with open(MODEL_METRICS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}

    existing.update(metrics_dict)

    with open(MODEL_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)


print("Loading GitHub fake-job dataset...")
df = pd.read_csv(DATA_FILE)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "fraudulent" in df.columns and "label" not in df.columns:
    df = df.rename(columns={"fraudulent": "label"})

if "text" not in df.columns:
    raise ValueError("Expected a 'text' column in github_fakejobs.csv")

# Keep only the columns needed for text-only classification
text_df = df[["text", "label"]].copy()
text_df["text"] = text_df["text"].fillna("")
text_df["label"] = text_df["label"].astype(int)

X = text_df["text"]
y = text_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training CountVectorizer and models...")
vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "github_rf": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
    ),
    "github_dt": DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced",
    ),
}

metrics = {}

for name, clf in models.items():
    print(f"  Training {name}...")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1]

    cv_scores = cross_val_score(
        clf,
        vectorizer.transform(X),
        y,
        cv=cv,
        scoring="accuracy",
    )

    feature_names = vectorizer.get_feature_names_out()
    importances = getattr(clf, "feature_importances_", None)
    top_features = []
    if importances is not None:
        top_indices = importances.argsort()[-10:][::-1]
        top_features = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top_indices
        ]

    metrics[name] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "top_features": top_features,
    }

    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(clf, f)

with open("github_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

save_metrics(metrics)

print("\nSaved external GitHub-based models:")
for name in models:
    print(f"  - {name}_model.pkl")
print("  - github_vectorizer.pkl")
print("\nMetrics saved to model_metrics.json")
