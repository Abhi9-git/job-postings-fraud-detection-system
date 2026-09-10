"""Shared pipeline constants for the fraud-detection project.

Single source of truth for the MAIN pipeline (logistic regression +
decision tree). ``app.py``, ``dtrees.py`` and ``LogisticRegression_03.py``
must all import from here instead of hardcoding their own copies — a past
drift between two hardcoded TF-IDF configs (336 vs 402 terms) caused
"X has 411 features, but DecisionTreeClassifier is expecting 345" errors.

The GitHub text-only models (``train_github_models.py``) intentionally use
a separate CountVectorizer pipeline; their constants live in that script.
"""

import re

# Text columns combined into one string, in this order, for both
# training and inference.
TEXT_COLS = ["job_description", "requirements", "benefits", "company_profile"]

# Categorical columns fitted with one LabelEncoder each.
CATEGORICAL_COLS = [
    "industry",
    "employment_type",
    "salary_range",
    "education_level",
    "department",
    "job_function",
]

# Structured feature order: 3 numeric + 6 encoded categoricals.
# Leakage-free by design: excludes text_length, is_gmail, has_logo.
STRUCTURED_COLS = [
    "required_experience_years",
    "num_open_positions",
    "telecommuting",
    "industry_enc",
    "employment_type_enc",
    "salary_range_enc",
    "education_level_enc",
    "department_enc",
    "job_function_enc",
]

# Must stay identical everywhere the main TF-IDF vectorizer is built.
# ~2000 terms + 9 structured = ~2009 total features. The vocab is wide on
# purpose: rare scam-domain words ("fee", "urgent", ...) must survive so the
# decision tree can split on them (min_df=2 mirrors the GitHub pipeline).
TFIDF_PARAMS = {
    "max_features": 2000,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
}

# Train/test split used by every training script and mirrored by
# the /api/evaluation endpoint.
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Total feature width the served main models expect.
EXPECTED_MAIN_FEATURES = 2009

# De-leakage: tokens that perfectly separate the classes in CleanData.csv
# ("global" opens every fake company_profile, "company" opens every real
# one) yet carry no real-world fraud signal. They are stripped from the
# combined text BEFORE TF-IDF in training (dtrees.py,
# LogisticRegression_03.py) and inference (app.py) so no model can key on
# them. Without this the decision tree collapses to a depth-2 stump that
# predicts REAL @ 1.0 for anything lacking the word "global".
LEAKAGE_TOKENS = frozenset({"global", "company"})

_LEAKAGE_RE = re.compile(
    r"\b(?:" + "|".join(sorted(LEAKAGE_TOKENS)) + r")\b",
    re.IGNORECASE,
)


def strip_leakage_tokens(text):
    """Remove leakage tokens (whole words, case-insensitive) from text."""
    return _LEAKAGE_RE.sub(" ", str(text))


# ---------------------------------------------------------------------------
# Real-text grafting.
#
# CleanData.csv's text columns are synthetic template word-salad: the classes
# differ only in which buzzword bigrams co-occur ("dynamic growth", ...), so
# any TF-IDF model trained on them is a constant function on real-world
# postings (the decision tree predicted REAL @ 1.0 for everything without the
# word "global"). To ground the MAIN pipeline in real fraud language, each
# CleanData row keeps its structured fields but its text is replaced with a
# label-matched posting from github_fakejobs.csv (truncated to wild-like
# length). Everything here is deterministic (fixed seeds), and every consumer
# — dtrees.py, LogisticRegression_03.py, app.py's /api/evaluation — MUST go
# through these helpers so training and evaluation see the identical frame.
# ---------------------------------------------------------------------------
CLEANDATA_FILE = "CleanData.csv"
GITHUB_TEXT_FILE = "github_fakejobs.csv"
GRAFT_SEED = 7
GRAFT_TEXT_WORDS = 100


def _truncate_words(text, n_words=GRAFT_TEXT_WORDS):
    return " ".join(str(text).split()[:n_words])


def load_main_frame(base_dir="."):
    """Build the deterministic grafted training frame.

    Returns a DataFrame with all CleanData columns plus:
      - ``combined_text``: label-matched real-world text (stripped + truncated)
      - ``text_group``: ``"<label>:<github-row>"`` id used to keep duplicate
        grafted texts on one side of the train/test split.
    """
    import os

    import numpy as np
    import pandas as pd

    clean = pd.read_csv(os.path.join(base_dir, CLEANDATA_FILE))
    gh = pd.read_csv(os.path.join(base_dir, GITHUB_TEXT_FILE))
    if "Unnamed: 0" in gh.columns:
        gh = gh.drop(columns=["Unnamed: 0"])
    if "fraudulent" in gh.columns and "label" not in gh.columns:
        gh = gh.rename(columns={"fraudulent": "label"})
    pools = {
        0: gh.loc[gh["label"].astype(int) == 0, "text"].fillna("").tolist(),
        1: gh.loc[gh["label"].astype(int) == 1, "text"].fillna("").tolist(),
    }
    if not pools[0] or not pools[1]:
        raise ValueError(f"No texts for both classes in {GITHUB_TEXT_FILE}")

    rng = np.random.RandomState(GRAFT_SEED)
    grafted, groups = [], []
    for cls in clean["is_fake"].astype(int).values:
        pool = pools[int(cls)]
        pick = int(rng.randint(0, len(pool)))
        grafted.append(_truncate_words(pool[pick]))
        groups.append(f"{int(cls)}:{pick}")

    frame = clean.copy()
    frame["combined_text"] = pd.Series(grafted).apply(strip_leakage_tokens)
    frame["text_group"] = groups
    return frame


def grafted_train_test_indices(y, groups, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """Deterministic per-class group shuffle of row indices.

    Rows sharing a grafted text (same ``text_group``) never land on both
    sides of the split, so test metrics are honest. Returns
    ``(train_idx, test_idx)`` as numpy arrays.
    """
    import numpy as np

    y = np.asarray(y)
    groups = np.asarray(groups)
    idx = np.arange(len(y))
    train_idx, test_idx = [], []
    for cls in (0, 1):
        cls_idx = idx[y == cls]
        cls_groups = groups[y == cls]
        _, first_seen = np.unique(cls_groups, return_index=True)
        unique_groups = cls_groups[np.sort(first_seen)]
        rng = np.random.RandomState(seed + int(cls))
        rng.shuffle(unique_groups)
        n_test = int(len(unique_groups) * test_size)
        test_groups = set(unique_groups[:n_test])
        is_test = np.array([g in test_groups for g in cls_groups])
        test_idx.extend(cls_idx[is_test].tolist())
        train_idx.extend(cls_idx[~is_test].tolist())
    return np.array(train_idx), np.array(test_idx)
