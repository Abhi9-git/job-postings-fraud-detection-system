"""Shared pipeline constants for the fraud-detection project.

Single source of truth for the MAIN pipeline (logistic regression +
decision tree). ``app.py``, ``dtrees.py`` and ``LogisticRegression_03.py``
must all import from here instead of hardcoding their own copies — a past
drift between two hardcoded TF-IDF configs (336 vs 402 terms) caused
"X has 411 features, but DecisionTreeClassifier is expecting 345" errors.

The GitHub text-only models (``train_github_models.py``) intentionally use
a separate CountVectorizer pipeline; their constants live in that script.
"""

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
# 402 terms + 9 structured = 411 total features.
TFIDF_PARAMS = {
    "max_features": 500,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
}

# Train/test split used by every training script and mirrored by
# the /api/evaluation endpoint.
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Total feature width the served main models expect.
EXPECTED_MAIN_FEATURES = 411
