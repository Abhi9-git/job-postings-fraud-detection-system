# 🕵️‍♂️ Fake Job Posting Detection & Comparison Dashboard

This project is a machine learning dashboard designed to detect fraudulent job postings. It combines text-based natural language processing (NLP) with structured metadata columns to train multiple classification models, evaluate their performance, and serve interactive predictions via a beautiful dark-themed web interface.

---

## 📁 Directory Structure
```text
Fk_Jb_Detection project/
├── CleanData.csv                    # Cleaned preprocessed dataset (3000 x 20)
├── fake_real_job_postings_3000x25.csv # Raw dataset (3000 x 25)
├── DataWrangling_01.ipynb           # Notebook: Data cleaning workflow
├── EDA_02.ipynb                     # Notebook: Exploratory Data Analysis
├── LogisticRegression_03.py          # Baseline CLI-based training & prediction script
├── train_comparisons.py             # Script to train LR, Random Forest, & Decision Tree
├── app.py                           # Flask backend web server
├── model_metrics.json               # Computed evaluation metrics for comparison
├── templates/
│   └── index.html                   # Glassmorphic frontend dashboard
├── venv/                            # Virtual environment directory
├── label_encoders.pkl               # Pickled categorical feature encoders
├── tfidf_vectorizer.pkl             # Pickled text TF-IDF vectorizer
├── logisticregression_model.pkl     # Pickled Logistic Regression model
├── randomforest_model.pkl           # Pickled Random Forest model
└── decisiontree_model.pkl           # Pickled Decision Tree model
```

---

## ⚡ Prerequisites & Installation

1. **Activate the Virtual Environment**:
   Ensure you are using the python environment in the workspace.
   ```bash
   source venv/bin/activate
   ```

2. **Required Dependencies**:
   The requirements are installed in `venv/`, including:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `scipy`
   - `matplotlib`
   - `seaborn`
   - `flask`

---

## 🚀 Running the Web Dashboard

1. Run the Flask server in development mode:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ⚙️ How the Detection Works
The classifier leverages a combined feature stack:
1. **NLP Text Vectorization**: Consolidates `job_description`, `requirements`, `benefits`, and `company_profile` into a single text block, transforming it using **TF-IDF Vectorization** (extracting top 500 unigrams & bigrams).
2. **Metadata Feature Encoding**: Checks whether the contact email is a Gmail address (`is_gmail`), company logo presence (`has_logo`), number of open positions, telecommuting, experience, and encodes categorical columns (e.g. `industry`, `employment_type`) using LabelEncoders.
3. **Stacked Sparse Matrix**: Combines both metadata and text sparse matrices using `scipy.sparse.hstack` before feeding the classifier.

---

## 📊 Models Compared
The system supports predictions and side-by-side metric comparisons between:
- **Logistic Regression**: Linear classifier optimized with L2 regularization.
- **Random Forest**: Multi-tree ensemble limiting max depth to avoid overfitting.
- **Decision Tree**: Fast rule-based tree model.
