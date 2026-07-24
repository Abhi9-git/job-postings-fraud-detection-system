import os
import pickle
import json
import uuid
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load models and pipeline components on startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading label encoders, TF-IDF vectorizers, and ML models...")
try:
    with open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_baseline = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_vectorizer_safe.pkl"), "rb") as f:
        tfidf_robust = pickle.load(f)
        
    models = {
        "baseline": {},
        "robust": {}
    }
    
    # Load Baseline Models
    for name in ["logisticregression", "randomforest", "decisiontree"]:
        model_path = os.path.join(BASE_DIR, f"{name}_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                models["baseline"][name] = pickle.load(f)
                print(f"Loaded baseline model: {name}")
                
    # Load Robust Models
    for name in ["logisticregression", "randomforest", "decisiontree"]:
        model_path = os.path.join(BASE_DIR, f"{name}_safe_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                models["robust"][name] = pickle.load(f)
                print(f"Loaded robust model: {name}")
except Exception as e:
    print(f"Error loading models or vectorizers: {e}")
    label_encoders = {}
    tfidf_baseline = None
    tfidf_robust = None
    models = {"baseline": {}, "robust": {}}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/options", methods=["GET"])
def get_options():
    """Returns unique categories for dropdown options."""
    options = {}
    try:
        for col, encoder in label_encoders.items():
            options[col] = list(encoder.classes_)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(options)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns the comparison metrics of all models."""
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics file not found"}), 404

@app.route("/api/predict", methods=["POST"])
def predict():
    """Performs real-time prediction using the selected model and mode."""
    data = request.json
    pipeline_type = data.get("pipeline_type", "robust").lower() # default to robust/safe mode
    selected_model_name = data.get("model", "logisticregression").lower()
    
    if pipeline_type not in models or selected_model_name not in models[pipeline_type]:
        return jsonify({"error": f"Model '{selected_model_name}' under '{pipeline_type}' not loaded"}), 400
    
    model = models[pipeline_type][selected_model_name]
    
    try:
        # Collect & combine texts
        job_desc = data.get("job_description", "").strip()
        requirements = data.get("requirements", "").strip()
        benefits = data.get("benefits", "").strip()
        company_prof = data.get("company_profile", "").strip()
        combined_text = f"{job_desc} {requirements} {benefits} {company_prof}"
        
        # Select vectorizer
        tfidf = tfidf_robust if pipeline_type == "robust" else tfidf_baseline
        tfidf_input = tfidf.transform([combined_text])
        
        # Get structured inputs
        experience = int(data.get("required_experience_years", 0))
        has_logo = int(data.get("has_logo", 0))
        open_pos = int(data.get("num_open_positions", 1))
        telecomm = int(data.get("telecommuting", 0))
        
        # Encoding categorical
        industry_val = data.get("industry")
        employment_val = data.get("employment_type")
        salary_val = data.get("salary_range")
        education_val = data.get("education_level")
        department_val = data.get("department")
        job_function_val = data.get("job_function")
        
        industry_enc = label_encoders["industry"].transform([industry_val])[0]
        employment_enc = label_encoders["employment_type"].transform([employment_val])[0]
        salary_enc = label_encoders["salary_range"].transform([salary_val])[0]
        education_enc = label_encoders["education_level"].transform([education_val])[0]
        department_enc = label_encoders["department"].transform([department_val])[0]
        job_function_enc = label_encoders["job_function"].transform([job_function_val])[0]
        
        if pipeline_type == "baseline":
            # Baseline includes text_length and is_gmail
            text_len = int(data.get("text_length", len(combined_text)))
            is_gmail = int(data.get("is_gmail", 0))
            structured_input = np.array([[
                experience, has_logo, open_pos, telecomm, text_len, is_gmail,
                industry_enc, employment_enc, salary_enc,
                education_enc, department_enc, job_function_enc
            ]])
        else:
            # Robust excludes text_length and is_gmail (leakages)
            structured_input = np.array([[
                experience, has_logo, open_pos, telecomm,
                industry_enc, employment_enc, salary_enc,
                education_enc, department_enc, job_function_enc
            ]])
        
        # Combine structured & sparse text TF-IDF
        input_combined = hstack([csr_matrix(structured_input), tfidf_input])
        
        # Predict
        prediction = int(model.predict(input_combined)[0])
        probability = model.predict_proba(input_combined)[0].tolist()
        
        return jsonify({
            "prediction": prediction,
            "probability": {
                "real": float(probability[0]),
                "fake": float(probability[1])
            },
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict-csv", methods=["POST"])
def predict_csv():
    """Predicts fake status for job postings uploaded in a CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    pipeline_type = request.form.get("pipeline_type", "robust").lower()
    selected_model_name = request.form.get("model", "logisticregression").lower()
    
    if pipeline_type not in models or selected_model_name not in models[pipeline_type]:
        return jsonify({"error": f"Model '{selected_model_name}' under '{pipeline_type}' not loaded"}), 400
        
    model = models[pipeline_type][selected_model_name]
    
    try:
        df = pd.read_csv(file)
        
        if len(df) == 0:
            return jsonify({"error": "Uploaded CSV is empty"}), 400
            
        # Combine text fields
        text_cols = ["job_description", "requirements", "benefits", "company_profile"]
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""
                
        combined_text = df[text_cols].fillna("").agg(" ".join, axis=1)
        
        # Select vectorizer
        tfidf = tfidf_robust if pipeline_type == "robust" else tfidf_baseline
        tfidf_input = tfidf.transform(combined_text)
        
        # Parse structured fields
        experience = df["required_experience_years"].fillna(0).astype(int) if "required_experience_years" in df.columns else pd.Series([0]*len(df))
        has_logo = df["has_logo"].fillna(0).astype(int) if "has_logo" in df.columns else pd.Series([0]*len(df))
        open_pos = df["num_open_positions"].fillna(1).astype(int) if "num_open_positions" in df.columns else pd.Series([1]*len(df))
        telecomm = df["telecommuting"].fillna(0).astype(int) if "telecommuting" in df.columns else pd.Series([0]*len(df))
        
        # Safely encode categorical columns
        def safe_encode(encoder, val):
            val_str = str(val).strip() if not pd.isna(val) else ""
            if val_str in encoder.classes_:
                return encoder.transform([val_str])[0]
            return 0
            
        industry_enc = df["industry"].apply(lambda x: safe_encode(label_encoders["industry"], x)) if "industry" in df.columns else pd.Series([0]*len(df))
        employment_enc = df["employment_type"].apply(lambda x: safe_encode(label_encoders["employment_type"], x)) if "employment_type" in df.columns else pd.Series([0]*len(df))
        salary_enc = df["salary_range"].apply(lambda x: safe_encode(label_encoders["salary_range"], x)) if "salary_range" in df.columns else pd.Series([0]*len(df))
        education_enc = df["education_level"].apply(lambda x: safe_encode(label_encoders["education_level"], x)) if "education_level" in df.columns else pd.Series([0]*len(df))
        department_enc = df["department"].apply(lambda x: safe_encode(label_encoders["department"], x)) if "department" in df.columns else pd.Series([0]*len(df))
        job_function_enc = df["job_function"].apply(lambda x: safe_encode(label_encoders["job_function"], x)) if "job_function" in df.columns else pd.Series([0]*len(df))
        
        if pipeline_type == "baseline":
            is_gmail = df["contact_email"].apply(lambda x: 1 if "gmail" in str(x).lower() else 0) if "contact_email" in df.columns else pd.Series([0]*len(df))
            text_len = combined_text.apply(len)
            structured_input = np.column_stack([
                experience.values, has_logo.values, open_pos.values, telecomm.values,
                text_len.values, is_gmail.values, industry_enc.values, employment_enc.values,
                salary_enc.values, education_enc.values, department_enc.values, job_function_enc.values
            ])
        else:
            structured_input = np.column_stack([
                experience.values, has_logo.values, open_pos.values, telecomm.values,
                industry_enc.values, employment_enc.values, salary_enc.values,
                education_enc.values, department_enc.values, job_function_enc.values
            ])
        
        # hstack
        input_combined = hstack([csr_matrix(structured_input), tfidf_input])
        
        # Predict
        predictions = model.predict(input_combined)
        probabilities = model.predict_proba(input_combined)
        
        df["predicted_label"] = predictions
        df["predicted_class"] = df["predicted_label"].map({0: "REAL", 1: "FAKE"})
        df["confidence_score"] = np.max(probabilities, axis=1)
        
        # Create temp folder if not exists
        temp_dir = os.path.join(BASE_DIR, "temp_predictions")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        out_path = os.path.join(temp_dir, f"{file_id}.csv")
        df.to_csv(out_path, index=False)
        
        # Build preview
        preview_df = df.head(50).copy().fillna("")
        
        return jsonify({
            "file_id": file_id,
            "summary": {
                "total": len(df),
                "real": int((predictions == 0).sum()),
                "fake": int((predictions == 1).sum())
            },
            "preview": preview_df.to_dict(orient="records"),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download-predictions/<file_id>", methods=["GET"])
def download_predictions(file_id):
    """Serves the generated prediction output file."""
    temp_dir = os.path.join(BASE_DIR, "temp_predictions")
    file_path = os.path.join(temp_dir, f"{file_id}.csv")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name="predicted_job_postings.csv", mimetype="text/csv")
    return jsonify({"error": "File not found or expired"}), 404

@app.route("/api/dataset-stats", methods=["GET"])
def dataset_stats():
    """Generates basic aggregate stats from CleanData.csv for the dashboard."""
    try:
        df_clean = pd.read_csv(os.path.join(BASE_DIR, "CleanData.csv"))
        
        total_rows = len(df_clean)
        is_fake_counts = df_clean["is_fake"].value_counts().to_dict()
        
        logo_stats = df_clean.groupby("is_fake")["has_logo"].mean().to_dict()
        df_clean["is_gmail"] = df_clean["contact_email"].apply(lambda x: 1 if "gmail" in str(x).lower() else 0)
        gmail_stats = df_clean.groupby("is_fake")["is_gmail"].mean().to_dict()
        top_industries = df_clean["industry"].value_counts().head(10).to_dict()
        avg_text_len = df_clean.groupby("is_fake")["text_length"].mean().to_dict()
        
        stats = {
            "total_records": total_rows,
            "classes": {
                "real": int(is_fake_counts.get(0, 0)),
                "fake": int(is_fake_counts.get(1, 0))
            },
            "has_logo_ratio": {
                "real": float(logo_stats.get(0, 0)),
                "fake": float(logo_stats.get(1, 0))
            },
            "is_gmail_ratio": {
                "real": float(gmail_stats.get(0, 0)),
                "fake": float(gmail_stats.get(1, 0))
            },
            "avg_text_length": {
                "real": float(avg_text_len.get(0, 0)),
                "fake": float(avg_text_len.get(1, 0))
            },
            "top_industries": top_industries
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns sample real and fake job postings from the dataset."""
    try:
        df_clean = pd.read_csv(os.path.join(BASE_DIR, "CleanData.csv"))
        
        real_sample = df_clean[df_clean["is_fake"] == 0].iloc[0].to_dict()
        fake_sample = df_clean[df_clean["is_fake"] == 1].iloc[0].to_dict()
        
        def clean_dict(d):
            new_d = {}
            for k, v in d.items():
                if isinstance(v, (np.integer, np.int64)):
                    new_d[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    new_d[k] = float(v)
                elif pd.isna(v):
                    new_d[k] = ""
                else:
                    new_d[k] = v
            return new_d
            
        return jsonify({
            "real": clean_dict(real_sample),
            "fake": clean_dict(fake_sample)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
