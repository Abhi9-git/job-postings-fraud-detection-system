import os
import pickle
import json
import uuid
import re
import numpy as np
import pandas as pd
import requests as http_requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template, send_file
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load models and pipeline components on startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading label encoders, TF-IDF vectorizer, scaler, and ML models...")
try:
    with open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None
        
    models = {}
    model_files = {
        "logisticregression": "lr_model.pkl",
        "decisiontree": "dtres_model.pkl",
        "github_rf": "github_rf_model.pkl",
        "github_dt": "github_dt_model.pkl"
    }

    for model_name, file_name in model_files.items():
        model_path = os.path.join(BASE_DIR, file_name)
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                models[model_name] = pickle.load(f)
                print(f"Loaded model: {model_name}")

    github_vectorizer_path = os.path.join(BASE_DIR, "github_vectorizer.pkl")
    if os.path.exists(github_vectorizer_path):
        with open(github_vectorizer_path, "rb") as f:
            github_vectorizer = pickle.load(f)
    else:
        github_vectorizer = None

except Exception as e:
    print(f"Error loading models or vectorizers: {e}")
    label_encoders = {}
    tfidf_vectorizer = None
    scaler = None
    models = {}

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
    """Returns the performance metrics of trained models."""
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics file not found"}), 404

def _build_feature_vector(data, selected_model_name=None):
    """Build the combined feature vector for prediction.
    
    Centralizes feature engineering so both manual and scraped
    predictions use the exact same pipeline — no hardcoded values or target leakages.
    """

    selected_model_name = selected_model_name or data.get("model", "logisticregression").lower()
    # Collect & combine texts
    job_desc = str(data.get("job_description", "")).strip()
    requirements = str(data.get("requirements", "")).strip()
    benefits = str(data.get("benefits", "")).strip()
    company_prof = str(data.get("company_profile", "")).strip()
    combined_text = f"{job_desc} {requirements} {benefits} {company_prof}"
    
    # NLP TF-IDF Vectorization
    if selected_model_name in ["github_rf", "github_dt"]:
        if github_vectorizer is None:
            raise ValueError("GitHub external vectorizer is not available")
        text_input = github_vectorizer.transform([combined_text])
        return text_input

    tfidf_input = tfidf_vectorizer.transform([combined_text])
    
    # Structured inputs with safe defaults
    experience = int(data.get("required_experience_years", 0))
    open_pos = int(data.get("num_open_positions", 1))
    telecomm = int(data.get("telecommuting", 0))
    
    # Encoding categorical fields with safe fallback
    def safe_encode(encoder_key, value):
        if encoder_key not in label_encoders:
            return 0
        encoder = label_encoders[encoder_key]
        val_str = str(value).strip() if value else ""
        if val_str in encoder.classes_:
            return encoder.transform([val_str])[0]
        return 0
    
    industry_enc = safe_encode("industry", data.get("industry"))
    employment_enc = safe_encode("employment_type", data.get("employment_type"))
    salary_enc = safe_encode("salary_range", data.get("salary_range"))
    education_enc = safe_encode("education_level", data.get("education_level"))
    department_enc = safe_encode("department", data.get("department"))
    job_function_enc = safe_encode("job_function", data.get("job_function"))
    
    # Leakage-free structured input array (excludes text_length, is_gmail, and has_logo)
    structured_input = np.array([[
        experience, open_pos, telecomm,
        industry_enc, employment_enc, salary_enc,
        education_enc, department_enc, job_function_enc
    ]], dtype=float)

    if scaler is not None:
        structured_input = scaler.transform(structured_input)
    
    # Combine structured & sparse text TF-IDF
    return hstack([csr_matrix(structured_input), tfidf_input])


@app.route("/api/predict", methods=["POST"])
def predict():
    """Performs real-time prediction using the selected model."""
    data = request.json
    selected_model_name = data.get("model", "logisticregression").lower()
    
    if selected_model_name not in models:
        return jsonify({"error": f"Model '{selected_model_name}' not loaded"}), 400
    
    model = models[selected_model_name]
    
    try:
        input_combined = _build_feature_vector(data, selected_model_name)
        
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


# ==========================================
# Internshala Web Scraping
# ==========================================

SCRAPER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def _map_stipend_to_salary_range(stipend_text):
    """Map Internshala stipend text to dataset salary_range categories."""
    if not stipend_text:
        return "Not Disclosed"
    
    numbers = re.findall(r'[\d,]+', stipend_text.replace(',', ''))
    if not numbers:
        return "Not Disclosed"
    
    amounts = [int(n) for n in numbers]
    max_monthly = max(amounts) if amounts else 0
    annual_inr = max_monthly * 12
    annual_usd_approx = annual_inr / 83
    
    if annual_usd_approx < 40000:
        return "$40k-$60k"
    elif annual_usd_approx < 60000:
        return "$40k-$60k"
    elif annual_usd_approx < 80000:
        return "$60k-$80k"
    elif annual_usd_approx < 100000:
        return "$80k-$100k"
    else:
        return "$80k-$100k"


def _map_employment_type(emp_type_text):
    """Map Internshala employment type to dataset categories."""
    if not emp_type_text:
        return "Internship"
    
    text_lower = emp_type_text.lower()
    
    if "full" in text_lower and "time" in text_lower:
        return "Full-time"
    elif "part" in text_lower and "time" in text_lower:
        return "Part-time"
    elif "contract" in text_lower:
        return "Contract"
    elif "intern" in text_lower:
        return "Internship"
    
    if "employment_type" in label_encoders:
        for known in label_encoders["employment_type"].classes_:
            if known.lower() in text_lower or text_lower in known.lower():
                return known
    
    return "Internship"


def _map_industry(industry_text):
    """Map scraped industry to the closest known dataset category."""
    if not industry_text or "industry" not in label_encoders:
        return "IT"
    
    known = label_encoders["industry"].classes_
    text_lower = industry_text.lower()
    
    for k in known:
        if k.lower() == text_lower or k.lower() in text_lower or text_lower in k.lower():
            return k
    
    keyword_map = {
        "IT": ["tech", "software", "computer", "internet", "digital", "saas", "it"],
        "Finance": ["finance", "banking", "insurance", "fintech", "accounting"],
        "Healthcare": ["health", "medical", "pharma", "hospital", "biotech"],
        "Education": ["education", "training", "learning", "academic", "university"],
        "Marketing": ["marketing", "advertising", "media", "entertainment", "creative"],
        "Retail": ["retail", "ecommerce", "consumer", "fashion", "food"]
    }
    
    for category, keywords in keyword_map.items():
        if category in known:
            for kw in keywords:
                if kw in text_lower:
                    return category
    
    return known[0]


def _infer_required_experience_years(scraped):
    """Infer experience from Internshala text so scraped data resembles training examples."""
    text_blob = " ".join([
        scraped.get("job_title", ""),
        scraped.get("job_description", ""),
        scraped.get("requirements", "")
    ]).lower()

    if any(term in text_blob for term in ["fresher", "freshers", "no experience", "entry level"]):
        return 0

    patterns = [
        r"(\d+)\s*(?:-|to)\s*(\d+)\s*years?\s*(?:of\s+)?experience",
        r"(\d+)\s*\+\s*years?\s*(?:of\s+)?experience",
        r"experience\s*(?:of\s+)?(\d+)\s*years?",
        r"(\d+)\s*years?\s*(?:of\s+)?experience"
    ]

    for pattern in patterns:
        match = re.search(pattern, text_blob)
        if match:
            numbers = [int(group) for group in match.groups() if group and group.isdigit()]
            if numbers:
                return min(max(numbers), 10)

    # fallback for common Internshala phrasing such as "0-1 years"
    year_matches = re.findall(r"(\d+)\s*years?", text_blob)
    if year_matches:
        return min(int(year_matches[0]), 10)

    return 0


def _infer_education_level(scraped):
    """Infer education level from scraped text to match encoder categories."""
    if "education_level" not in label_encoders:
        return "Bachelor"

    known = list(label_encoders["education_level"].classes_)
    text_blob = " ".join([
        scraped.get("job_title", ""),
        scraped.get("job_description", ""),
        scraped.get("requirements", "")
    ]).lower()

    if any(term in text_blob for term in ["phd", "doctorate"]):
        return "PhD"
    if any(term in text_blob for term in ["master", "masters", "graduate degree"]):
        return "Master"
    if any(term in text_blob for term in ["bachelor", "degree", "undergraduate", "graduation"]):
        return "Bachelor"
    if any(term in text_blob for term in ["12th", "class 12", "school", "high school"]):
        return "High School"

    return known[0] if known else "Bachelor"


def _infer_department(scraped):
    """Infer department from scraped job text to match training categories."""
    if "department" not in label_encoders:
        return "Engineering"

    known = list(label_encoders["department"].classes_)
    text_blob = " ".join([
        scraped.get("job_title", ""),
        scraped.get("job_description", ""),
        scraped.get("requirements", ""),
        scraped.get("benefits", "")
    ]).lower()

    if any(term in text_blob for term in ["design", "ui", "ux", "graphic", "creative"]):
        return "Design"
    if any(term in text_blob for term in ["hr", "human resource", "recruitment", "talent"]):
        return "HR"
    if any(term in text_blob for term in ["marketing", "content", "social media", "seo", "brand"]):
        return "Marketing"
    if any(term in text_blob for term in ["sales", "business development", "client", "account"]):
        return "Sales"
    if any(term in text_blob for term in ["software", "developer", "engineer", "python", "java", "data", "product"]):
        return "Engineering"

    # fallback toward a reasonable department based on industry if nothing matches
    industry = _map_industry(scraped.get("industry_raw", ""))
    if industry == "Marketing":
        return "Marketing"
    if industry == "Education":
        return "Engineering"
    if industry == "Finance":
        return "Sales"

    return known[0] if known else "Engineering"


def _infer_job_function(scraped):
    """Infer job function from scraped job text to match training categories."""
    if "job_function" not in label_encoders:
        return "Development"

    known = list(label_encoders["job_function"].classes_)
    text_blob = " ".join([
        scraped.get("job_title", ""),
        scraped.get("job_description", ""),
        scraped.get("requirements", "")
    ]).lower()

    if any(term in text_blob for term in ["analysis", "analyst", "data analyst", "research"]):
        return "Analysis"
    if any(term in text_blob for term in ["support", "customer support", "operations", "service"]):
        return "Support"
    if any(term in text_blob for term in ["manager", "lead", "head", "team lead"]):
        return "Management"

    # Most Internshala roles are technical/development-focused unless evidence says otherwise
    return "Development" if "Development" in known else known[0]


def _scrape_internshala(url):
    """Scrape job/internship data from an Internshala URL.
    
    Extracts JSON-LD structured data or parses HTML directly.
    """
    try:
        resp = http_requests.get(url, headers=SCRAPER_HEADERS, timeout=15)
        resp.raise_for_status()
    except http_requests.exceptions.RequestException as e:
        return None, f"Failed to fetch URL: {str(e)}"
    
    soup = BeautifulSoup(resp.text, "html.parser")
    scraped = {}
    
    # JSON-LD structured data
    json_ld_data = None
    for script_tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script_tag.string)
            if isinstance(data, dict) and data.get("@type") == "JobPosting":
                json_ld_data = data
                break
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("@type") == "JobPosting":
                        json_ld_data = item
                        break
        except (json.JSONDecodeError, TypeError):
            continue
    
    if json_ld_data:
        scraped["job_title"] = json_ld_data.get("title", "")
        raw_desc = json_ld_data.get("description", "")
        desc_soup = BeautifulSoup(raw_desc, "html.parser")
        scraped["job_description"] = desc_soup.get_text(separator=" ", strip=True)
        
        raw_resp = json_ld_data.get("responsibilities", "")
        resp_soup = BeautifulSoup(raw_resp, "html.parser")
        scraped["requirements"] = resp_soup.get_text(separator=" ", strip=True)
        
        scraped["skills"] = json_ld_data.get("skills", "")
        scraped["industry_raw"] = json_ld_data.get("industry", "")
        scraped["employment_type_raw"] = json_ld_data.get("employmentType", "")
        scraped["num_open_positions"] = json_ld_data.get("totalJobOpenings", 1)
        
        org = json_ld_data.get("hiringOrganization", {})
        scraped["company_name"] = org.get("name", "")
        scraped["has_logo"] = 1 if org.get("logo") else 0
        scraped["company_website"] = org.get("sameAs", "")
        
        salary_info = json_ld_data.get("baseSalary", {})
        if salary_info:
            value = salary_info.get("value", {})
            min_val = value.get("minValue", "")
            max_val = value.get("maxValue", "")
            currency = salary_info.get("currency", "INR")
            unit = value.get("unitText", "MONTH")
            scraped["stipend_raw"] = f"{currency} {min_val}-{max_val}/{unit}"
        
        about_section = soup.find("div", class_="text-container about_company_text_container")
        if about_section:
            scraped["company_profile"] = about_section.get_text(separator=" ", strip=True)
        else:
            if "About Company:" in raw_desc:
                about_part = raw_desc.split("About Company:")[-1]
                about_soup = BeautifulSoup(about_part, "html.parser")
                scraped["company_profile"] = about_soup.get_text(separator=" ", strip=True)
            else:
                scraped["company_profile"] = f"{scraped['company_name']} is a company."
    
    else:
        title_el = soup.find("h1", class_="heading_title")
        scraped["job_title"] = title_el.get_text(strip=True) if title_el else ""
        
        company_el = soup.find("div", class_="company_name")
        scraped["company_name"] = company_el.get_text(strip=True) if company_el else ""
        
        logo_el = soup.find("div", class_="internship_logo")
        scraped["has_logo"] = 1 if logo_el and logo_el.find("img") else 0
        
        about_section = soup.find("div", class_="internship_details")
        scraped["job_description"] = about_section.get_text(separator=" ", strip=True) if about_section else ""
        
        company_section = soup.find("div", class_="text-container about_company_text_container")
        scraped["company_profile"] = company_section.get_text(separator=" ", strip=True) if company_section else ""
        
        stipend_el = soup.find("span", class_="stipend")
        scraped["stipend_raw"] = stipend_el.get_text(strip=True) if stipend_el else ""
        
        scraped["num_open_positions"] = 1
        scraped["requirements"] = ""
        scraped["industry_raw"] = ""
        scraped["employment_type_raw"] = "INTERN"
    
    perks_section = soup.find("div", class_="round_tabs_container")
    if perks_section:
        perks = [tag.get_text(strip=True) for tag in perks_section.find_all("div", class_="round_tabs")]
        scraped["benefits"] = ", ".join(perks) if perks else ""
    else:
        scraped["benefits"] = ""
    
    return scraped, None


@app.route("/api/scrape-internshala", methods=["POST"])
def scrape_internshala():
    """Scrape an Internshala URL and predict if the posting is fake."""
    data = request.json
    url = data.get("url", "").strip()
    selected_model_name = data.get("model", "logisticregression").lower()
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    if "internshala.com" not in url:
        return jsonify({"error": "Please provide a valid Internshala URL"}), 400
    
    if selected_model_name not in models:
        return jsonify({"error": f"Model '{selected_model_name}' not loaded"}), 400
    
    model = models[selected_model_name]
    
    # Scrape the page
    scraped, error = _scrape_internshala(url)
    if error:
        return jsonify({"error": error}), 400
    
    # Map scraped fields to the exact training schema used by the saved models.
    prediction_input = {
        "job_description": scraped.get("job_description", ""),
        "requirements": scraped.get("requirements", ""),
        "benefits": scraped.get("benefits", ""),
        "company_profile": scraped.get("company_profile", ""),
        "required_experience_years": _infer_required_experience_years(scraped),
        "num_open_positions": int(scraped.get("num_open_positions", 1)),
        "telecommuting": 0,
        "industry": _map_industry(scraped.get("industry_raw", "")),
        "employment_type": _map_employment_type(scraped.get("employment_type_raw", "")),
        "salary_range": _map_stipend_to_salary_range(scraped.get("stipend_raw", "")),
        "education_level": _infer_education_level(scraped),
        "department": _infer_department(scraped),
        "job_function": _infer_job_function(scraped),
    }
    
    full_text = f"{scraped.get('job_title', '')} {scraped.get('job_description', '')}".lower()
    if "work from home" in full_text or "wfh" in full_text or "remote" in full_text:
        prediction_input["telecommuting"] = 1
    
    try:
        input_combined = _build_feature_vector(prediction_input, selected_model_name)
        
        prediction = int(model.predict(input_combined)[0])
        probability = model.predict_proba(input_combined)[0].tolist()
        
        return jsonify({
            "prediction": prediction,
            "probability": {
                "real": float(probability[0]),
                "fake": float(probability[1])
            },
            "scraped_data": {
                "job_title": scraped.get("job_title", ""),
                "company_name": scraped.get("company_name", ""),
                "job_description": scraped.get("job_description", "")[:500],
                "company_profile": scraped.get("company_profile", "")[:500],
                "requirements": scraped.get("requirements", "")[:300],
                "benefits": scraped.get("benefits", "")[:300],
                "skills": scraped.get("skills", ""),
                "stipend": scraped.get("stipend_raw", "Not Disclosed"),
                "has_logo": scraped.get("has_logo", 0),
                "num_open_positions": scraped.get("num_open_positions", 1),
                "industry": prediction_input["industry"],
                "employment_type": prediction_input["employment_type"],
            },
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@app.route("/api/predict-csv", methods=["POST"])
def predict_csv():
    """Predicts fake status for job postings uploaded in a CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    selected_model_name = request.form.get("model", "logisticregression").lower()
    
    if selected_model_name not in models:
        return jsonify({"error": f"Model '{selected_model_name}' not loaded"}), 400
        
    model = models[selected_model_name]
    
    try:
        df = pd.read_csv(file)
        
        if len(df) == 0:
            return jsonify({"error": "Uploaded CSV is empty"}), 400
            
        text_cols = ["job_description", "requirements", "benefits", "company_profile"]
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""

        combined_text = df[text_cols].fillna("").agg(" ".join, axis=1)

        if selected_model_name in ["github_rf", "github_dt"]:
            if github_vectorizer is None:
                return jsonify({"error": "GitHub external vectorizer is not available"}), 400
            input_combined = github_vectorizer.transform(combined_text)
        else:
            tfidf_input = tfidf_vectorizer.transform(combined_text)

            experience = df["required_experience_years"].fillna(0).astype(int) if "required_experience_years" in df.columns else pd.Series([0]*len(df))
            open_pos = df["num_open_positions"].fillna(1).astype(int) if "num_open_positions" in df.columns else pd.Series([1]*len(df))
            telecomm = df["telecommuting"].fillna(0).astype(int) if "telecommuting" in df.columns else pd.Series([0]*len(df))

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

            structured_input = np.column_stack([
                experience.values, open_pos.values, telecomm.values,
                industry_enc.values, employment_enc.values, salary_enc.values,
                education_enc.values, department_enc.values, job_function_enc.values
            ]).astype(float)

            if scaler is not None:
                structured_input = scaler.transform(structured_input)

            input_combined = hstack([csr_matrix(structured_input), tfidf_input])
        
        predictions = model.predict(input_combined)
        probabilities = model.predict_proba(input_combined)
        
        df["predicted_label"] = predictions
        df["predicted_class"] = df["predicted_label"].map({0: "REAL", 1: "FAKE"})
        df["confidence_score"] = np.max(probabilities, axis=1)
        
        temp_dir = os.path.join(BASE_DIR, "temp_predictions")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        out_path = os.path.join(temp_dir, f"{file_id}.csv")
        df.to_csv(out_path, index=False)
        
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

@app.route("/api/evaluation", methods=["GET"])
def model_evaluation():
    """Returns confusion matrices + ROC curve points per model for the dashboard.

    Rebuilds the exact 80/20 stratified split used at training time
    (test_size=0.20, random_state=42) with the *loaded* vectorizers/scaler,
    so the curves always match the models actually serving predictions.
    Result is cached in-memory after the first call.
    """
    global _eval_cache
    if "_eval_cache" not in globals():
        _eval_cache = None
    if _eval_cache is not None:
        return jsonify(_eval_cache)

    def _downsample(xs, ys, max_points=120):
        n = len(xs)
        if n <= max_points:
            return [float(v) for v in xs], [float(v) for v in ys]
        idx = np.linspace(0, n - 1, max_points).astype(int)
        # always keep first/last points for a proper ROC shape
        idx[0], idx[-1] = 0, n - 1
        return [float(xs[i]) for i in idx], [float(ys[i]) for i in idx]

    def _score_block(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fpr_d, tpr_d = _downsample(fpr, tpr)
        return {
            "confusion_matrix": {"labels": ["REAL", "FAKE"], "matrix": cm},
            "roc": {
                "fpr": fpr_d,
                "tpr": tpr_d,
                "auc": float(roc_auc_score(y_true, y_prob)),
            },
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "n_test": int(len(y_true)),
        }

    try:
        out = {}

        # ---- Main pipeline models (CleanData.csv, shared TF-IDF + scaler) ----
        main_needed = [m for m in ("logisticregression", "decisiontree") if m in models]
        if main_needed and tfidf_vectorizer is not None:
            df = pd.read_csv(os.path.join(BASE_DIR, "CleanData.csv"))
            text_cols = ["job_description", "requirements", "benefits", "company_profile"]
            for col in text_cols:
                if col not in df.columns:
                    df[col] = ""
            combined = df[text_cols].fillna("").agg(" ".join, axis=1)
            tfidf_mat = tfidf_vectorizer.transform(combined)

            def _enc(col, val):
                if col not in label_encoders:
                    return 0
                enc = label_encoders[col]
                s = str(val).strip() if not pd.isna(val) else ""
                return int(enc.transform([s])[0]) if s in enc.classes_ else 0

            struct = np.column_stack([
                df["required_experience_years"].fillna(0).astype(int).values
                if "required_experience_years" in df.columns else np.zeros(len(df)),
                df["num_open_positions"].fillna(1).astype(int).values
                if "num_open_positions" in df.columns else np.ones(len(df)),
                df["telecommuting"].fillna(0).astype(int).values
                if "telecommuting" in df.columns else np.zeros(len(df)),
                df["industry"].apply(lambda x: _enc("industry", x)).values
                if "industry" in df.columns else np.zeros(len(df)),
                df["employment_type"].apply(lambda x: _enc("employment_type", x)).values
                if "employment_type" in df.columns else np.zeros(len(df)),
                df["salary_range"].apply(lambda x: _enc("salary_range", x)).values
                if "salary_range" in df.columns else np.zeros(len(df)),
                df["education_level"].apply(lambda x: _enc("education_level", x)).values
                if "education_level" in df.columns else np.zeros(len(df)),
                df["department"].apply(lambda x: _enc("department", x)).values
                if "department" in df.columns else np.zeros(len(df)),
                df["job_function"].apply(lambda x: _enc("job_function", x)).values
                if "job_function" in df.columns else np.zeros(len(df)),
            ]).astype(float)
            if scaler is not None:
                struct = scaler.transform(struct)
            X_all = hstack([csr_matrix(struct), tfidf_mat])
            y_all = df["is_fake"].astype(int).values
            # Same split as training scripts
            idx = np.arange(len(y_all))
            idx_train, idx_test = train_test_split(
                idx, test_size=0.20, random_state=42, stratify=y_all
            )
            X_test, y_test = X_all[idx_test], y_all[idx_test]
            for name in main_needed:
                clf = models[name]
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1]
                out[name] = _score_block(y_test, y_pred, y_prob)

        # ---- GitHub text-only models (github_fakejobs.csv, CountVectorizer) ----
        gh_needed = [m for m in ("github_rf", "github_dt") if m in models]
        if gh_needed and github_vectorizer is not None:
            gh_path = os.path.join(BASE_DIR, "github_fakejobs.csv")
            if os.path.exists(gh_path):
                gdf = pd.read_csv(gh_path)
                if "Unnamed: 0" in gdf.columns:
                    gdf = gdf.drop(columns=["Unnamed: 0"])
                if "fraudulent" in gdf.columns and "label" not in gdf.columns:
                    gdf = gdf.rename(columns={"fraudulent": "label"})
                texts = gdf["text"].fillna("").astype(str)
                labels = gdf["label"].astype(int).values
                X_all_gh = github_vectorizer.transform(texts)
                idx = np.arange(len(labels))
                _, idx_test = train_test_split(
                    idx, test_size=0.20, random_state=42, stratify=labels
                )
                X_test_gh, y_test_gh = X_all_gh[idx_test], labels[idx_test]
                for name in gh_needed:
                    clf = models[name]
                    y_pred = clf.predict(X_test_gh)
                    y_prob = clf.predict_proba(X_test_gh)[:, 1]
                    out[name] = _score_block(y_test_gh, y_pred, y_prob)

        _eval_cache = out
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
