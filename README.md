# GetAround - Delay Analysis & Pricing API  
A data science project for Jedha (Hugging Face deployment)

---

## 🎯 Objective

GetAround wants to reduce **late-return issues** and improve **rental price consistency**.

This project delivers:

- A **delay analysis** to define the right buffer between rentals  
- A **pricing model** predicting fair rental prices  
- A **FastAPI prediction API** deployed on Hugging Face  
- A **Streamlit dashboard** to explore delays & buffer policies  

---

## 🌐 Online Apps (public)

### **📊 Streamlit Dashboard**  
https://huggingface.co/spaces/smargot/getaround-dashboard  

Features:
- **Analyse descriptive** → global view of delays & impacted rentals  
- **Threshold picker** → interactive tool to test different buffer policies  

### **🧠 Prediction API (FastAPI)**  
Space page:  
https://huggingface.co/spaces/smargot/getaround-api  

Direct API (Swagger UI):  
https://smargot-getaround-api.hf.space/docs  

Endpoints:
- `GET /` → API status and documentation 
- `POST /predict` → Predicts rental price from car features  

Anyone can use these links — the Space is **public**.

---

## 🧩 What the project does

### **1. Delay Analysis**
- Measures delay frequency & duration  
- Quantifies impact on next rentals  
- Simulates buffer options (30, 60, 90 minutes…)  
- Helps balance customer satisfaction & owner revenue  

### **2. Pricing Model**
- Gradient Boosting trained on GetAround pricing dataset  
- Handles both numerical & categorical car attributes  
- Packaged as an MLflow `pyfunc` model  
- Served through FastAPI on Hugging Face  

---
## 🧪 Local Setup (Optional)

The easiest way to use this project is through the **online applications** on Hugging Face.  
Local setup is only needed if you want to run the Streamlit dashboard or the FastAPI API on your own machine.

### Requirements
- Python 3.10+ installed  
- `pip` available in your terminal (WSL recommended)

---

### 1. Clone the repository

```bash
git clone https://github.com/smargot/getaround_HF.git
cd getaround_HF
```

---

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv
```

Activate it with:

- **Windows PowerShell:**
  ```bash
  .venv\Scripts\activate
  ```
- **WSL / Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

You can skip this step if you prefer to use your global Python environment.

---

### 3. Install the Python dependencies

This project uses standard libraries such as FastAPI, Uvicorn, Streamlit, scikit-learn, MLflow, pandas, and numpy.

```bash
pip install fastapi uvicorn streamlit scikit-learn mlflow pandas numpy
```

If some packages are already installed, `pip` will simply reuse them.

---

### 4. Run the Streamlit dashboard

```bash
streamlit run apps/streamlit/app.py
```

Open in your browser:  
**http://localhost:8501**

---

### 5. Run the FastAPI pricing API

In another terminal:

```bash
uvicorn apps.api.main:app --reload
```

Open the interactive documentation:  
**http://localhost:8000/docs**



## 🏗️ Repository Structure (simplified)

```text
apps/
  api/         → FastAPI app (Hugging Face API Space)
  streamlit/   → Dashboard app

data/
  raw/         → Original datasets
  processed/   → Cleaned datasets

hf_api_package/
  MLflow_model/ → Packaged model used on Hugging Face

notebooks/
  analysis_delay-final.ipynb
  get_around_pricing_analysis.ipynb
  ML_pricing.ipynb

