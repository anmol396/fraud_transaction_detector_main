<div align="center">

# 🚀 FraudB | AI-Driven Fraud Analytics & Assistant

[![Django](https://img.shields.io/badge/Django-5.x-0C4B33?logo=django&logoColor=white)](https://www.djangoproject.com/) 
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-13aa52?logo=mongodb&logoColor=white)](https://www.mongodb.com/) 
[![n8n](https://img.shields.io/badge/Automation-n8n-f05a24?logo=n8n&logoColor=white)](https://n8n.io) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 🌟 Overview

FraudB is a **full-featured fraud analytics platform** with an AI assistant designed to help businesses monitor, analyze, and act on potentially fraudulent transactions. It combines **transaction-level risk scoring, predictive insights, and analytics dashboards** to empower informed decision-making.

Key highlights:  
- **AI Assistant:** Provides actionable guidance based on live analytics and historical trends.  
- **Fraud Analytics:** Metrics across customizable time windows, risk levels, and channel insights.  
- **Predictions:** Supports single and batch transaction predictions with contextual summaries.  
- **Authentication & Alerts:** OTP-based email verification, profile management, and alert notifications.  
- **Production-Ready:** Secure setup with CSRF protection, environment-driven configuration, and HTTPS-ready deployment.

---

## 🛠️ Tech Stack

- **Backend:** Django 5, Gunicorn, WhiteNoise  
- **Database:** MongoDB (Atlas or self-hosted)  
- **AI Orchestration:** n8n Webhook Integration  
- **Frontend:** Bootstrap 5 templates  
- **Security & Infra:** Environment variables, CSRF hardening, HTTPS-ready
 ```
| Layer                          | Technologies Used                                  | Description                                                                                                     |
| ------------------------------ | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 🎨 **Frontend**                | React 18 · Vite · Tailwind CSS · Axios             | Collects transaction inputs, displays fraud predictions, and visualizes insights through interactive dashboards |
| ⚙️ **Backend**                 | FastAPI · Uvicorn · Pandas · Scikit-Learn · Joblib | Handles RESTful API requests, data preprocessing, and communication between frontend and ML model               |
| 🧠 **ML Engine / Model Layer** | XGBoost · Random Forest · SHAP                     | Performs fraud classification, calculates risk probabilities, and generates explainability insights             |
| 📂 **Data Layer / Database**   | CSV · PostgreSQL (optional)                        | Stores preprocessed transactions, predictions, and user data for audit and retraining                           |
| 🔍 **Explainability Layer**    | SHAP · LIME · PDP                                  | Provides interpretable AI explanations for model decisions and feature importance                               |
| 📊 **Monitoring Layer**        | Grafana · InfluxDB                                 | Tracks real-time system health, API latency, and model performance metrics                                      |
| 🔗 **Integration Layer**       | n8n Workflow Automation                            | Automates backend operations such as alerting, email notifications, and retraining triggers                     | 

 ```

---

## ✨ Features

- **Assistant Chat:** Business-facing AI guidance for fraud monitoring.  
- **Analytics Dashboard:** Overview of fraud rate, top risky channels, and time-series trends.  
- **Transaction Predictions:** Single or batch predictions with risk scoring and descriptive insights.  
- **Authentication:** Secure OTP/email-based login, password reset, and user preferences.  
- **Security:** CSRF protection, HTTPS-ready configuration, secure secret management.

---

## 📁 Project Layout

- `project_settings/settings.py` — Environment-driven configuration (email, hosts, security).  
- `core/`, `api/`, `accounts/`, `ml/` — Main Django apps.  
- `static/` → Collected to `staticfiles/` for production use.  
- `.env.example` — Template to create your `.env` file safely.  
- `.gitignore` — Excludes sensitive data, caches, sessions, media, and logs.
---
## 📁 Project Structure
```
fraud_transaction_detector/
│
├── accounts/                         # Handles user accounts, auth, etc.
│   ├── migrations/                  # Django migration files
│   ├── templates/
│   │   └── accounts/
│   │       └── emails/              # Email templates for user interactions
│   └── __init__.py
│
├── api/                              # API-related code (views, serializers, urls)
│   └── __init__.py
│
├── core/                             # Core app: dashboard, home, etc.
│   ├── templates/
│   │   └── core/                    # Templates for core app
│   ├── templatetags/               # Custom Django template tags
│   └── __init__.py
│
├── ml/                               # Machine learning-related code
│   ├── data/                        # Raw/processed datasets
│   ├── models/
│   │   └── fraud_models/            # Trained fraud detection models
│   ├── notebook/                   # Jupyter notebooks for EDA/training
│   └── __init__.py
│
├── project_settings/                # Django settings module
│   └── __init__.py
│
├── static/                           # Static files (CSS, JS, images)
│   └── assets/
│       ├── css/
│       ├── imgs/
│       └── js/
│
├── manage.py                         # Django entry point
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files/folders to ignore in git
├── README.md                         # Project overview and setup instructions
└── LICENSE                           # (Optional) License for your project


```
---
## ⚡ Installation Guide

### 🧩 Backend Setup
```

cd fraud_transaction_detector-main/ml
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux


pip install -r requirements.txt
uvicorn app:app --reload
Visit http://127.0.0.1:8000/docs
```

```
🖥️ Frontend Setup
cd ../frontend
npm install
npm run dev

Frontend runs on http://localhost:5173
```
## ⚡ Quick Start (Local)

1. **Create a virtual environment and install dependencies**:  
  
    pip install -r requirements.txt
   
2. **Copy environment template and configure**:  
   
    cp .env.example .env
   
3. **Collect static files and run the server**:  
    
    python manage.py collectstatic --noinput
    python manage.py runserver
    
4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🔑 Environment Configuration

Key variables (all documented in `.env.example`):  
- `SECRET_KEY`, `DJANGO_DEBUG`  
- `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`  
- `MONGO_URI`, `MONGO_DB`  
- Email configuration: `EMAIL_*` (Gmail App Password recommended)  
- n8n integration: `N8N_WEBHOOK_URL`, `N8N_WEBHOOK_TOKEN`  
- Policy and rule settings: thresholds, rule weights

---

## 🔗 n8n Integration

- App posts transaction payloads to `N8N_WEBHOOK_URL`.  
- Respond with JSON: `{ "reply": "..." }`.  
- Optional token-based security using `N8N_WEBHOOK_TOKEN`.

---

## 🛡️ Security Checklist

- Rotate `SECRET_KEY` and keep `DJANGO_DEBUG=0` in production.  
- Set `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS` correctly.  
- Serve over HTTPS with secure cookies and SSL redirect.  
- Never commit `.env` or sensitive databases; `.gitignore` already excludes them.

---

## 🐛 Troubleshooting

- **403 CSRF:** Ensure `CSRF_TRUSTED_ORIGINS` includes scheme and domain.  
- **Email issues:** Verify TLS/SSL, port, and App Password for sending emails.  
- **n8n errors:** Check logs and webhook URL/token configuration.  
- **Static files 404:** Re-run `collectstatic` and verify WhiteNoise setup.

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) file.
