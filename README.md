# LoanGuard AI
### Intelligent Loan Assessment & Risk Prediction Platform

<p align="center">
  <strong>Turn Financial Data Into Smarter Lending Insights.</strong>
  <br/>
  An end-to-end Machine Learning platform for loan approval prediction, probability estimation, and data-driven risk assessment.
</p>

<p align="center">
  <a href="https://loanguard-ai-1.streamlit.app/"><strong>Explore Live Demo ↗</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/AbdelrhmanAkl/LoanGuard-AI">Source Code</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost-15803D?style=flat-square" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Status-Portfolio%20Project-64748B?style=flat-square" alt="Project Status"/>
</p>

---

## Overview

**LoanGuard AI** is a Machine Learning-powered loan assessment platform that predicts loan approval outcomes from applicant financial profiles, loan characteristics, and credit history.

Built around an optimized **XGBoost classification model**, the platform transforms structured applicant data into probability-based predictions through an automated preprocessing pipeline and an intuitive interactive dashboard.

The goal is to demonstrate how Machine Learning can support financial risk analysis through a practical, accessible, and reproducible application.

Rather than exposing users to complex model internals, LoanGuard AI provides a straightforward experience:

- Enter an applicant's financial information.
- Automatically validate and preprocess the input.
- Generate an approval probability.
- Review the predicted outcome and supporting information.

**The result:** a complete ML application that connects model development, inference, validation, and user experience in one platform.

> **Important:** LoanGuard AI is currently an educational and portfolio demonstration. It is not a certified credit scoring product or an authorized lending decision system. Real-world financial deployment requires independent validation, fairness assessment, regulatory compliance, security controls, and human oversight.

## Live Product

<p align="center">
  <a href="https://loanguard-ai-1.streamlit.app/">
    <strong>Launch LoanGuard AI →</strong>
  </a>
  <br/>
  Experience the interactive loan assessment application.
</p>

**Platform:** Streamlit Community Cloud  
**Application:** https://loanguard-ai-1.streamlit.app/

---

## Product Capabilities

| Capability | Description |
|---|---|
| Intelligent Loan Prediction | Predicts loan approval outcomes using an XGBoost classifier. |
| Probability-Based Assessment | Returns estimated approval and rejection probabilities. |
| Automated Feature Processing | Applies the trained numerical and categorical preprocessing pipeline. |
| Applicant Input Validation | Checks required fields, supported categories, and numerical ranges. |
| Dynamic Financial Calculation | Calculates loan-to-income ratio automatically from applicant inputs. |
| Configurable Classification Threshold | Applies the selected 0.47 decision threshold. |
| Interactive User Interface | Provides an accessible Streamlit application for running assessments. |
| Modular ML Architecture | Separates configuration, preprocessing, model inference, and validation. |

## How It Works

LoanGuard AI follows an end-to-end prediction workflow:

```text
Applicant Financial Information
              |
              v
      Input Validation
              |
              v
   Feature Preparation
              |
              v
  Trained Preprocessing Pipeline
              |
              v
       XGBoost Model
              |
              v
    Approval Probability
              |
              v
     Decision Threshold
              |
              v
     Assessment Results
```

The application uses the same preprocessing artifacts fitted during model development, ensuring that incoming data follows the expected transformation process.

### Assessment Inputs

The model uses nine selected features:

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `loan_amnt` | Requested loan amount |
| `loan_intent` | Purpose of the loan |
| `loan_percent_income` | Loan amount as a proportion of income |
| `cb_person_cred_hist_length` | Length of credit history |
| `credit_score` | Applicant's credit score |
| `previous_loan_defaults_on_file` | Previous loan default history |

The loan-to-income ratio is calculated automatically:

`loan_percent_income = loan_amnt / person_income`

Users do not need to calculate or enter this derived feature manually.

---

## Model Performance

The final XGBoost model was evaluated using a test split.

### Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | **90.41%** |
| Precision | **76.38%** |
| Recall | **82.30%** |
| F1-Score | **79.23%** |
| ROC-AUC | **96.42%** |
| PR-AUC | **90.05%** |

### Classification Results

Confusion matrix:

| Actual / Predicted | Rejected | Approved |
|---|---:|---:|
| Rejected | 6,491 | 509 |
| Approved | 354 | 1,646 |

### Decision Threshold

The final classification threshold is configured to **0.47**.

The model generates an approval probability, which is compared against this threshold to produce the predicted class.

The threshold is a model configuration, not a lender's actual approval policy.

### Evaluation Transparency

The test split was also used in earlier baseline model comparisons during experimentation. The final XGBoost configuration and threshold were not tuned against the test results.

Consequently, these scores should be interpreted as reported experimental results rather than a completely untouched independent benchmark.

---

## Dataset & Model Development

The project uses a loan approval dataset containing:

| Dataset Property | Value |
|---|---:|
| Records | 45,000 |
| Original Features | 13 |
| Target Variable | `loan_status` |
| Prediction Type | Binary Classification |

**Target encoding:**

- `0` — Rejected
- `1` — Approved

The dataset has an imbalanced target distribution, making it important to evaluate more than accuracy alone.

The development workflow includes:

1. Data inspection and quality assessment.
2. Exploratory data analysis.
3. Feature selection and preprocessing.
4. Class imbalance handling.
5. Classification model experimentation.
6. XGBoost model configuration and training.
7. Model evaluation using classification metrics.
8. Exporting the trained preprocessing and model artifacts.
9. Integrating the inference workflow into Streamlit.

### Final Model Configuration

| Component | Implementation |
|---|---|
| Algorithm | XGBoost Classifier |
| Learning Task | Binary Classification |
| Preprocessing | Scikit-learn ColumnTransformer |
| Numerical Transformation | Imputation + RobustScaler |
| Categorical Transformation | Imputation + One-Hot Encoding |
| Classification Threshold | 0.47 |
| Random State | 42 |

---

## Technical Architecture

LoanGuard AI follows a modular architecture designed to keep model inference, input validation, configuration, and the user interface separate.

```text
LoanGuard-AI/
│
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   ├── predictor.py
│   └── validation.py
│
├── models/
│   ├── loanguard_preprocessor.joblib
│   ├── loanguard_xgb_model.json
│   └── model_metadata.json
│
├── data/
│   ├── raw/
│   │   └── loan_data.csv
│   └── processed/
│
├── notebooks/
│   └── LoanGuard_AI_Loan_Approval_Prediction.ipynb
│
├── tests/
│   └── test_model.py
│
├── outputs/
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Application Components

**`app.py` — User Interface**

Handles the Streamlit application, assessment workflow, and presentation of prediction results.

**`src/config.py` — Configuration Management**

Loads project paths, model metadata, feature definitions, and the classification threshold.

**`src/model_loader.py` — Model Artifact Management**

Loads and caches the preprocessing pipeline and trained XGBoost model.

**`src/predictor.py` — Prediction Engine**

Transforms applicant data and generates approval and rejection probabilities.

**`src/validation.py` — Input Validation**

Validates required features, numerical ranges, and categorical values before prediction.

**`models/` — Model Artifacts**

Contains the exported preprocessing pipeline, XGBoost model, and model metadata.

**`notebooks/` — Model Development**

Contains the notebook used for data analysis, experimentation, training, and evaluation.

**`tests/` — Quality Assurance**

Contains automated tests for artifact loading, input validation, and prediction behavior.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing and ML pipeline |
| XGBoost | Classification model |
| Joblib | Preprocessing artifact serialization |
| Streamlit | Interactive web application |
| Plotly | Data visualization |
| Pytest | Automated testing |
| Git & GitHub | Version control and source management |

---

## Run Locally

### Requirements

- Python 3.13
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/LoanGuard-AI.git

cd LoanGuard-AI
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

Open the local application:

`http://localhost:8501`

### 5. Run Automated Tests

```bash
python -m pytest tests/test_model.py -v
```

The test suite covers:

- Model artifact loading.
- Valid application input.
- Invalid input handling.
- Missing feature detection.
- Prediction output validation.
- Probability consistency.

---

## Model Artifacts & Reproducibility

The trained model and preprocessing pipeline are exported separately:

- `loanguard_preprocessor.joblib`
- `loanguard_xgb_model.json`
- `model_metadata.json`

This separation allows the application to load preprocessing and model artifacts independently while preserving the expected feature transformation workflow.

The project uses pinned dependency versions to improve reproducibility between development and deployment environments.

---

## Responsible Use & Limitations

Loan approval is a high-impact financial decision. A machine learning prediction alone is not sufficient to establish a person's creditworthiness or determine lending eligibility.

Before any real-world deployment, a system like this would require:

- Independent validation on representative external data.
- Probability calibration and decision-threshold analysis.
- Fairness and bias testing.
- Explainability and auditability.
- Data privacy and security controls.
- Regulatory and legal review.
- Human oversight and a formal decision governance process.

Historical lending data can encode existing biases. Model probabilities should not be interpreted as guarantees or objective financial outcomes.

**LoanGuard AI is a technical demonstration of an ML-powered assessment workflow, not a production-approved lending system.**

---

## Roadmap

Potential future development directions:

- [ ] SHAP-based prediction explanations.
- [ ] Probability calibration and threshold analysis.
- [ ] Fairness and bias evaluation.
- [ ] Independent external validation.
- [ ] Model monitoring and data drift detection.
- [ ] Assessment history with secure persistence.
- [ ] Exportable assessment reports.
- [ ] Authentication and role-based access.
- [ ] Production deployment architecture and governance controls.

---

## Author

**Abdelrahman Akl**

AI Engineer | Machine Learning | Deep Learning | Agentic AI

Building practical AI applications that connect machine learning research with real-world software experiences.

- **GitHub:** [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- **LinkedIn:** [Abdelrahman Akl](https://www.linkedin.com/in/abdelrahmanakl/)

---

<p align="center">
  <strong>LoanGuard AI</strong>
  <br/>
  Intelligent Assessment. Responsible AI. Better Insights.
  <br/><br/>
  Built with Python, XGBoost, Scikit-learn, and Streamlit.
</p>