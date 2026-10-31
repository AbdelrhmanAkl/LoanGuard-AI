# LoanGuard AI

### AI-Powered Loan Approval Prediction & Risk Assessment

<p align="center">
  <strong>Smarter Risk Assessment. Data-Driven Decisions.</strong>
</p>

<p align="center">
  <a href="https://loanguard-ai-1.streamlit.app/">Live Demo</a> •
  <a href="https://github.com/AbdelrhmanAkl/LoanGuard-AI">GitHub Repository</a>
</p>

---

## Overview

**LoanGuard AI** is an end-to-end Machine Learning application designed to predict loan approval outcomes using applicant financial information, loan characteristics, and credit history.

Powered by **XGBoost**, the system combines a structured preprocessing pipeline with probability-based predictions and an interactive Streamlit dashboard.

The application allows users to enter applicant information, estimate loan approval probability, and view the model's predicted outcome against a configurable decision threshold.

> **Disclaimer:** LoanGuard AI is an educational and portfolio project. It is not a certified credit scoring system and must not be used to make real-world lending or financial decisions.

## Live Application

Explore the interactive application:

**[Launch LoanGuard AI](https://loanguard-ai-1.streamlit.app/)**

## Key Features

- **Machine Learning Prediction:** XGBoost-based binary classification.
- **Automated Preprocessing:** Numerical and categorical feature transformations.
- **Probability Estimation:** Approval and rejection probability outputs.
- **Configurable Decision Threshold:** Classification threshold set to `0.47`.
- **Interactive Dashboard:** Clean and accessible Streamlit interface.
- **Input Validation:** Checks required fields, allowed categories, and numerical ranges.
- **Modular Architecture:** Separate modules for configuration, model loading, prediction, and validation.
- **Model Testing:** Automated tests using Pytest.
- **Reproducibility:** Version-pinned dependencies and saved model artifacts.

## Machine Learning Pipeline

The project follows a structured machine learning workflow:

1. Load and inspect the loan application dataset.
2. Perform data quality checks and exploratory analysis.
3. Handle numerical and categorical features.
4. Address class imbalance during model training.
5. Train and evaluate classification models.
6. Select and configure the final XGBoost model.
7. Export preprocessing and model artifacts separately.
8. Integrate the trained model into a Streamlit application.

### Final Model

| Component | Configuration |
|---|---|
| Algorithm | XGBoost Classifier |
| Task | Binary Classification |
| Preprocessing | Scikit-learn ColumnTransformer |
| Numerical Processing | Imputation + RobustScaler |
| Categorical Processing | Imputation + One-Hot Encoding |
| Decision Threshold | 0.47 |
| Random State | 42 |

### Input Features

The final model uses nine features:

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `loan_amnt` | Requested loan amount |
| `loan_intent` | Purpose of the loan |
| `loan_percent_income` | Loan amount relative to income |
| `cb_person_cred_hist_length` | Credit history length |
| `credit_score` | Applicant's credit score |
| `previous_loan_defaults_on_file` | Previous loan default history |

The application calculates `loan_percent_income` automatically using the requested loan amount and annual income.

## Dataset

The project uses a loan approval dataset with the following characteristics:

| Property | Value |
|---|---:|
| Total Records | 45,000 |
| Total Original Columns | 14 |
| Selected Model Features | 9 |
| Target Variable | `loan_status` |
| Task | Binary Classification |

**Target encoding:**
- `0` — Rejected
- `1` — Approved

The dataset contains an imbalanced target distribution, making precision, recall, and F1-score important evaluation metrics.

Dataset redistribution must comply with the original dataset's license and terms.

## Model Performance

The final XGBoost model was evaluated using a test split.

| Metric | Test Score |
|---|---:|
| Accuracy | 90.41% |
| Precision | 76.38% |
| Recall | 82.30% |
| F1-Score | 79.23% |
| ROC-AUC | 96.42% |
| PR-AUC | 90.05% |

### Confusion Matrix

| Actual / Predicted | Rejected | Approved |
|---|---:|---:|
| Rejected | 6,491 | 509 |
| Approved | 354 | 1,646 |

The model's classification threshold is configured to `0.47`.

**Evaluation note:** The test split was also used in earlier baseline model comparisons during experimentation. The final XGBoost configuration and threshold were not tuned against the test results, but the reported evaluation should not be interpreted as a completely untouched independent benchmark.

## Project Architecture

```text
LoanGuard-AI/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   └── loan_data.csv
│   └── processed/
│
├── models/
│   ├── loanguard_preprocessor.joblib
│   ├── loanguard_xgb_model.json
│   └── model_metadata.json
│
├── notebooks/
│   └── LoanGuard_AI_Loan_Approval_Prediction.ipynb
│
├── outputs/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   ├── predictor.py
│   └── validation.py
│
├── tests/
│   └── test_model.py
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
└── .gitignore
```

### Core Modules

- **`app.py`** — Streamlit user interface and application workflow.
- **`src/config.py`** — Project paths, model metadata, and configuration.
- **`src/model_loader.py`** — Loads and caches model artifacts.
- **`src/predictor.py`** — Applies preprocessing and generates predictions.
- **`src/validation.py`** — Validates applicant input data.
- **`models/`** — Serialized preprocessing pipeline, XGBoost model, and metadata.
- **`notebooks/`** — Model development, experimentation, and evaluation.
- **`tests/`** — Automated application and model tests.

## Installation & Local Setup

### Prerequisites

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

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Running Tests

Run the automated test suite:

```bash
python -m pytest tests/test_model.py -v
```

The tests cover:
- Model artifact loading.
- Valid application input.
- Invalid input handling.
- Missing feature detection.
- Prediction output validation.
- Probability consistency.

## Technologies & Tools

- **Python 3.13**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **Streamlit**
- **Plotly**
- **Pytest**
- **Git & GitHub**

## Limitations & Future Improvements

Potential future improvements include:

- Probability calibration and threshold optimization based on business costs.
- Explainable AI using SHAP.
- Fairness and bias evaluation across relevant applicant groups.
- More extensive cross-validation and independent external evaluation.
- Model monitoring and data drift detection.
- Improved model documentation and dataset provenance.
- Human review and governance workflows for any real-world application.

Historical lending data can contain bias. Model predictions and probabilities should not be interpreted as objective or guaranteed financial outcomes.

## Author

**Abdelrahman Akl**

AI Engineer | Machine Learning | Deep Learning | Agentic AI

- **GitHub:** [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- **LinkedIn:** [Abdelrahman Akl](https://www.linkedin.com/in/abdelrahmanakl/)

---

<p align="center">
  Built with Python, XGBoost, and Streamlit.
</p>