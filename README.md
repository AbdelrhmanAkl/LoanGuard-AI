# LoanGuard AI
### Loan Approval Prediction & Risk Assessment — Ready-to-Deploy ML Application

<p align="center">
  <strong>A complete, production-structured machine learning application for loan approval prediction.</strong>
  <br/>
  Trained model · Validated inference pipeline · Interactive dashboard · Automated tests · Full source code.
</p>

<p align="center">
  <a href="https://loanguard-ai-1.streamlit.app/"><strong>▶ Live Demo</strong></a>
  &nbsp;·&nbsp;
  <a href="#whats-included">What's Included</a>
  &nbsp;·&nbsp;
  <a href="#licensing--purchase">Licensing & Purchase</a>
  &nbsp;·&nbsp;
  <a href="#contact">Contact</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Model-XGBoost-15803D?style=flat-square" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Tests-Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" alt="Pytest"/>
</p>

---

## Why LoanGuard AI

Building a credible ML application takes more than a notebook. You need a trained model, a reproducible preprocessing pipeline, input validation, a usable interface, tests, and clean structure.

**LoanGuard AI delivers all of it in one package**, so you can demo, customize, integrate, or build on it instead of starting from zero.

| Benefit | What it means for you |
|---|---|
| **Complete end-to-end workflow** | From raw applicant input to probability and decision, nothing missing. |
| **Strong, transparent metrics** | 96.42% ROC-AUC with full confusion matrix and honest evaluation notes. |
| **Clean modular code** | Config, preprocessing, inference, validation, and UI are separated. |
| **Reproducible** | Pinned dependencies and exported model artifacts. |
| **Tested** | Automated test suite covering loading, validation, and prediction. |
| **Easy to deploy** | Runs locally or on Streamlit Community Cloud in minutes. |

---

## Live Demo

Try the application before you decide:

**https://loanguard-ai-1.streamlit.app/**

Enter an applicant profile and get an approval probability and predicted outcome instantly.

<!-- Add screenshots here -->
<!--
<p align="center">
  <img src="outputs/screenshot_dashboard.png" width="85%" alt="Dashboard"/>
</p>
-->

---

## Who Is This For

- **Developers & ML engineers** who want a solid base for a credit-scoring or risk-assessment project.
- **Fintech startups & agencies** that need a working prototype to show clients or investors.
- **Students & educators** looking for a complete, well-structured applied ML case study.
- **Data science teams** that want a reference architecture for model → validation → UI.

---

## What's Included

| Deliverable | Details |
|---|---|
| **Full source code** | Streamlit app and modular `src/` package. |
| **Trained model artifacts** | XGBoost model (`.json`), preprocessing pipeline (`.joblib`), and metadata. |
| **Training notebook** | Complete EDA, preprocessing, experimentation, training, and evaluation. |
| **Automated tests** | Pytest suite for artifact loading, validation, and predictions. |
| **Sample dataset** | 45,000-record loan dataset used for training (see note on data licensing below). |
| **Documentation** | This README plus inline code documentation. |
| **Deployment-ready config** | `requirements.txt` with pinned versions and Streamlit configuration. |

---

## Features

| Capability | Description |
|---|---|
| Loan approval prediction | XGBoost binary classifier. |
| Probability output | Returns approval and rejection probabilities, not just a label. |
| Automated preprocessing | Uses the same fitted `ColumnTransformer` as training, preventing train/serve mismatch. |
| Input validation | Checks required fields, supported categories, and numeric ranges before prediction. |
| Derived feature calculation | `loan_percent_income` is computed automatically from loan amount and income. |
| Configurable threshold | Decision threshold set to **0.47** and easily adjustable in config. |
| Interactive dashboard | Clean Streamlit interface for running assessments. |
| Modular architecture | Easy to extend, swap models, or embed in another system. |

---

## How It Works

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
     Decision Threshold (0.47)
              |
              v
     Assessment Results
```

### Model Inputs

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `loan_amnt` | Requested loan amount |
| `loan_intent` | Purpose of the loan |
| `loan_percent_income` | Loan amount ÷ income (calculated automatically) |
| `cb_person_cred_hist_length` | Length of credit history |
| `credit_score` | Applicant's credit score |
| `previous_loan_defaults_on_file` | Previous default history |

---

## Model Performance

Evaluated on a held-out test split (9,000 records).

| Metric | Score |
|---|---:|
| Accuracy | **90.41%** |
| Precision | **76.38%** |
| Recall | **82.30%** |
| F1-Score | **79.23%** |
| ROC-AUC | **96.42%** |
| PR-AUC | **90.05%** |

**Confusion matrix**

| Actual \ Predicted | Rejected | Approved |
|---|---:|---:|
| Rejected | 6,491 | 509 |
| Approved | 354 | 1,646 |

**Evaluation transparency:** the test split was also used in earlier baseline model comparisons. The final XGBoost configuration and threshold were not tuned against test results, but these figures should be read as reported experimental results rather than a fully independent benchmark. We recommend validating on your own data before relying on them.

---

## Model Configuration

| Component | Implementation |
|---|---|
| Algorithm | XGBoost Classifier |
| Task | Binary classification (`0` = Rejected, `1` = Approved) |
| Preprocessing | Scikit-learn `ColumnTransformer` |
| Numerical features | Imputation + `RobustScaler` |
| Categorical features | Imputation + One-Hot Encoding |
| Decision threshold | 0.47 |
| Random state | 42 |

---

## Project Structure

```text
LoanGuard-AI/
├── app.py                      # Streamlit application
├── src/
│   ├── config.py               # Paths, metadata, features, threshold
│   ├── model_loader.py         # Loads and caches model artifacts
│   ├── predictor.py            # Transformation and probability generation
│   └── validation.py           # Input validation
├── models/
│   ├── loanguard_preprocessor.joblib
│   ├── loanguard_xgb_model.json
│   └── model_metadata.json
├── data/
│   ├── raw/loan_data.csv
│   └── processed/
├── notebooks/
│   └── LoanGuard_AI_Loan_Approval_Prediction.ipynb
├── tests/
│   └── test_model.py
├── outputs/
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

**Requirements:** Python 3.13, Git, pip.

```bash
# 1. Clone
git clone https://github.com/AbdelrhmanAkl/LoanGuard-AI.git
cd LoanGuard-AI

# 2. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open **http://localhost:8501**.

**Run the tests**

```bash
python -m pytest tests/test_model.py -v
```

---

## Customization & Integration

The modular design makes it straightforward to adapt:

- **Change the threshold** in `src/config.py` to trade precision for recall.
- **Retrain on your own data** using the included notebook and re-export the artifacts.
- **Swap the model** (e.g., LightGBM, CatBoost) while keeping the same pipeline structure.
- **Embed the predictor** (`src/predictor.py`) in a FastAPI/Flask backend or another service.
- **Rebrand the interface** by editing `app.py` and `.streamlit/config.toml`.

---

## Technology Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Joblib · Streamlit · Plotly · Pytest

---

## Licensing & Purchase

| Option | Includes |
|---|---|
| **Standard License** | Full source code and artifacts for use in **one** project or product. |
| **Extended License** | Standard + rights for commercial redistribution / client work. |
| **Custom** | Adaptation to your data, branding, or deployment environment. |

> **Fill in before publishing:** pricing, exact license terms, and refund/support policy.

**License terms (summary — replace with your final terms):**

- Permitted: use, modify, and deploy within the scope of the purchased license.
- Not permitted: reselling or redistributing the source code as a standalone product unless covered by an Extended License.
- The software is provided "as is", without warranty of any kind.

**Data licensing note:** the included training dataset comes from a public source. Buyers who plan commercial use should review the original dataset's license and terms, or retrain on their own data.

---

## Support

- Setup assistance for the included deliverables.
- Bug fixes for issues in the delivered code.
- Optional paid customization and integration services.

> **Fill in before publishing:** support duration and response time.

---

## Responsible Use & Limitations

Loan approval is a high-impact financial decision. **LoanGuard AI is a technical demonstration of an ML-based assessment workflow. It is not a certified credit-scoring product or an authorized lending decision system.**

Before using any similar system in real-world lending, you are responsible for:

- Independent validation on representative external data.
- Probability calibration and threshold analysis.
- Fairness and bias testing across protected groups.
- Explainability and auditability.
- Data privacy and security controls.
- Regulatory and legal compliance in your jurisdiction.
- Human oversight and a formal decision-governance process.

Historical lending data can encode existing biases. Model probabilities are estimates, not guarantees or objective financial outcomes. The 0.47 threshold is a model configuration, not a lender's approval policy.

---

## Roadmap

- [ ] SHAP-based prediction explanations
- [ ] Probability calibration and threshold analysis
- [ ] Fairness and bias evaluation
- [ ] Independent external validation
- [ ] Model monitoring and drift detection
- [ ] REST API (FastAPI) endpoint
- [ ] Secure assessment history
- [ ] Exportable PDF assessment reports
- [ ] Authentication and role-based access

---

## FAQ

**Can I use this in production lending?**
Not as-is. It is a demonstration-grade system. Production use requires the validation, fairness, compliance, and governance steps listed above.

**Can I retrain it on my own data?**
Yes. The notebook documents the full workflow, and the artifacts can be re-exported in the same format.

**Does it need a GPU?**
No. Inference and training run on a standard CPU.

**Can I deploy it myself?**
Yes. It runs locally or on Streamlit Community Cloud, and can be containerized for other environments.

---

## Contact

**Abdelrahman Akl** — AI Engineer | Machine Learning | Deep Learning | Agentic AI

- GitHub: [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- LinkedIn: [Abdelrahman Akl](https://www.linkedin.com/in/abdelrahmanakl/)
- Email: _add your email here_

---

<p align="center">
  <strong>LoanGuard AI</strong><br/>
  Intelligent Assessment. Responsible AI. Better Insights.<br/><br/>
  Built with Python, XGBoost, Scikit-learn, and Streamlit.
</p>