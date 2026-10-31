import streamlit as st

from src.config import (
    PROJECT_NAME,
    MODEL_TYPE,
    DECISION_THRESHOLD,
    TEST_METRICS,
    MODEL_VERSIONS,
)
from src.predictor import predict_loan
from src.validation import validate_application


st.set_page_config(
    page_title="LoanGuard AI | Loan Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- Custom Styling -----------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F7F9FC;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.9);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E8EDF3;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .brand {
        color: #0F766E;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #172B4D;
        font-size: clamp(30px, 4vw, 43px);
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.15;
        margin: 0;
    }

    .hero-description {
        color: #718096;
        font-size: 15px;
        line-height: 1.7;
        margin-top: 12px;
        max-width: 680px;
    }

    .section-title {
        color: #172B4D;
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .section-description {
        color: #8290A5;
        font-size: 13px;
        margin-bottom: 20px;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid #E8EDF3;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 18px;
    }

    .result-approved {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 14px;
        padding: 22px;
    }

    .result-rejected {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-radius: 14px;
        padding: 22px;
    }

    .result-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 800;
        margin: 0;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8EDF3;
        border-radius: 12px;
        padding: 16px;
        min-height: 95px;
    }

    .metric-label {
        color: #8290A5;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #172B4D;
        font-size: 23px;
        font-weight: 800;
    }

    .sidebar-brand {
        color: #172B4D;
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        color: #8290A5;
        font-size: 12px;
        margin-top: 3px;
    }

    .disclaimer {
        color: #718096;
        background: #F0F4F8;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 12px;
        line-height: 1.7;
        margin-top: 20px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #0F766E;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
        min-height: 44px;
        transition: background 0.2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #115E59;
        color: white;
        border: none;
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E8EDF3;
        border-radius: 16px;
        padding: 24px;
    }

    label, .stSelectbox label, .stNumberInput label {
        color: #34445D !important;
        font-weight: 600 !important;
    }

    hr {
        border-color: #E8EDF3;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------- Sidebar ---------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">LoanGuard <span style="color:#0F766E;">AI</span></div>
        <div class="sidebar-subtitle">Intelligent Loan Risk Assessment</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**MODEL INFORMATION**")
    st.caption(f"Model: {MODEL_TYPE}")
    st.caption(f"Decision threshold: {DECISION_THRESHOLD:.0%}")
    st.caption(f"Input features: 9")

    st.divider()

    st.markdown("**MODEL EVALUATION**")
    st.caption("Reported test-set metrics from the training experiment.")

    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]

    for label, key in metric_labels:
        st.metric(label, f"{TEST_METRICS[key]:.2%}")

    st.divider()

    with st.expander("Technical versions"):
        for package, version in MODEL_VERSIONS.items():
            st.caption(f"{package}: {version}")

    st.caption("LoanGuard AI · Prototype")


# --------------------------------- Header ----------------------------------

st.markdown('<div class="brand">AI-powered financial intelligence</div>', unsafe_allow_html=True)

st.markdown(
    '<h1 class="hero-title">Loan Approval<br>Prediction</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="hero-description">
        Evaluate a loan application using a machine learning model.
        Enter the applicant's financial profile to generate a prediction
        and explore the model's estimated approval probability.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ----------------------------- Application Form ---------------------------

st.markdown(
    """
    <div class="section-title">Applicant information</div>
    <div class="section-description">
        Complete the fields below to evaluate a loan application.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("loan_application_form"):
    st.markdown("**01 / Personal profile**")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input(
            "Applicant age",
            min_value=18,
            max_value=100,
            value=28,
            step=1,
            help="Applicant's age in years.",
        )

    with col2:
        person_income = st.number_input(
            "Annual income",
            min_value=1,
            max_value=10_000_000,
            value=60000,
            step=1000,
            help="Annual income in the dataset's currency units.",
        )

    person_home_ownership = st.selectbox(
        "Home ownership",
        options=["RENT", "OWN", "MORTGAGE", "OTHER"],
        format_func=lambda value: {
            "RENT": "Rent",
            "OWN": "Own",
            "MORTGAGE": "Mortgage",
            "OTHER": "Other",
        }[value],
    )

    st.divider()
    st.markdown("**02 / Loan details**")

    col3, col4 = st.columns(2)

    with col3:
        loan_amnt = st.number_input(
            "Requested loan amount",
            min_value=1,
            max_value=10_000_000,
            value=15000,
            step=500,
        )

    with col4:
        loan_intent = st.selectbox(
            "Loan purpose",
            options=[
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
            ],
            format_func=lambda value: {
                "PERSONAL": "Personal",
                "EDUCATION": "Education",
                "MEDICAL": "Medical",
                "VENTURE": "Business venture",
                "HOMEIMPROVEMENT": "Home improvement",
                "DEBTCONSOLIDATION": "Debt consolidation",
            }[value],
        )

    st.caption(
        "The loan-to-income ratio is calculated automatically from the "
        "requested loan amount and annual income."
    )

    st.divider()
    st.markdown("**03 / Credit profile**")

    col5, col6 = st.columns(2)

    with col5:
        cb_person_cred_hist_length = st.number_input(
            "Credit history length (years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
        )

    with col6:
        credit_score = st.number_input(
            "Credit score",
            min_value=300,
            max_value=850,
            value=680,
            step=1,
        )

    previous_loan_defaults_on_file = st.selectbox(
        "Previous loan defaults on file",
        options=["No", "Yes"],
        format_func=lambda value: "No" if value == "No" else "Yes",
    )

    st.write("")

    submitted = st.form_submit_button(
        "Evaluate loan application",
        use_container_width=True,
    )


# ------------------------------- Prediction -------------------------------

if submitted:
    loan_percent_income = loan_amnt / person_income

    application_data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
    }

    validation_errors = validate_application(application_data)

    st.divider()

    st.markdown(
        '<div class="section-title">Assessment results</div>',
        unsafe_allow_html=True,
    )

    if validation_errors:
        st.error("Please review the following input issues:")
        for error in validation_errors:
            st.write(f"- {error}")

    else:
        try:
            with st.spinner("Analyzing the application..."):
                result = predict_loan(application_data)

            is_approved = result["prediction"] == 1
            approval_probability = result["approval_probability"]
            rejection_probability = result["rejection_probability"]

            result_class = "result-approved" if is_approved else "result-rejected"
            result_color = "#047857" if is_approved else "#BE123C"

            st.markdown(
                f"""
                <div class="{result_class}">
                    <div class="result-label" style="color:{result_color};">
                        Model prediction
                    </div>
                    <p class="result-value" style="color:{result_color};">
                        {result["decision"]}
                    </p>
                    <div style="color:#64748B;font-size:13px;margin-top:7px;">
                        Based on the configured classification threshold.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            probability_col, threshold_col = st.columns(2)

            with probability_col:
                st.markdown("**Approval probability**")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:800;color:#0F766E;'>"
                    f"{approval_probability:.2%}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(approval_probability)

            with threshold_col:
                st.markdown("**Decision threshold**")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:800;color:#172B4D;'>"
                    f"{result['threshold']:.0%}</div>",
                    unsafe_allow_html=True,
                )
                st.progress(result["threshold"])

            st.write("")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(
                "Approval probability",
                f"{approval_probability:.2%}",
            )
            metric_col2.metric(
                "Rejection probability",
                f"{rejection_probability:.2%}",
            )
            metric_col3.metric(
                "Loan-to-income ratio",
                f"{loan_percent_income:.2%}",
            )

            with st.expander("View submitted application details"):
                st.json(application_data)

            st.caption(
                "The prediction is generated by a machine learning model. "
                "It is not a guarantee of loan approval or rejection."
            )

        except Exception as error:
            st.error("The application could not be evaluated.")
            st.exception(error)


# -------------------------------- Disclaimer ------------------------------

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important notice</strong><br>
        LoanGuard AI is an educational machine learning prototype.
        Its predictions are not financial advice and must not be used
        as the sole basis for real lending, credit approval, or rejection.
        Real-world credit decisions require appropriate regulatory review,
        fairness assessment, explainability, data-quality controls,
        and human oversight.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.caption("LoanGuard AI · Machine Learning Prototype")
