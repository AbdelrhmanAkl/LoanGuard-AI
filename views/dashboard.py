"""Dashboard: landing page. Shows only real, offline model-evaluation metrics."""
import streamlit as st

from src.config import DECISION_THRESHOLD, TEST_METRICS
from src.ui import theme as ui

ui.header(
    "Loan risk intelligence",
    "Smarter Risk Assessment.<br>Data-Driven Decisions.",
    "LoanGuard AI estimates how likely a loan application is to be approved, using "
    "applicant, loan and credit-history details scored by a trained XGBoost model.",
)

if st.button("Start New Assessment", type="primary"):
    st.switch_page("views/assessment.py")

st.write("")
ui.section("Model performance", "Offline evaluation on a held-out test split. These are not live business metrics.")

KPIS = [
    ("Accuracy", "accuracy", "Share of test applications classified correctly."),
    ("Precision", "precision", "Of predicted approvals, how many were truly approved."),
    ("Recall", "recall", "Of truly approved applications, how many the model found."),
    ("ROC-AUC", "roc_auc", "How well the model separates approved from rejected."),
]
for col, (label, key, note) in zip(st.columns(4), KPIS):
    with col:
        ui.metric_card(label, f"{TEST_METRICS[key]:.2%}", note)
st.page_link("views/performance.py", label="See full model performance", icon=":material/arrow_forward:")

st.write("")
ui.section("How an assessment works", "Three steps, no technical knowledge needed.")
for col, (n, title, body) in zip(st.columns(3), [
    (1, "Enter the details", "Provide applicant, loan and credit information. The loan-to-income ratio is calculated for you."),
    (2, "The model scores it", "A trained XGBoost model estimates the probability that the application is approved."),
    (3, "Review the result", f"The application is predicted as approved when the probability reaches the {DECISION_THRESHOLD:.0%} decision threshold."),
]):
    with col:
        ui.step_card(n, title, body)
ui.disclaimer()
