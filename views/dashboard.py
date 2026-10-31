"""Dashboard: landing page. Shows only real, offline model-evaluation metrics."""
import streamlit as st

from src.config import DECISION_THRESHOLD, TEST_METRICS
from src.ui import theme as ui

with st.container(key="hero"):
    ui.render("""
        <div class="lg-hero-eyebrow">Loan risk intelligence</div>
        <h1 class="lg-hero-title">Smarter Risk Assessment.<br>Data-Driven Decisions.</h1>
        <p class="lg-hero-lead">Check how likely a loan application is to be approved. Answer a few
        simple questions and get a clear, transparent result from a trained XGBoost model.</p>""")
    if st.button("Start New Assessment", type="primary"):
        st.switch_page("views/assessment.py")
    ui.render(f"""
        <div class="lg-pills"><span class="lg-pill">XGBoost model</span>
        <span class="lg-pill">8 simple inputs</span>
        <span class="lg-pill">Decision threshold {DECISION_THRESHOLD:.2f}</span></div>""")

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
    (1, "Enter the details", "Applicant, loan and credit information. The loan-to-income ratio is calculated for you."),
    (2, "The model scores it", "A trained XGBoost model estimates the probability that the application is approved."),
    (3, "Review the result", f"Predicted as approved when the probability reaches the {DECISION_THRESHOLD:.0%} decision threshold."),
]):
    with col:
        ui.step_card(n, title, body)
ui.disclaimer()
