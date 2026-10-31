"""About the Model: plain-language description, features and limitations."""
import streamlit as st

from src.config import DECISION_THRESHOLD, MODEL_TYPE, MODEL_VERSIONS
from src.ui import theme as ui

ui.header("About the model", "What LoanGuard AI does",
          "A machine learning prototype that estimates loan approval probability for educational use.")

left, right = st.columns(2, gap="large")
with left:
    ui.section("Inputs the model uses")
    st.markdown("""
| Input | Meaning |
|---|---|
| Age | Applicant's age in years |
| Annual income | Yearly income |
| Home ownership | Rent, own, mortgage or other |
| Loan amount | Amount requested |
| Loan purpose | What the loan is for |
| Loan-to-income ratio | Calculated: loan amount / annual income |
| Credit history length | Years of credit history |
| Credit score | 300-850 scale |
| Previous defaults | Whether a default is on file |
""")
with right:
    ui.section("How a result is produced")
    ui.card(f'<div class="lg-metric-note">Your inputs are cleaned and scaled by the saved preprocessing '
            f'pipeline, then scored by an {MODEL_TYPE} trained on 45,000 historical loan records. The '
            f'model outputs an approval probability, and the application is predicted as approved when '
            f'that probability reaches {DECISION_THRESHOLD:.2f}.</div>')
    st.write("")
    ui.section("Limitations")
    ui.card('<div class="lg-metric-note">Historical lending data can contain bias, and probabilities should '
            'not be read as objective or guaranteed outcomes. The project has not been through fairness '
            'evaluation, external validation or calibration. Real credit decisions need regulatory review, '
            'explainability and human oversight.</div>')

st.write("")
with st.expander("Technical versions"):
    for package, version in MODEL_VERSIONS.items():
        st.caption(f"{package}: {version}")
st.link_button("View source on GitHub", "https://github.com/AbdelrhmanAkl/LoanGuard-AI")
ui.disclaimer()
