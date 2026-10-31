"""Loan Assessment: guided form and prediction result.

Prediction logic lives in src.predictor; this page only collects input,
calls it and presents the exact output.
"""
import logging

import plotly.graph_objects as go
import streamlit as st

from src.predictor import predict_loan
from src.ui import theme as ui
from src.validation import validate_application

log = logging.getLogger(__name__)

HOME = {"RENT": "Rent", "OWN": "Own", "MORTGAGE": "Mortgage", "OTHER": "Other"}
INTENT = {
    "PERSONAL": "Personal", "EDUCATION": "Education", "MEDICAL": "Medical",
    "VENTURE": "Business venture", "HOMEIMPROVEMENT": "Home improvement",
    "DEBTCONSOLIDATION": "Debt consolidation",
}
LABELS = {
    "person_age": "Age", "person_income": "Annual income", "loan_amnt": "Requested loan amount",
    "cb_person_cred_hist_length": "Credit history length", "credit_score": "Credit score",
    "person_home_ownership": "Home ownership", "loan_intent": "Loan purpose",
    "previous_loan_defaults_on_file": "Previous loan defaults",
}


def friendly_errors(errors: list[str]) -> list[str]:
    """Translate validation messages (src.validation is unchanged) for end users."""
    out = []
    for error in errors:
        if error.startswith("loan_percent_income"):
            out.append("The requested loan is larger than the annual income. This model supports "
                       "a loan-to-income ratio of 100% or less.")
            continue
        for key, label in LABELS.items():
            if error.startswith(key):
                error = label + error[len(key):]
                break
        out.append(error)
    return out


def probability_chart(p: float, threshold: float, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=[1], y=[""], orientation="h", marker_color="#E2E8F0", hoverinfo="skip")
    fig.add_bar(x=[p], y=[""], orientation="h", marker_color=color,
                hovertemplate="Approval probability: %{x:.2%}<extra></extra>")
    fig.add_vline(x=threshold, line_width=2, line_color=ui.NAVY,
                  annotation_text=f"Threshold {threshold:.0%}", annotation_position="top")
    fig.update_layout(
        barmode="overlay", showlegend=False, height=120, bargap=0.1,
        margin=dict(l=0, r=0, t=34, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=ui.MUTED),
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
    fig.update_yaxes(visible=False)
    return fig


def render_result(result: dict) -> None:
    approved = result["prediction"] == 1
    p, threshold = result["approval_probability"], result["threshold"]
    relation = "at or above" if approved else "below"
    ui.render(f"""
        <div class="lg-verdict {'ok' if approved else 'no'}">
          <div class="lg-eyebrow" style="color:inherit">Model prediction</div>
          <div class="lg-verdict-title">{result['decision']}</div>
          <p>The model's estimated approval probability ({p:.2%}) is {relation} the {threshold:.0%} decision threshold.</p>
        </div>""")
    st.plotly_chart(probability_chart(p, threshold, ui.TEAL if approved else ui.DANGER_MUTED),
                    width="stretch", config={"displayModeBar": False})
    for col, (label, value) in zip(st.columns(3), [
        ("Approval", f"{p:.2%}"), ("Rejection", f"{result['rejection_probability']:.2%}"),
        ("Threshold", f"{threshold:.0%}"),
    ]):
        with col:
            ui.metric_card(label, value)
    ui.disclaimer()


ui.header("Loan assessment", "New loan assessment",
          "Answer eight short questions. Hover the (?) icons if you are unsure what a field means.")

left, right = st.columns([3, 2], gap="large")

with left:
    ui.section("1 · Applicant information")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 100, 28, 1, key="lg_in_age", help="Applicant's age in years (18-100).")
    income = c2.number_input("Annual income ($)", 1, 10_000_000, 60_000, 1_000, key="lg_in_income",
                             help="Total yearly income before tax.")
    home = st.selectbox("Home ownership", list(HOME), format_func=HOME.get, key="lg_in_home",
                        help="Current housing situation of the applicant.")

    ui.section("2 · Loan details")
    c3, c4 = st.columns(2)
    amount = c3.number_input("Requested loan amount ($)", 1, 10_000_000, 15_000, 500, key="lg_in_amount",
                             help="How much the applicant wants to borrow.")
    intent = c4.selectbox("Loan purpose", list(INTENT), format_func=INTENT.get, key="lg_in_intent",
                          help="What the loan will be used for.")

    ui.section("3 · Credit profile")
    c5, c6 = st.columns(2)
    history = c5.number_input("Credit history length (years)", 0, 100, 5, 1, key="lg_in_history",
                              help="How many years the applicant has had credit.")
    score = c6.number_input("Credit score", 300, 850, 680, 1, key="lg_in_score",
                            help="Credit score on the 300-850 scale.")
    defaults = st.selectbox("Previous loan defaults", ["No", "Yes"], key="lg_in_defaults",
                            help="Has the applicant ever defaulted on a loan?")
    st.caption("Amounts use the training dataset's currency units, shown as $.")
    evaluate = st.button("Evaluate application", type="primary", width="stretch")

ratio = amount / income  # derived feature, never entered manually
data = {
    "person_age": age, "person_income": income, "person_home_ownership": home,
    "loan_amnt": amount, "loan_intent": intent, "loan_percent_income": ratio,
    "cb_person_cred_hist_length": history, "credit_score": score,
    "previous_loan_defaults_on_file": defaults,
}

if evaluate:
    errors = validate_application(data)
    if errors:
        st.session_state["lg_state"] = {"inputs": data, "errors": friendly_errors(errors)}
    else:
        try:
            with left, st.spinner("Scoring application..."):
                st.session_state["lg_state"] = {"inputs": data, "result": predict_loan(data)}
        except Exception:
            log.exception("Prediction failed")
            st.session_state["lg_state"] = {"inputs": data, "failed": True}

with right:
    warn = ('<div class="lg-metric-note" style="color:#B4474F">Loan exceeds annual income; '
            'the model supports 100% or less.</div>' if ratio > 1 else "")
    ui.card(f'<div class="lg-eyebrow">Application summary</div>'
            f'<div class="lg-metric-label">Loan-to-income ratio (calculated)</div>'
            f'<div class="lg-metric-value">{ratio:.2%}</div>'
            f'<div class="lg-metric-note">{ui.money(amount)} requested against {ui.money(income)} annual income.</div>{warn}')
    st.write("")
    state = st.session_state.get("lg_state")
    if state is None:
        ui.card('<div class="lg-metric-label">No assessment yet</div>'
                '<div class="lg-metric-note">Complete the form and select <strong>Evaluate application</strong> '
                'to see the model prediction here.</div>')
    elif state["inputs"] != data:
        st.info("Inputs changed since the last evaluation. Select Evaluate application to refresh.")
    elif state.get("errors"):
        st.error("Please review these inputs:")
        for message in state["errors"]:
            st.write(f"- {message}")
    elif state.get("failed"):
        st.error("We couldn't evaluate this application. Please check the inputs and try again.")
    else:
        render_result(state["result"])
