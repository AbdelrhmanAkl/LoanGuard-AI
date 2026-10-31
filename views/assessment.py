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


def gauge(p: float, threshold: float, color: str) -> go.Figure:
    """Gauge showing the exact model probability against the decision threshold."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p,
        number={"valueformat": ".2%", "font": {"size": 40, "color": ui.NAVY}},
        title={"text": "Approval probability", "font": {"size": 14, "color": ui.MUTED}},
        gauge={
            "axis": {"range": [0, 1], "tickvals": [0, .25, .5, .75, 1], "tickformat": ".0%",
                     "tickfont": {"size": 11, "color": ui.MUTED}},
            "bar": {"color": color, "thickness": 0.3}, "bgcolor": "#FFFFFF", "borderwidth": 0,
            "steps": [{"range": [0, threshold], "color": "#FBEFEF"}, {"range": [threshold, 1], "color": "#E6F6F3"}],
            "threshold": {"line": {"color": ui.NAVY, "width": 3}, "thickness": 0.9, "value": threshold},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=44, r=44, t=50, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans, sans-serif"))
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
    st.plotly_chart(gauge(p, threshold, ui.TEAL if approved else ui.DANGER_MUTED),
                    width="stretch", config={"displayModeBar": False})
    st.caption(f"The dark marker shows the {threshold:.0%} decision threshold.")
    ui.stats_row([("Approval", f"{p:.2%}"), ("Rejection", f"{result['rejection_probability']:.2%}"),
                  ("Threshold", f"{threshold:.0%}")])
    ui.disclaimer()


ui.header("Loan assessment", "Assess a loan application",
          "Answer a few simple questions. Hover the (?) icons if you are unsure what a field means.")

left, right = st.columns([5, 4], gap="large")

with left:
    with st.container(border=True, key="card_applicant"):
        ui.card_title(1, "Applicant")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 100, 28, key="lg_in_age", help="Applicant's age in years.")
        with c2:
            income = st.number_input("Annual income ($)", 1, 10_000_000, 60_000, 1_000, format="%d",
                                     key="lg_in_income", help="Total yearly income before tax.")
        home = st.segmented_control("Home ownership", list(HOME), format_func=HOME.get, default="RENT",
                                    key="lg_in_home", help="Current housing situation.") or "RENT"

    with st.container(border=True, key="card_loan"):
        ui.card_title(2, "Loan")
        c3, c4 = st.columns(2)
        with c3:
            amount = st.number_input("Requested amount ($)", 1, 10_000_000, 15_000, 500, format="%d",
                                     key="lg_in_amount", help="How much the applicant wants to borrow.")
        with c4:
            intent = st.selectbox("Purpose", list(INTENT), format_func=INTENT.get, key="lg_in_intent",
                                  help="What the loan will be used for.")

    with st.container(border=True, key="card_credit"):
        ui.card_title(3, "Credit profile")
        c5, c6 = st.columns(2)
        with c5:
            history = st.slider("Credit history (years)", 0, 50, 5, key="lg_in_history",
                                help="How many years the applicant has had credit.")
        with c6:
            score = st.slider("Credit score", 300, 850, 680, key="lg_in_score",
                              help="Credit score on the 300-850 scale.")
        defaults = st.segmented_control("Previous loan defaults", ["No", "Yes"], default="No",
                                        key="lg_in_defaults", help="Has the applicant ever defaulted on a loan?") or "No"
    st.caption("Amounts use the training dataset's currency units, shown as $.")

ratio = amount / income  # derived feature, never entered manually
data = {
    "person_age": age, "person_income": income, "person_home_ownership": home,
    "loan_amnt": amount, "loan_intent": intent, "loan_percent_income": ratio,
    "cb_person_cred_hist_length": history, "credit_score": score,
    "previous_loan_defaults_on_file": defaults,
}

with right:
    warn = ('<div class="lg-metric-note" style="color:#B4474F">Loan exceeds annual income; '
            'the model supports 100% or less.</div>' if ratio > 1 else "")
    ui.card(f'<div class="lg-eyebrow">Application summary</div>'
            f'<div class="lg-metric-label">Loan-to-income ratio (calculated)</div>'
            f'<div class="lg-metric-value">{ratio:.2%}</div>'
            f'<div class="lg-metric-note">{ui.money(amount)} requested against {ui.money(income)} annual income.</div>{warn}')
    st.write("")
    if st.button("Evaluate application", type="primary", width="stretch"):
        errors = validate_application(data)
        if errors:
            st.session_state["lg_state"] = {"inputs": data, "errors": friendly_errors(errors)}
        else:
            try:
                with st.spinner("Scoring application..."):
                    st.session_state["lg_state"] = {"inputs": data, "result": predict_loan(data)}
            except Exception:
                log.exception("Prediction failed")
                st.session_state["lg_state"] = {"inputs": data, "failed": True}

    state = st.session_state.get("lg_state")
    if state is None:
        ui.card('<div class="lg-metric-label">No assessment yet</div>'
                '<div class="lg-metric-note">Fill in the details and select <strong>Evaluate application</strong> '
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
