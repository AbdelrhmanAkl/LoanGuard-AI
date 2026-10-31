"""LoanGuard AI - entry point: page config, navigation, theme and model status."""
import streamlit as st

# Must remain the first Streamlit command.
st.set_page_config(
    page_title="LoanGuard AI | Smarter Risk Assessment",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import DECISION_THRESHOLD, MODEL_TYPE
from src.model_loader import load_model_artifacts
from src.ui import theme as ui

# Keep assessment inputs alive while the user visits other pages.
for _key in [k for k in st.session_state if k.startswith("lg_in_")]:
    st.session_state[_key] = st.session_state[_key]

PAGES = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/assessment.py", title="Loan Assessment", icon=":material/fact_check:"),
    st.Page("views/performance.py", title="Model Performance", icon=":material/monitoring:"),
    st.Page("views/about.py", title="About the Model", icon=":material/info:"),
]

navigation = st.navigation(PAGES, position="hidden")
ui.inject_css()

with st.sidebar:
    ui.render('<div class="lg-brand">LoanGuard<span> AI</span></div>'
              '<div class="lg-brand-sub">Smarter Risk Assessment</div>')
    for page in PAGES:
        st.page_link(page)
    st.divider()
    try:
        load_model_artifacts()  # cached; cheap after the first call
        model_ready = True
    except Exception:
        model_ready = False
    ui.model_status(model_ready, MODEL_TYPE, DECISION_THRESHOLD)

navigation.run()
