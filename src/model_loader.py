import joblib
import streamlit as st
import xgboost as xgb

from src.config import MODEL_PATH, PREPROCESSOR_PATH


@st.cache_resource(show_spinner="Loading LoanGuard AI model...")
def load_model_artifacts():
    """Load and cache the preprocessing pipeline and trained XGBoost model."""

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    return preprocessor, model
