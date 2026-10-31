"""Model Performance: real offline test-split metrics only (no invented curves)."""
import plotly.graph_objects as go
import streamlit as st

from src.config import DECISION_THRESHOLD, TEST_METRICS
from src.ui import theme as ui

# Test-split confusion matrix as published in the project README (threshold 0.47).
# It is reconciled against model_metadata.json below and hidden if it disagrees.
TN, FP, FN, TP = 6491, 509, 354, 1646


def matrix_matches_metrics() -> bool:
    total = TN + FP + FN + TP
    precision, recall = TP / (TP + FP), TP / (TP + FN)
    derived = {
        "accuracy": (TN + TP) / total, "precision": precision, "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall),
    }
    return all(abs(v - TEST_METRICS[k]) < 1e-4 for k, v in derived.items())


def confusion_chart() -> go.Figure:
    z = [[TN, FP], [FN, TP]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Rejected", "Approved"], y=["Rejected", "Approved"], text=z,
        texttemplate="%{text:,}", textfont=dict(size=20, color=ui.NAVY),
        colorscale=[[0, "#F8FAFC"], [1, "#5EEAD4"]], showscale=False,
        hovertemplate="Actual %{y}, predicted %{x}: %{z:,}<extra></extra>",
    ))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Predicted", side="bottom"), yaxis=dict(title="Actual", autorange="reversed"),
        font=dict(family="DM Sans, sans-serif", color=ui.MUTED),
    )
    return fig


ui.header("Model performance", "How the model performed",
          "Evaluation results from the offline test split used during model development.")
ui.render('<span class="lg-badge">Offline test-split evaluation - not live application metrics</span>')

METRICS = [
    ("Accuracy", "accuracy", "Overall share of correct predictions."),
    ("Precision", "precision", "Of predicted approvals, the share that were truly approved."),
    ("Recall", "recall", "Of truly approved applications, the share the model caught."),
    ("F1-Score", "f1_score", "Balance between precision and recall."),
    ("ROC-AUC", "roc_auc", "Ability to rank approved above rejected across all thresholds."),
    ("PR-AUC", "pr_auc", "Precision-recall trade-off; useful with imbalanced classes."),
]
for row in (METRICS[:3], METRICS[3:]):
    for col, (label, key, note) in zip(st.columns(3), row):
        with col:
            ui.metric_card(label, f"{TEST_METRICS[key]:.2%}", note)
    st.write("")

left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("Confusion matrix", f"Test split at the {DECISION_THRESHOLD:.2f} decision threshold.")
    if matrix_matches_metrics():
        st.plotly_chart(confusion_chart(), width="stretch", config={"displayModeBar": False})
        st.caption(f"{TN + FP + FN + TP:,} test applications. Counts reconcile with the reported metrics.")
    else:
        st.warning("Confusion matrix hidden: it does not reconcile with the stored metrics.")
with right:
    ui.section("Decision threshold")
    ui.card(f'<div class="lg-metric-value">{DECISION_THRESHOLD:.2f}</div>'
            '<div class="lg-metric-note">An application is predicted as approved when its estimated approval '
            'probability is at or above this value. A lower threshold approves more applications (more true '
            'approvals found, but more false approvals); a higher one is stricter.</div>')

st.write("")
ui.section("Model architecture")
st.markdown(f"""
| Component | Configuration |
|---|---|
| Algorithm | XGBoost classifier (binary) |
| Preprocessing | scikit-learn ColumnTransformer |
| Numeric features | Imputation + RobustScaler |
| Categorical features | Imputation + One-Hot Encoding |
| Input features | 9 (loan-to-income ratio is calculated automatically) |
| Decision threshold | {DECISION_THRESHOLD:.2f} |
""")
ui.card('<div class="lg-metric-label">Evaluation note</div><div class="lg-metric-note">The test split was also '
        'used in earlier baseline model comparisons. The final configuration and threshold were not tuned on '
        'the test results, but these figures are not a completely untouched, independent benchmark.</div>')
