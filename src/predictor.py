import pandas as pd

from src.config import DECISION_THRESHOLD, FEATURES
from src.model_loader import load_model_artifacts


def predict_loan(application_data: dict) -> dict:
    """Predict loan approval using the trained preprocessing and model artifacts."""

    missing_features = set(FEATURES) - set(application_data)

    if missing_features:
        raise ValueError(
            f"Missing required features: {', '.join(sorted(missing_features))}"
        )

    input_data = pd.DataFrame(
        [{feature: application_data[feature] for feature in FEATURES}],
        columns=FEATURES,
    )

    preprocessor, model = load_model_artifacts()

    transformed_data = preprocessor.transform(input_data)

    approval_probability = float(
        model.predict_proba(transformed_data)[0][1]
    )

    prediction = int(approval_probability >= DECISION_THRESHOLD)

    return {
        "prediction": prediction,
        "decision": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": approval_probability,
        "rejection_probability": 1.0 - approval_probability,
        "threshold": DECISION_THRESHOLD,
    }
