import pytest

from src.config import FEATURES, DECISION_THRESHOLD
from src.model_loader import load_model_artifacts
from src.predictor import predict_loan
from src.validation import validate_application


@pytest.fixture
def valid_application():
    return {
        "person_age": 22,
        "person_income": 71948,
        "person_home_ownership": "RENT",
        "loan_amnt": 35000,
        "loan_intent": "PERSONAL",
        "loan_percent_income": 0.49,
        "cb_person_cred_hist_length": 3,
        "credit_score": 561,
        "previous_loan_defaults_on_file": "No",
    }


def test_model_artifacts_load_successfully():
    preprocessor, model = load_model_artifacts()

    assert preprocessor is not None
    assert model is not None
    assert len(preprocessor.feature_names_in_) == len(FEATURES)
    assert model.n_features_in_ > 0


def test_valid_application_passes_validation(valid_application):
    assert validate_application(valid_application) == []


def test_invalid_application_fails_validation(valid_application):
    valid_application["person_age"] = 15

    errors = validate_application(valid_application)

    assert len(errors) > 0
    assert any("person_age" in error for error in errors)


def test_missing_features_are_detected(valid_application):
    del valid_application["credit_score"]

    errors = validate_application(valid_application)

    assert any("Missing required fields" in error for error in errors)


def test_prediction_returns_valid_result(valid_application):
    result = predict_loan(valid_application)

    assert result["decision"] in {"Approved", "Rejected"}
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["approval_probability"] <= 1.0
    assert 0.0 <= result["rejection_probability"] <= 1.0
    assert result["threshold"] == DECISION_THRESHOLD


def test_prediction_probabilities_sum_to_one(valid_application):
    result = predict_loan(valid_application)

    total = (
        result["approval_probability"]
        + result["rejection_probability"]
    )

    assert total == pytest.approx(1.0)
