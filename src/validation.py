from src.config import FEATURES


ALLOWED_VALUES = {
    "person_home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_intent": {
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    },
    "previous_loan_defaults_on_file": {"Yes", "No"},
}

NUMERIC_RANGES = {
    "person_age": (18, 100),
    "person_income": (0, 10_000_000),
    "loan_amnt": (0, 10_000_000),
    "loan_percent_income": (0, 1),
    "cb_person_cred_hist_length": (0, 100),
    "credit_score": (300, 850),
}


def validate_application(application_data: dict) -> list[str]:
    """Validate loan application fields and return a list of errors."""

    errors = []

    missing_features = set(FEATURES) - set(application_data)

    if missing_features:
        errors.append(
            f"Missing required fields: {', '.join(sorted(missing_features))}"
        )
        return errors

    for feature, (minimum, maximum) in NUMERIC_RANGES.items():
        value = application_data[feature]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{feature} must be a numeric value.")
            continue

        if not minimum <= value <= maximum:
            errors.append(
                f"{feature} must be between {minimum} and {maximum}."
            )

    for feature, allowed_values in ALLOWED_VALUES.items():
        value = application_data[feature]

        if not isinstance(value, str) or value not in allowed_values:
            errors.append(
                f"{feature} must be one of: {', '.join(sorted(allowed_values))}."
            )

    return errors
