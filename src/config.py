from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PREPROCESSOR_PATH = MODEL_DIR / "loanguard_preprocessor.joblib"
MODEL_PATH = MODEL_DIR / "loanguard_xgb_model.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DATASET_PATH = RAW_DATA_DIR / "loan_data.csv"

with METADATA_PATH.open("r", encoding="utf-8") as file:
    MODEL_METADATA = json.load(file)

PROJECT_NAME = MODEL_METADATA["project"]
MODEL_TYPE = MODEL_METADATA["model_type"]
DECISION_THRESHOLD = float(MODEL_METADATA["threshold"])

FEATURES = MODEL_METADATA["features"]
MODEL_VERSIONS = MODEL_METADATA["versions"]
TEST_METRICS = MODEL_METADATA["test_metrics"]

if not 0.0 <= DECISION_THRESHOLD <= 1.0:
    raise ValueError("Model threshold must be between 0 and 1.")

for artifact in (PREPROCESSOR_PATH, MODEL_PATH, METADATA_PATH):
    if not artifact.is_file():
        raise FileNotFoundError(f"Required model artifact not found: {artifact}")
