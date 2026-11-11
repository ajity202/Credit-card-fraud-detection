
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"


def load_model_bundle():
    """Load the trained model and metadata."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "threshold",
        "features",
        "model_name",
    }

    missing = required_keys - set(bundle.keys())

    if missing:
        raise ValueError(
            f"Model bundle is missing keys: {missing}"
        )

    return bundle


def validate_features(data, features):
    """Validate transaction feature columns."""

    missing = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    values = data[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if values.isnull().any().any():
        raise ValueError(
            "Features contain missing or non-numeric values."
        )

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            "Features contain infinite values."
        )

    return values


def predict_batch(data, bundle):
    """Generate predictions for multiple transactions."""

    features = bundle["features"]
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    input_data = validate_features(data, features)

    probabilities = model.predict_proba(
        input_data
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = data.copy()

    results["fraud_probability"] = probabilities
    results["prediction"] = predictions

    results["status"] = np.where(
        predictions == 1,
        "Fraud",
        "Legitimate"
    )

    results["risk_level"] = np.select(
        [
            probabilities >= 0.90,
            probabilities >= 0.70,
            probabilities >= 0.40,
        ],
        [
            "Critical",
            "High",
            "Medium",
        ],
        default="Low",
    )

    results["recommended_action"] = np.where(
        predictions == 1,
        "Review transaction",
        "Allow transaction"
    )

    return results


def predict_single(features_dict, bundle):
    """Generate prediction for one transaction."""

    data = pd.DataFrame([features_dict])

    results = predict_batch(data, bundle)

    return results.iloc[0].to_dict()
