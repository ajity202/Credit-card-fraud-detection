from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR


MODEL_PATH = MODEL_DIR / "fraud_model.joblib"


def load_model_bundle(model_path=MODEL_PATH):
    """
    Load the trained fraud detection model bundle.
    """

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    return joblib.load(model_path)


def validate_features(transaction, expected_features):
    """
    Validate that the transaction contains
    exactly the features expected by the model.
    """

    missing_features = [
        feature
        for feature in expected_features
        if feature not in transaction.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    return transaction[expected_features]


def predict_transaction(transaction, model_bundle=None):
    """
    Predict whether a transaction is fraudulent.

    Parameters
    ----------
    transaction : pandas.DataFrame
        A DataFrame containing one or more transactions.

    model_bundle : dict, optional
        Previously loaded model bundle.

    Returns
    -------
    pandas.DataFrame
        Prediction results.
    """

    if model_bundle is None:
        model_bundle = load_model_bundle()

    model = model_bundle["model"]
    threshold = model_bundle["threshold"]
    expected_features = model_bundle["features"]

    transaction = validate_features(
        transaction,
        expected_features
    )

    probabilities = model.predict_proba(
        transaction
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = pd.DataFrame({
        "fraud_probability": probabilities,
        "prediction": predictions
    })

    results["status"] = results["prediction"].map({
        0: "Legitimate",
        1: "Fraud"
    })

    results["risk_level"] = results["fraud_probability"].apply(
        get_risk_level
    )

    results["recommended_action"] = results["risk_level"].apply(
        get_recommended_action
    )

    return results


def get_risk_level(probability):
    """
    Convert fraud probability into a human-readable risk level.
    """

    if probability >= 0.70:
        return "High"

    if probability >= 0.30:
        return "Medium"

    return "Low"


def get_recommended_action(risk_level):
    """
    Return an action recommendation based on risk level.
    """

    actions = {
        "High": "Block or manually review",
        "Medium": "Review transaction",
        "Low": "Allow transaction"
    }

    return actions[risk_level]