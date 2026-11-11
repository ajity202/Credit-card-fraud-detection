from pathlib import Path
from typing import Dict, List, Any

import io
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "creditcard.csv"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

FEATURE_IMPORTANCE_PATH = (
    METRICS_DIR
    / "feature_importance.csv"
)

SHAP_IMPORTANCE_PATH = (
    METRICS_DIR
    / "shap_feature_importance.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )


MODEL_BUNDLE = joblib.load(MODEL_PATH)

MODEL = MODEL_BUNDLE["model"]

THRESHOLD = float(
    MODEL_BUNDLE["threshold"]
)

FEATURES = MODEL_BUNDLE["features"]

MODEL_NAME = MODEL_BUNDLE["model_name"]

VALIDATION_METRICS = MODEL_BUNDLE.get(
    "validation_metrics",
    {}
)

TEST_METRICS = MODEL_BUNDLE.get(
    "test_metrics",
    {}
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FraudGuard API",
    description=(
        "Production-style machine learning API for "
        "credit card fraud detection using Random Forest."
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class TransactionRequest(BaseModel):
    """
    Request schema for single transaction prediction.
    """

    features: Dict[str, float] = Field(
        ...,
        description=(
            "Transaction features containing "
            "Time, V1-V28 and Amount."
        )
    )


class BatchTransactionRequest(BaseModel):
    """
    Request schema for JSON batch prediction.
    """

    transactions: List[Dict[str, float]] = Field(
        ...,
        min_length=1,
        description="List of transaction feature dictionaries."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_features(
    features: Dict[str, float]
) -> None:
    """
    Validate required model features.
    """

    if not isinstance(features, dict):
        raise HTTPException(
            status_code=422,
            detail="Features must be provided as a dictionary."
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in features
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features.",
                "missing_features": missing_features,
            },
        )

    invalid_features = []

    for feature in FEATURES:

        value = features[feature]

        try:
            numeric_value = float(value)

        except (TypeError, ValueError):

            invalid_features.append(
                {
                    "feature": feature,
                    "value": value,
                    "reason": "Value must be numeric."
                }
            )

            continue

        if not np.isfinite(numeric_value):

            invalid_features.append(
                {
                    "feature": feature,
                    "value": value,
                    "reason": "Value must be finite."
                }
            )

    if invalid_features:

        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid feature values.",
                "invalid_features": invalid_features,
            },
        )


def prepare_transaction(
    features: Dict[str, float]
) -> pd.DataFrame:
    """
    Convert transaction dictionary into
    model-ready DataFrame.
    """

    validate_features(features)

    data = {
        feature: float(features[feature])
        for feature in FEATURES
    }

    transaction_df = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    return transaction_df


def get_risk_information(
    probability: float
) -> Dict[str, str]:
    """
    Convert fraud probability into
    risk level and recommended action.
    """

    if probability >= 0.90:

        return {
            "risk_level": "Critical",
            "recommended_action": "Block transaction"
        }

    if probability >= 0.70:

        return {
            "risk_level": "High",
            "recommended_action": "Review transaction"
        }

    if probability >= 0.40:

        return {
            "risk_level": "Medium",
            "recommended_action": "Investigate transaction"
        }

    if probability >= 0.20:

        return {
            "risk_level": "Low",
            "recommended_action": "Allow with monitoring"
        }

    return {
        "risk_level": "Low",
        "recommended_action": "Allow transaction"
    }


def generate_prediction(
    transaction_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate fraud prediction and risk information.
    """

    probability = float(
        MODEL.predict_proba(
            transaction_df
        )[0, 1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    if prediction == 1:

        status = "Fraud"

    else:

        status = "Legitimate"

    risk_information = get_risk_information(
        probability
    )

    return {
        "fraud_probability": round(
            probability,
            6
        ),
        "prediction": prediction,
        "status": status,
        "risk_level": risk_information[
            "risk_level"
        ],
        "recommended_action": risk_information[
            "recommended_action"
        ],
    }


def validate_dataframe(
    df: pd.DataFrame
) -> None:
    """
    Validate uploaded transaction DataFrame.
    """

    if df.empty:

        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV contains no transactions."
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV is missing required columns.",
                "missing_columns": missing_features,
            },
        )

    invalid_columns = []

    for feature in FEATURES:

        if not pd.api.types.is_numeric_dtype(
            df[feature]
        ):

            invalid_columns.append(feature)

    if invalid_columns:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Some required columns contain "
                    "non-numeric values."
                ),
                "invalid_columns": invalid_columns,
            },
        )

    numeric_data = df[FEATURES]

    if numeric_data.isnull().any().any():

        null_columns = (
            numeric_data.columns[
                numeric_data.isnull().any()
            ]
            .tolist()
        )

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Missing values found in "
                    "required feature columns."
                ),
                "columns": null_columns,
            },
        )

    if not np.isfinite(
        numeric_data.to_numpy()
    ).all():

        raise HTTPException(
            status_code=422,
            detail=(
                "Required feature columns contain "
                "infinite values."
            ),
        )


def generate_batch_predictions(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate predictions for a complete DataFrame.
    """

    validate_dataframe(df)

    prediction_features = df[
        FEATURES
    ].copy()

    probabilities = MODEL.predict_proba(
        prediction_features
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    result_df = df.copy()

    result_df[
        "fraud_probability"
    ] = probabilities.round(6)

    result_df[
        "prediction"
    ] = predictions

    result_df[
        "status"
    ] = np.where(
        predictions == 1,
        "Fraud",
        "Legitimate"
    )

    risk_levels = [
        get_risk_information(
            float(probability)
        )["risk_level"]
        for probability in probabilities
    ]

    recommended_actions = [
        get_risk_information(
            float(probability)
        )["recommended_action"]
        for probability in probabilities
    ]

    result_df[
        "risk_level"
    ] = risk_levels

    result_df[
        "recommended_action"
    ] = recommended_actions

    return result_df


def create_batch_summary(
    result_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Create summary information for batch predictions.
    """

    total_transactions = len(
        result_df
    )

    fraud_transactions = int(
        result_df["prediction"].sum()
    )

    legitimate_transactions = (
        total_transactions
        - fraud_transactions
    )

    fraud_rate = (
        fraud_transactions
        / total_transactions
        if total_transactions > 0
        else 0
    )

    average_probability = float(
        result_df[
            "fraud_probability"
        ].mean()
    )

    maximum_probability = float(
        result_df[
            "fraud_probability"
        ].max()
    )

    return {
        "total_transactions": total_transactions,
        "fraud_transactions": fraud_transactions,
        "legitimate_transactions": legitimate_transactions,
        "fraud_rate": round(
            fraud_rate,
            6
        ),
        "average_fraud_probability": round(
            average_probability,
            6
        ),
        "maximum_fraud_probability": round(
            maximum_probability,
            6
        ),
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API status endpoint.
    """

    return {
        "message": "FraudGuard API is running",
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "features": len(FEATURES),
        "version": "2.0.0",
        "status": "online",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    API health check.
    """

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "threshold": THRESHOLD,
        "feature_count": len(FEATURES),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():
    """
    Return model metadata and performance metrics.
    """

    return {
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "number_of_features": len(FEATURES),
        "features": FEATURES,
        "validation_metrics": VALIDATION_METRICS,
        "test_metrics": TEST_METRICS,
    }


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

@app.post("/predict")
def predict_transaction(
    request: TransactionRequest
):
    """
    Predict whether a single transaction is fraudulent.
    """

    transaction_df = prepare_transaction(
        request.features
    )

    result = generate_prediction(
        transaction_df
    )

    return result


# ============================================================
# BATCH PREDICTION - JSON
# ============================================================

@app.post("/predict-batch")
def predict_batch(
    request: BatchTransactionRequest
):
    """
    Predict multiple transactions supplied as JSON.
    """

    results = []

    for index, transaction in enumerate(
        request.transactions
    ):

        transaction_df = prepare_transaction(
            transaction
        )

        prediction = generate_prediction(
            transaction_df
        )

        prediction[
            "transaction_index"
        ] = index

        results.append(
            prediction
        )

    fraud_count = sum(
        result["prediction"] == 1
        for result in results
    )

    legitimate_count = (
        len(results)
        - fraud_count
    )

    return {
        "total_transactions": len(results),
        "fraud_transactions": fraud_count,
        "legitimate_transactions": legitimate_count,
        "fraud_rate": round(
            fraud_count / len(results),
            6
        ),
        "results": results,
    }


# ============================================================
# BATCH PREDICTION - CSV UPLOAD
# ============================================================

@app.post("/predict-batch-csv")
async def predict_batch_csv(
    file: UploadFile = File(...)
):
    """
    Predict transactions from an uploaded CSV file.
    """

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file name provided."
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty."
            )

        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to read CSV file: "
                f"{str(e)}"
            ),
        )

    result_df = generate_batch_predictions(
        df
    )

    summary = create_batch_summary(
        result_df
    )

    return {
        "filename": file.filename,
        **summary,
        "results": result_df.to_dict(
            orient="records"
        ),
    }


# ============================================================
# BATCH PREDICTION ALIAS
# ============================================================

@app.post("/batch-predict")
async def batch_predict_alias(
    file: UploadFile = File(...)
):
    """
    Alias for /predict-batch-csv.

    Useful for frontend applications that use
    the shorter /batch-predict endpoint.
    """

    return await predict_batch_csv(
        file
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@app.get("/feature-importance")
def feature_importance():
    """
    Return Random Forest feature importance.
    """

    if not FEATURE_IMPORTANCE_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Feature importance file "
                "is not available."
            )
        )

    try:

        df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

        return {
            "model": MODEL_NAME,
            "features": df.to_dict(
                orient="records"
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load feature importance: "
                f"{str(e)}"
            ),
        )


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

@app.get("/shap-importance")
def shap_importance():
    """
    Return global SHAP feature importance.
    """

    if not SHAP_IMPORTANCE_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "SHAP feature importance file "
                "is not available."
            )
        )

    try:

        df = pd.read_csv(
            SHAP_IMPORTANCE_PATH
        )

        return {
            "method": "SHAP",
            "features": df.to_dict(
                orient="records"
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load SHAP importance: "
                f"{str(e)}"
            ),
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

@app.get("/dataset-info")
def dataset_info():
    """
    Return basic dataset statistics.
    """

    if not DATA_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    try:

        df = pd.read_csv(
            DATA_PATH
        )

        if "Class" not in df.columns:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Target column 'Class' "
                    "was not found."
                )
            )

        total = len(df)

        fraud = int(
            df["Class"].sum()
        )

        legitimate = (
            total - fraud
        )

        missing_values = int(
            df.isnull().sum().sum()
        )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        fraud_rate = (
            fraud / total
            if total > 0
            else 0
        )

        return {
            "rows": total,
            "columns": len(df.columns),
            "features": len(FEATURES),
            "fraud_transactions": fraud,
            "legitimate_transactions": legitimate,
            "fraud_rate": round(
                fraud_rate,
                6
            ),
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read dataset: "
                f"{str(e)}"
            ),
        )


# ============================================================
# MODEL METRICS
# ============================================================

@app.get("/metrics")
def model_metrics():
    """
    Return validation and test metrics.
    """

    return {
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "validation": VALIDATION_METRICS,
        "test": TEST_METRICS,
    }


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
def startup_event():

    print("=" * 60)
    print("FraudGuard API")
    print("=" * 60)
    print(f"Model       : {MODEL_NAME}")
    print(f"Threshold   : {THRESHOLD}")
    print(f"Features    : {len(FEATURES)}")
    print(f"Model path  : {MODEL_PATH}")
    print("=" * 60)