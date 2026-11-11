# ============================================================
# FraudGuard - Streamlit Application
# Credit Card Fraud Detection
# ============================================================

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "creditcard.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

API_URL = "http://127.0.0.1:8001"

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Main application
    ------------------------------------------------------- */

    .main {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* -------------------------------------------------------
       Sidebar
    ------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background-color: #eef3f9;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }


    /* -------------------------------------------------------
       Typography
    ------------------------------------------------------- */

    h1,
    h2,
    h3 {
        color: #172b4d;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       Metric cards
    ------------------------------------------------------- */

    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dce3ec;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }


    /* -------------------------------------------------------
       Buttons
    ------------------------------------------------------- */

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #d5dce5;
        background-color: #ffffff;
        color: #17365d;
        min-height: 42px;
        font-weight: 500;
    }

    .stButton > button:hover {
        border-color: #4a90e2;
        color: #1769aa;
    }


    /* -------------------------------------------------------
       Information boxes
    ------------------------------------------------------- */

    .info-box {
        background-color: #eaf3ff;
        border-radius: 8px;
        padding: 14px 16px;
        color: #1769aa;
        margin: 10px 0;
    }

    .success-box {
        background-color: #e9f8ef;
        border-radius: 8px;
        padding: 14px 16px;
        color: #18794e;
        margin: 10px 0;
    }

    .warning-box {
        background-color: #fff9df;
        border-radius: 8px;
        padding: 14px 16px;
        color: #946200;
        margin: 10px 0;
    }


    /* -------------------------------------------------------
       Section separator
    ------------------------------------------------------- */

    .section-line {
        border-top: 1px solid #dfe5ec;
        margin: 28px 0;
    }


    /* -------------------------------------------------------
       Result cards
    ------------------------------------------------------- */

    .fraud-result {
        background-color: #fff0f0;
        border: 1px solid #f0b5b5;
        border-radius: 12px;
        padding: 20px;
    }

    .legitimate-result {
        background-color: #eefaf3;
        border: 1px solid #b8dfc8;
        border-radius: 12px;
        padding: 20px;
    }


    /* -------------------------------------------------------
       Footer
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding-top: 25px;
        padding-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset():
    """Load the original credit card fraud dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# API FUNCTIONS
# ============================================================

@st.cache_data(ttl=60)
def get_model_info():
    """Retrieve model information from FastAPI."""

    response = requests.get(
        f"{API_URL}/model-info",
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def check_api():
    """Check whether FastAPI is available."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def predict_transaction(features):
    """Send one transaction to the FastAPI prediction endpoint."""

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=20,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Prediction API request failed: {exc}"
        ) from exc


def predict_batch_csv(uploaded_file):
    """
    Send an uploaded CSV directly to FastAPI.

    This is significantly faster than sending one API request
    per transaction.
    """

    try:
        file_bytes = uploaded_file.getvalue()

        files = {
            "file": (
                uploaded_file.name,
                io.BytesIO(file_bytes),
                "text/csv",
            )
        }

        response = requests.post(
            f"{API_URL}/predict-batch-csv",
            files=files,
            timeout=120,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Batch prediction API request failed: {exc}"
        ) from exc


def get_feature_importance():
    """Retrieve feature importance from FastAPI."""

    response = requests.get(
        f"{API_URL}/feature-importance",
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def get_shap_importance():
    """Retrieve SHAP importance from FastAPI."""

    response = requests.get(
        f"{API_URL}/shap-importance",
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# LOAD DATA + MODEL
# ============================================================

try:
    df = load_dataset()
    model_info = get_model_info()

except Exception as exc:

    st.error(
        "Unable to load project data or connect to the API."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = model_info.get(
    "model",
    "Random Forest",
)

THRESHOLD = float(
    model_info.get(
        "threshold",
        0.70,
    )
)

VALIDATION_METRICS = model_info.get(
    "validation_metrics",
    {},
)

TEST_METRICS = model_info.get(
    "test_metrics",
    {},
)


# ============================================================
# DATASET STATISTICS
# ============================================================

total_transactions = len(df)

fraud_transactions = int(
    df[TARGET_COLUMN].sum()
)

legitimate_transactions = (
    total_transactions - fraud_transactions
)

fraud_rate = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0
    else 0
)

missing_values = int(
    df.isna().sum().sum()
)

duplicate_rows = int(
    df.duplicated().sum()
)

number_of_features = (
    len(df.columns) - 1
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size: 22px;
            font-weight: 700;
            color: #17365d;
            margin-bottom: 10px;
        ">
            🛡️ FraudGuard
        </div>

        <div style="
            font-size: 13px;
            color: #17365d;
            margin-bottom: 25px;
        ">
            Detect Today. Safer Tomorrow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Predict Transaction",
            "Batch Prediction",
            "Transactions",
            "Model Performance",
            "Data Insights",
            "Documentation",
        ],
    )

    st.divider()

    st.caption("Credit Card Fraud Detection")
    st.caption(
        "Built with Python • ML • FastAPI • Streamlit"
    )

    st.divider()

    if check_api():
        st.success("API: Online")
    else:
        st.error("API: Offline")

    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Threshold: {THRESHOLD:.2f}")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("Welcome to FraudGuard")

    st.write(
        "Monitor transactions, detect fraud, and explore "
        "machine learning insights."
    )

    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            f"{total_transactions:,}",
        )

    with col2:
        st.metric(
            "Fraud Transactions",
            f"{fraud_transactions:,}",
        )

    with col3:
        st.metric(
            "Legitimate Transactions",
            f"{legitimate_transactions:,}",
        )

    with col4:

        roc_auc = float(
            TEST_METRICS.get(
                "roc_auc",
                0,
            )
        )

        st.metric(
            "Model ROC-AUC",
            f"{roc_auc:.2%}",
        )

    st.caption(
        f"Fraud rate: {fraud_rate:.4f}%"
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Transaction Overview
    # --------------------------------------------------------

    st.subheader("Transaction Overview")

    try:

        trend_df = df[
            ["Time", TARGET_COLUMN]
        ].copy()

        trend_df["Time Period"] = pd.qcut(
            trend_df["Time"],
            q=10,
            labels=False,
            duplicates="drop",
        )

        trend_summary = (
            trend_df
            .groupby(
                "Time Period",
                observed=True,
            )
            .agg(
                Transactions=(TARGET_COLUMN, "size"),
                Fraud=("Class", "sum"),
            )
            .reset_index()
        )

        trend_summary["Legitimate"] = (
            trend_summary["Transactions"]
            - trend_summary["Fraud"]
        )

        trend_chart = trend_summary[
            [
                "Transactions",
                "Fraud",
                "Legitimate",
            ]
        ]

        trend_chart.index = [
            f"Period {i + 1}"
            for i in range(
                len(trend_chart)
            )
        ]

        st.line_chart(
            trend_chart,
            width="stretch",
        )

        st.caption(
            "Transaction activity grouped into ten "
            "time-based periods."
        )

    except Exception as exc:

        st.warning(
            f"Unable to display transaction trend: {exc}"
        )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Amount Overview
    # --------------------------------------------------------

    st.subheader("Transaction Amount Overview")

    amount_col1, amount_col2 = st.columns(2)

    with amount_col1:

        amount_summary = pd.DataFrame(
            {
                "Class": [
                    "Fraud",
                    "Legitimate",
                ],
                "Median Amount": [
                    df.loc[
                        df[TARGET_COLUMN] == 1,
                        "Amount",
                    ].median(),
                    df.loc[
                        df[TARGET_COLUMN] == 0,
                        "Amount",
                    ].median(),
                ],
            }
        )

        amount_chart = (
            amount_summary
            .set_index("Class")
        )

        st.bar_chart(
            amount_chart,
            width="stretch",
        )

        st.caption(
            "Median transaction amount by class."
        )

    with amount_col2:

        recent_df = df[
            [
                "Time",
                "Amount",
                "Class",
            ]
        ].tail(10).copy()

        recent_df["Status"] = np.where(
            recent_df["Class"] == 1,
            "Fraud",
            "Legitimate",
        )

        st.dataframe(
            recent_df[
                [
                    "Time",
                    "Amount",
                    "Status",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Recent transactions from the dataset."
        )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Quick Actions
    # --------------------------------------------------------

    st.subheader("Quick Actions")

    action1, action2, action3 = st.columns(3)

    with action1:

        if st.button(
            "🔍 Predict a Transaction",
            use_container_width=True,
        ):
            st.info(
                "Select 'Predict Transaction' "
                "from the sidebar."
            )

    with action2:

        if st.button(
            "📁 Batch Prediction",
            use_container_width=True,
        ):
            st.info(
                "Select 'Batch Prediction' "
                "from the sidebar."
            )

    with action3:

        if st.button(
            "📊 Model Insights",
            use_container_width=True,
        ):
            st.info(
                "Select 'Model Performance' "
                "from the sidebar."
            )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------------

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Shape",
            f"{df.shape[0]:,} × {df.shape[1]}",
        )

    with col2:
        st.metric(
            "Missing Values",
            f"{missing_values:,}",
        )

    with col3:
        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}",
        )

    with col4:
        st.metric(
            "Features",
            f"{number_of_features}",
        )


# ============================================================
# PREDICT TRANSACTION
# ============================================================

elif page == "Predict Transaction":

    st.title("Predict a Transaction")

    st.write(
        "Enter transaction features and FraudGuard will "
        "send them to the trained machine learning model."
    )

    st.warning(
        "V1–V28 are anonymized PCA-based features from "
        "the original dataset."
    )

    # --------------------------------------------------------
    # Example transaction
    # --------------------------------------------------------

    with st.expander(
        "› Load an example transaction"
    ):

        example_index = st.number_input(
            "Dataset row",
            min_value=0,
            max_value=max(
                len(df) - 1,
                0,
            ),
            value=0,
            step=1,
        )

        if st.button(
            "Load Example",
            key="load_example",
        ):

            st.session_state[
                "example_transaction"
            ] = (
                df.loc[
                    example_index,
                    FEATURE_COLUMNS,
                ]
                .to_dict()
            )

            st.success(
                "Example transaction loaded."
            )

    if "example_transaction" in st.session_state:

        example_values = st.session_state[
            "example_transaction"
        ]

    else:

        example_values = {
            feature: 0.0
            for feature in FEATURE_COLUMNS
        }

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Transaction Details
    # --------------------------------------------------------

    st.subheader("Transaction Details")

    time_col, amount_col = st.columns(2)

    with time_col:

        time_value = st.number_input(
            "Time (seconds)",
            value=float(
                example_values.get(
                    "Time",
                    0,
                )
            ),
            format="%.2f",
        )

    with amount_col:

        amount_value = st.number_input(
            "Amount",
            min_value=0.0,
            value=float(
                example_values.get(
                    "Amount",
                    0,
                )
            ),
            format="%.2f",
        )

    # --------------------------------------------------------
    # PCA Features
    # --------------------------------------------------------

    st.subheader("PCA Features")

    feature_values = {}

    feature_names = [
        f"V{i}"
        for i in range(1, 29)
    ]

    for start in range(
        0,
        len(feature_names),
        3,
    ):

        cols = st.columns(3)

        for col, feature in zip(
            cols,
            feature_names[
                start:start + 3
            ],
        ):

            with col:

                feature_values[
                    feature
                ] = st.number_input(
                    feature,
                    value=float(
                        example_values.get(
                            feature,
                            0,
                        )
                    ),
                    format="%.6f",
                    key=f"predict_{feature}",
                )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Transaction",
        type="primary",
        use_container_width=True,
    ):

        payload_features = {
            "Time": time_value,
            **feature_values,
            "Amount": amount_value,
        }

        try:

            with st.spinner(
                "Analyzing transaction..."
            ):

                result = predict_transaction(
                    payload_features
                )

            probability = float(
                result.get(
                    "fraud_probability",
                    0,
                )
            )

            prediction = int(
                result.get(
                    "prediction",
                    0,
                )
            )

            status = result.get(
                "status",
                "Legitimate",
            )

            risk_level = result.get(
                "risk_level",
                "Low",
            )

            action = result.get(
                "recommended_action",
                "Allow transaction",
            )

            st.markdown(
                '<div class="section-line"></div>',
                unsafe_allow_html=True,
            )

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "🚨 Potential Fraud Detected"
                )

            else:

                st.success(
                    "✅ Transaction Classified as Legitimate"
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2%}",
                )

            with col2:

                st.metric(
                    "Risk Level",
                    risk_level,
                )

            with col3:

                st.metric(
                    "Prediction",
                    status,
                )

            st.info(
                f"Recommended action: **{action}**"
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.title("Batch Prediction")

    st.write(
        "Upload a CSV containing multiple transactions "
        "and generate fraud predictions."
    )

    st.info(
        "Required columns: Time, V1–V28, Amount"
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"File uploaded: "
                f"{uploaded_file.name}"
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            missing_columns = [
                column
                for column in FEATURE_COLUMNS
                if column not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded file is missing "
                    "required columns:"
                )

                st.write(
                    ", ".join(
                        missing_columns
                    )
                )

                st.stop()

            # ------------------------------------------------
            # Basic validation
            # ------------------------------------------------

            st.write(
                f"Rows loaded: "
                f"**{len(batch_df):,}**"
            )

            numeric_check = batch_df[
                FEATURE_COLUMNS
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            invalid_values = int(
                numeric_check.isna()
                .sum()
                .sum()
            )

            if invalid_values > 0:

                st.error(
                    f"The uploaded file contains "
                    f"{invalid_values:,} invalid or "
                    "missing feature values."
                )

                st.stop()

            st.dataframe(
                batch_df.head(10),
                width="stretch",
            )

            # ------------------------------------------------
            # Generate predictions
            # ------------------------------------------------

            if st.button(
                "🚀 Generate Predictions",
                type="primary",
                use_container_width=True,
            ):

                try:

                    with st.spinner(
                        "Sending transactions to "
                        "FraudGuard API..."
                    ):

                        api_result = (
                            predict_batch_csv(
                                uploaded_file
                            )
                        )

                    results = api_result.get(
                        "results",
                        [],
                    )

                    results_df = pd.DataFrame(
                        results
                    )

                    if results_df.empty:

                        st.warning(
                            "The API returned no prediction results."
                        )

                        st.stop()

                    # ------------------------------------------------
                    # Summary
                    # ------------------------------------------------

                    fraud_count = int(
                        api_result.get(
                            "fraud_transactions",
                            0,
                        )
                    )

                    legitimate_count = int(
                        api_result.get(
                            "legitimate_transactions",
                            0,
                        )
                    )

                    total_processed = int(
                        api_result.get(
                            "total_transactions",
                            len(results_df),
                        )
                    )

                    batch_fraud_rate = float(
                        api_result.get(
                            "fraud_rate",
                            0,
                        )
                    )

                    average_probability = float(
                        results_df[
                            "fraud_probability"
                        ]
                        .mean()
                    )

                    maximum_probability = float(
                        results_df[
                            "fraud_probability"
                        ]
                        .max()
                    )

                    st.markdown(
                        '<div class="section-line"></div>',
                        unsafe_allow_html=True,
                    )

                    st.subheader(
                        "Prediction Summary"
                    )

                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )

                    with col1:

                        st.metric(
                            "Total Processed",
                            f"{total_processed:,}",
                        )

                    with col2:

                        st.metric(
                            "Potential Fraud",
                            f"{fraud_count:,}",
                        )

                    with col3:

                        st.metric(
                            "Legitimate",
                            f"{legitimate_count:,}",
                        )

                    with col4:

                        st.metric(
                            "Fraud Rate",
                            f"{batch_fraud_rate:.2%}",
                        )

                    st.markdown(
                        '<div class="section-line"></div>',
                        unsafe_allow_html=True,
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Average Fraud Probability",
                            f"{average_probability:.2%}",
                        )

                    with col2:

                        st.metric(
                            "Highest Fraud Probability",
                            f"{maximum_probability:.2%}",
                        )

                    # ------------------------------------------------
                    # Results
                    # ------------------------------------------------

                    st.subheader(
                        "Prediction Results"
                    )

                    display_columns = [
                        column
                        for column in [
                            "Time",
                            "Amount",
                            "fraud_probability",
                            "prediction",
                            "status",
                            "risk_level",
                            "recommended_action",
                        ]
                        if column in results_df.columns
                    ]

                    st.dataframe(
                        results_df[
                            display_columns
                        ],
                        width="stretch",
                        height=500,
                    )

                    # ------------------------------------------------
                    # Fraud-only view
                    # ------------------------------------------------

                    fraud_results = (
                        results_df[
                            results_df["prediction"] == 1
                        ]
                        if "prediction"
                        in results_df.columns
                        else pd.DataFrame()
                    )

                    if not fraud_results.empty:

                        st.subheader(
                            "Flagged Transactions"
                        )

                        st.warning(
                            f"{len(fraud_results):,} "
                            "transaction(s) were flagged "
                            "as potentially fraudulent."
                        )

                        st.dataframe(
                            fraud_results[
                                display_columns
                            ],
                            width="stretch",
                        )

                    # ------------------------------------------------
                    # Download
                    # ------------------------------------------------

                    csv_data = (
                        results_df
                        .to_csv(
                            index=False
                        )
                        .encode("utf-8")
                    )

                    st.download_button(
                        "⬇️ Download Prediction Results",
                        data=csv_data,
                        file_name=(
                            "fraudguard_predictions.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as exc:

                    st.error(
                        f"Batch prediction failed: {exc}"
                    )

        except Exception as exc:

            st.error(
                f"Unable to process uploaded file: {exc}"
            )


# ============================================================
# TRANSACTIONS
# ============================================================

elif page == "Transactions":

    st.title("Transactions")

    st.write(
        "Explore transactions from the original dataset."
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "All",
                "Legitimate",
                "Fraud",
            ],
        )

    min_amount = float(
        df["Amount"].min()
    )

    max_amount = float(
        df["Amount"].max()
    )

    with col2:

        minimum_amount = st.number_input(
            "Minimum Amount",
            min_value=min_amount,
            max_value=max_amount,
            value=min_amount,
            format="%.2f",
        )

    with col3:

        maximum_amount = st.number_input(
            "Maximum Amount",
            min_value=min_amount,
            max_value=max_amount,
            value=max_amount,
            format="%.2f",
        )

    filtered_df = df.copy()

    # --------------------------------------------------------
    # Transaction type filter
    # --------------------------------------------------------

    if transaction_type == "Fraud":

        filtered_df = filtered_df[
            filtered_df[TARGET_COLUMN] == 1
        ]

    elif transaction_type == "Legitimate":

        filtered_df = filtered_df[
            filtered_df[TARGET_COLUMN] == 0
        ]

    # --------------------------------------------------------
    # Amount filter
    # --------------------------------------------------------

    if minimum_amount > maximum_amount:

        st.error(
            "Minimum amount cannot be greater than "
            "maximum amount."
        )

        st.stop()

    filtered_df = filtered_df[
        (
            filtered_df["Amount"]
            >= minimum_amount
        )
        &
        (
            filtered_df["Amount"]
            <= maximum_amount
        )
    ]

    st.write(
        f"Showing **{len(filtered_df):,}** transactions"
    )

    # --------------------------------------------------------
    # Display data
    # --------------------------------------------------------

    display_df = filtered_df.head(500)

    display_df = display_df.copy()

    if TARGET_COLUMN in display_df.columns:

        display_df["Status"] = np.where(
            display_df[TARGET_COLUMN] == 1,
            "Fraud",
            "Legitimate",
        )

    st.dataframe(
        display_df,
        width="stretch",
        height=500,
    )

    if len(filtered_df) > 500:

        st.info(
            "Showing the first 500 matching transactions "
            "for better dashboard performance."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        f"Performance of the {MODEL_NAME} "
        "fraud detection model."
    )

    st.info(
        f"Classification threshold: "
        f"{THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Precision",
            f"{float(TEST_METRICS.get('precision', 0)):.2%}",
        )

    with col2:

        st.metric(
            "Recall",
            f"{float(TEST_METRICS.get('recall', 0)):.2%}",
        )

    with col3:

        st.metric(
            "F1 Score",
            f"{float(TEST_METRICS.get('f1_score', 0)):.2%}",
        )

    with col4:

        st.metric(
            "ROC-AUC",
            f"{float(TEST_METRICS.get('roc_auc', 0)):.2%}",
        )

    with col5:

        st.metric(
            "PR-AUC",
            f"{float(TEST_METRICS.get('pr_auc', 0)):.2%}",
        )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader(
        "Test Set Confusion Matrix"
    )

    confusion_matrix_candidates = [
        FIGURES_DIR / "confusion_matrix.png",
        FIGURES_DIR / "test_confusion_matrix.png",
        FIGURES_DIR / "confusion_matrix_test.png",
    ]

    confusion_matrix_path = next(
        (
            path
            for path in confusion_matrix_candidates
            if path.exists()
        ),
        None,
    )

    tn = TEST_METRICS.get(
        "true_negatives"
    )

    fp = TEST_METRICS.get(
        "false_positives"
    )

    fn = TEST_METRICS.get(
        "false_negatives"
    )

    tp = TEST_METRICS.get(
        "true_positives"
    )

    if all(
        value is not None
        for value in [
            tn,
            fp,
            fn,
            tp,
        ]
    ):

        cm1, cm2, cm3, cm4 = st.columns(4)

        with cm1:

            st.metric(
                "True Negatives",
                f"{int(tn):,}",
            )

        with cm2:

            st.metric(
                "False Positives",
                f"{int(fp):,}",
            )

        with cm3:

            st.metric(
                "False Negatives",
                f"{int(fn):,}",
            )

        with cm4:

            st.metric(
                "True Positives",
                f"{int(tp):,}",
            )

    if confusion_matrix_path:

        st.image(
            str(confusion_matrix_path),
            width="stretch",
        )

    else:

        st.info(
            "Confusion matrix image is not available."
        )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    st.subheader(
        f"{MODEL_NAME} Feature Importance"
    )

    feature_importance_candidates = [
        FIGURES_DIR / "feature_importance.png",
        FIGURES_DIR / "random_forest_feature_importance.png",
        FIGURES_DIR / "model_feature_importance.png",
    ]

    feature_importance_path = next(
        (
            path
            for path in feature_importance_candidates
            if path.exists()
        ),
        None,
    )

    if feature_importance_path:

        st.image(
            str(feature_importance_path),
            width="stretch",
        )

    else:

        try:

            importance_result = (
                get_feature_importance()
            )

            importance_df = pd.DataFrame(
                importance_result.get(
                    "features",
                    [],
                )
            )

            if not importance_df.empty:

                st.dataframe(
                    importance_df,
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.info(
                    "Feature importance data is not available."
                )

        except Exception:

            st.info(
                "Feature importance is not available."
            )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.subheader(
        "SHAP Feature Importance"
    )

    shap_candidates = [
        FIGURES_DIR / "shap_feature_importance.png",
        FIGURES_DIR / "shap_summary.png",
        FIGURES_DIR / "shap_summary_bar.png",
    ]

    shap_path = next(
        (
            path
            for path in shap_candidates
            if path.exists()
        ),
        None,
    )

    if shap_path:

        st.image(
            str(shap_path),
            width="stretch",
        )

        st.caption(
            "SHAP provides model explainability by "
            "showing which features contribute most "
            "to model predictions."
        )

    else:

        try:

            shap_result = (
                get_shap_importance()
            )

            shap_df = pd.DataFrame(
                shap_result.get(
                    "features",
                    [],
                )
            )

            if not shap_df.empty:

                st.dataframe(
                    shap_df,
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.info(
                    "SHAP visualization is not available."
                )

        except Exception:

            st.info(
                "SHAP visualization is not available."
            )


# ============================================================
# DATA INSIGHTS
# ============================================================

elif page == "Data Insights":

    st.title("Data Insights")

    st.write(
        "Explore transaction patterns and fraud "
        "characteristics."
    )

    # --------------------------------------------------------
    # Class Distribution
    # --------------------------------------------------------

    st.subheader("Class Distribution")

    class_distribution = pd.DataFrame(
        {
            "Class": [
                "Fraud",
                "Legitimate",
            ],
            "Transactions": [
                fraud_transactions,
                legitimate_transactions,
            ],
        }
    )

    class_chart = (
        class_distribution
        .set_index("Class")
    )

    st.bar_chart(
        class_chart,
        width="stretch",
    )

    st.caption(
        "The dataset is highly imbalanced, with fraud "
        "representing a very small proportion of transactions."
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Transaction Amount Statistics
    # --------------------------------------------------------

    st.subheader(
        "Transaction Amount Statistics"
    )

    legitimate_amounts = df.loc[
        df[TARGET_COLUMN] == 0,
        "Amount",
    ]

    fraud_amounts = df.loc[
        df[TARGET_COLUMN] == 1,
        "Amount",
    ]

    amount_stats = pd.DataFrame(
        {
            "Class": [
                "Legitimate",
                "Fraud",
            ],
            "Count": [
                legitimate_amounts.count(),
                fraud_amounts.count(),
            ],
            "Mean": [
                legitimate_amounts.mean(),
                fraud_amounts.mean(),
            ],
            "Median": [
                legitimate_amounts.median(),
                fraud_amounts.median(),
            ],
            "Minimum": [
                legitimate_amounts.min(),
                fraud_amounts.min(),
            ],
            "Maximum": [
                legitimate_amounts.max(),
                fraud_amounts.max(),
            ],
        }
    )

    st.dataframe(
        amount_stats.round(2),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Fraud Amount Distribution
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Fraud Amount Distribution"
        )

        fraud_amount_plot = (
            df.loc[
                df[TARGET_COLUMN] == 1,
                ["Amount"],
            ]
            .reset_index(drop=True)
        )

        if not fraud_amount_plot.empty:

            fraud_amount_plot.index = [
                f"Fraud {i + 1}"
                for i in range(
                    len(fraud_amount_plot)
                )
            ]

            st.line_chart(
                fraud_amount_plot,
                width="stretch",
            )

        else:

            st.info(
                "No fraud transactions available."
            )

    # --------------------------------------------------------
    # Feature Correlation
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Feature Correlation with Fraud"
        )

        try:

            correlation = (
                df
                .corr(numeric_only=True)[
                    TARGET_COLUMN
                ]
                .drop(TARGET_COLUMN)
                .sort_values()
            )

            correlation_df = (
                correlation
                .to_frame(
                    name="Correlation"
                )
            )

            st.bar_chart(
                correlation_df,
                width="stretch",
            )

        except Exception as exc:

            st.warning(
                f"Unable to calculate correlations: {exc}"
            )


# ============================================================
# DOCUMENTATION
# ============================================================

elif page == "Documentation":

    st.title(
        "FraudGuard Documentation"
    )

    st.write(
        "FraudGuard is an end-to-end machine learning "
        "application for detecting potentially fraudulent "
        "credit card transactions."
    )

    # --------------------------------------------------------
    # Project Workflow
    # --------------------------------------------------------

    st.subheader(
        "Project Workflow"
    )

    workflow = [
        "1. Raw Dataset",
        "2. Data Validation",
        "3. Exploratory Data Analysis",
        "4. Preprocessing",
        "5. SMOTE for Class Imbalance",
        "6. Machine Learning",
        "7. Threshold Optimization",
        "8. Model Evaluation",
        "9. SHAP Explainability",
        "10. FastAPI Deployment",
        "11. FraudGuard Streamlit Dashboard",
    ]

    for step in workflow:

        st.write(step)

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Current Model
    # --------------------------------------------------------

    st.subheader(
        "Current Model"
    )

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        f"**Classification Threshold:** "
        f"{THRESHOLD:.2f}"
    )

    st.write(
        "**Purpose:** Identify transactions that may "
        "require fraud review."
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Input Features
    # --------------------------------------------------------

    st.subheader(
        "Input Features"
    )

    st.write(
        f"The model expects "
        f"**{len(FEATURE_COLUMNS)} transaction features:**"
    )

    feature_col1, feature_col2, feature_col3 = (
        st.columns(3)
    )

    for index, feature in enumerate(
        FEATURE_COLUMNS
    ):

        if index % 3 == 0:
            container = feature_col1

        elif index % 3 == 1:
            container = feature_col2

        else:
            container = feature_col3

        with container:

            st.write(
                f"• {feature}"
            )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Dataset Information
    # --------------------------------------------------------

    st.subheader(
        "Dataset Information"
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write(
            f"**Transactions:** "
            f"{total_transactions:,}"
        )

        st.write(
            f"**Fraud transactions:** "
            f"{fraud_transactions:,}"
        )

        st.write(
            f"**Legitimate transactions:** "
            f"{legitimate_transactions:,}"
        )

        st.write(
            f"**Fraud rate:** "
            f"{fraud_rate:.4f}%"
        )

    with info_col2:

        st.write(
            f"**Features:** "
            f"{number_of_features}"
        )

        st.write(
            f"**Missing values:** "
            f"{missing_values:,}"
        )

        st.write(
            f"**Duplicate rows:** "
            f"{duplicate_rows:,}"
        )

        st.write(
            f"**Dataset shape:** "
            f"{df.shape[0]:,} × {df.shape[1]}"
        )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Technology Stack
    # --------------------------------------------------------

    st.subheader(
        "Technology Stack"
    )

    st.markdown(
        """
        - **Python**
        - **Pandas**
        - **NumPy**
        - **Scikit-learn**
        - **Imbalanced-learn**
        - **SHAP**
        - **FastAPI**
        - **Streamlit**
        - **Joblib**
        """
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # API Architecture
    # --------------------------------------------------------

    st.subheader(
        "Application Architecture"
    )

    st.code(
        """
        User
          │
          ▼
        FraudGuard Streamlit
          │
          │ HTTP Request
          ▼
        FastAPI
          │
          ▼
        Random Forest Model
          │
          ▼
        Fraud Probability
          │
          ▼
        Risk Classification
        """,
        language="text",
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    st.subheader(
        "Classification Threshold"
    )

    st.write(
        f"The current fraud classification threshold "
        f"is **{THRESHOLD:.2f}**."
    )

    st.write(
        "Transactions with a predicted fraud probability "
        "at or above this threshold are classified as "
        "potentially fraudulent."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        FraudGuard • Credit Card Fraud Detection<br>
        Built with Python • Scikit-learn • FastAPI • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)