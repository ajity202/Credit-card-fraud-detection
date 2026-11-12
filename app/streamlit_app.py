# ============================================================
# FraudGuard - Streamlit Application
# Direct model inference; no separate FastAPI server required.
# ============================================================

from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TARGET_COLUMN = "Class"
DEFAULT_FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

# Support both `streamlit run app/streamlit_app.py` and package-style imports.
for path in (str(PROJECT_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from backend import load_model_bundle
except ImportError:
    from app.backend import load_model_bundle

# Styling
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background-color: #eef3f9; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #172b4d; font-weight: 700; }
    div[data-testid="metric-container"] {
        background: #fff; border: 1px solid #dce3ec; border-radius: 12px;
        padding: 16px; box-shadow: 0 2px 8px rgba(15,23,42,.04);
    }
    .stButton > button {
        border-radius: 8px; border: 1px solid #d5dce5; background: #fff;
        color: #17365d; min-height: 42px; font-weight: 500;
    }
    .stButton > button:hover { border-color: #4a90e2; color: #1769aa; }
    .section-line { border-top: 1px solid #dfe5ec; margin: 28px 0; }
    .footer { text-align: center; color: #94a3b8; font-size: 12px; padding: 25px 0 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL + DATA LOADING
# ============================================================

@st.cache_resource
def get_model_bundle():
    """Load the serialized model bundle once per Streamlit process."""
    bundle = load_model_bundle()
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Model bundle must be a dictionary containing a 'model' key.")
    return bundle


@st.cache_data
def load_dataset():
    """Load the optional labelled reference dataset if it is deployed."""
    if not DATA_PATH.exists():
        return None
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        return None
    return data


def get_features(bundle):
    features = bundle.get("features") or DEFAULT_FEATURE_COLUMNS
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("The model bundle contains an invalid feature list.")
    return list(features)


def positive_class_probability(model, model_input):
    """Extract the probability belonging to fraud class 1."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The loaded model does not support predict_proba().")
    probabilities = np.asarray(model.predict_proba(model_input))
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("The model does not contain fraud class 1.")
    return probabilities[:, classes.index(1)]


def risk_details(probability, threshold):
    if probability >= threshold:
        return "High", "Review or block transaction"
    if probability >= max(0.30, threshold * 0.60):
        return "Medium", "Send transaction for additional review"
    return "Low", "Allow transaction, subject to normal controls"


def predict_dataframe(input_df, bundle):
    """Run batch predictions locally using the loaded model."""
    features = get_features(bundle)
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.70))
    missing = [column for column in features if column not in input_df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    model_input = input_df.loc[:, features].copy()
    for column in features:
        model_input[column] = pd.to_numeric(model_input[column], errors="coerce")
    if model_input.isna().any().any():
        count = int(model_input.isna().sum().sum())
        raise ValueError(f"Found {count:,} missing or non-numeric feature values.")
    if not np.isfinite(model_input.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite or invalid numeric values.")

    probabilities = positive_class_probability(model, model_input)
    predictions = (probabilities >= threshold).astype(int)
    results = input_df.copy()
    results["fraud_probability"] = probabilities
    results["prediction"] = predictions
    results["status"] = np.where(predictions == 1, "Potential Fraud", "Legitimate")
    details = [risk_details(float(p), threshold) for p in probabilities]
    results["risk_level"] = [item[0] for item in details]
    results["recommended_action"] = [item[1] for item in details]
    return results


def predict_single(features_dict, bundle):
    features = get_features(bundle)
    row = pd.DataFrame([[features_dict.get(f, 0.0) for f in features]], columns=features)
    result = predict_dataframe(row, bundle).iloc[0]
    return {
        "fraud_probability": float(result["fraud_probability"]),
        "prediction": int(result["prediction"]),
        "status": str(result["status"]),
        "risk_level": str(result["risk_level"]),
        "recommended_action": str(result["recommended_action"]),
    }


def feature_importance_frame(bundle):
    model = bundle["model"]
    features = get_features(bundle)
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    values = np.asarray(model.feature_importances_, dtype=float)
    if len(values) != len(features):
        return pd.DataFrame(columns=["Feature", "Importance"])
    return (
        pd.DataFrame({"Feature": features, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def metric_percent(metrics, key):
    value = metrics.get(key)
    if value is None:
        return "0.00%"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "0.00%"


def dataset_required_message():
    st.info(
        "No active dataset is available yet. Upload a transaction CSV in "
        "Batch Prediction and generate predictions to populate the Dashboard, "
        "Transactions, and Data Insights pages."
    )


# Load the model as essential; load the original dataset only if present.
try:
    MODEL_BUNDLE = get_model_bundle()
    MODEL = MODEL_BUNDLE["model"]
    FEATURE_COLUMNS = get_features(MODEL_BUNDLE)
    MODEL_NAME = MODEL_BUNDLE.get("model_name", "Random Forest")
    THRESHOLD = float(MODEL_BUNDLE.get("threshold", 0.70))
    VALIDATION_METRICS = MODEL_BUNDLE.get("validation_metrics", {}) or {}
    TEST_METRICS = MODEL_BUNDLE.get("test_metrics", {}) or {}
except Exception as exc:
    st.error("FraudGuard could not load its trained model.")
    st.exception(exc)
    st.stop()


# Session-wide dataset: batch uploads are shared by all pages in this browser session.
if "active_dataset" not in st.session_state:
    st.session_state["active_dataset"] = None
if "active_dataset_name" not in st.session_state:
    st.session_state["active_dataset_name"] = None
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None
if "batch_file_fingerprint" not in st.session_state:
    st.session_state["batch_file_fingerprint"] = None

try:
    reference_df = load_dataset()
except Exception as exc:
    reference_df = None
    st.warning(f"Optional reference dataset could not be loaded: {exc}")

active_df = st.session_state["active_dataset"]
if active_df is not None:
    df = active_df.copy()
    dataset_source = st.session_state.get("active_dataset_name") or "Uploaded dataset"
else:
    df = reference_df.copy() if reference_df is not None else None
    dataset_source = "Reference dataset"

dataset_available = df is not None and not df.empty
analysis_column = None
analysis_type = "Unlabelled transactions"

if dataset_available:
    if "prediction" in df.columns:
        analysis_column = "prediction"
        analysis_type = "Model predictions"
    elif TARGET_COLUMN in df.columns:
        analysis_column = TARGET_COLUMN
        analysis_type = "Actual labels"

if dataset_available:
    total_transactions = len(df)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    number_of_features = len(FEATURE_COLUMNS)
    if analysis_column:
        label_values = pd.to_numeric(df[analysis_column], errors="coerce")
        fraud_transactions = int((label_values == 1).sum())
        legitimate_transactions = int((label_values == 0).sum())
        fraud_rate = fraud_transactions / total_transactions * 100 if total_transactions else 0.0
    else:
        fraud_transactions = legitimate_transactions = 0
        fraud_rate = 0.0
else:
    total_transactions = fraud_transactions = legitimate_transactions = 0
    fraud_rate = 0.0
    missing_values = duplicate_rows = 0
    number_of_features = len(FEATURE_COLUMNS)


def status_label(value):
    try:
        return "Potential Fraud" if int(value) == 1 else "Legitimate"
    except (TypeError, ValueError):
        return "Unknown"


def get_label_column(data):
    if data is None:
        return None
    if "prediction" in data.columns:
        return "prediction"
    if TARGET_COLUMN in data.columns:
        return TARGET_COLUMN
    return None


def get_label_name(data):
    column = get_label_column(data)
    if column == "prediction":
        return "Predicted Status"
    if column == TARGET_COLUMN:
        return "Actual Status"
    return "Status not available"


def navigate_to(page_name):
    """Navigate to a page using the sidebar navigation state."""
    st.session_state["page_navigation"] = page_name


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:22px;font-weight:700;color:#17365d;margin-bottom:10px;">
            🛡️ FraudGuard
        </div>
        <div style="font-size:13px;color:#17365d;margin-bottom:25px;">
            Detect Today. Safer Tomorrow.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Dashboard", "Predict Transaction", "Batch Prediction",
            "Transactions", "Model Performance", "Data Insights", "Documentation",
        ],
        key="page_navigation",
    )
    st.divider()
    st.caption("Credit Card Fraud Detection")
    st.caption("Built with Python • Scikit-learn • Streamlit")
    st.divider()
    st.success("Model: Loaded")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Threshold: {THRESHOLD:.2f}")
    st.caption("Dataset: " + ("Loaded" if dataset_available else "Not loaded"))


# ============================================================
# DASHBOARD
# ============================================================


if page == "Dashboard":
    st.title("Welcome to FraudGuard")
    st.write("Monitor transaction data, detect potentially fraudulent activity, and explore machine learning insights.")
    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Transactions", f"{total_transactions:,}" if dataset_available else "0")
    c2.metric("Potential Fraud", f"{fraud_transactions:,}" if dataset_available and analysis_column else "0")
    c3.metric("Legitimate", f"{legitimate_transactions:,}" if dataset_available and analysis_column else "0")
    c4.metric("Model ROC-AUC", metric_percent(TEST_METRICS, "roc_auc"))

    if dataset_available:
        st.caption(f"Data source: {dataset_source} • Analysis basis: {analysis_type}")
        if not analysis_column:
            st.info("Dataset loaded. Generate batch predictions to calculate predicted fraud and legitimate counts.")
        else:
            st.caption(f"Flagged rate: {fraud_rate:.4f}% based on {analysis_type.lower()}.")
    else:
        dataset_required_message()

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Transaction Overview")
    if dataset_available and "Time" in df.columns:
        try:
            if analysis_column:
                trend = df[["Time", analysis_column]].copy()
                trend[analysis_column] = pd.to_numeric(trend[analysis_column], errors="coerce")
                trend["Time Period"] = pd.qcut(
                    trend["Time"], q=min(10, max(1, trend["Time"].nunique())),
                    labels=False, duplicates="drop",
                )
                summary = trend.groupby("Time Period", observed=True).agg(
                    Transactions=("Time", "size"),
                    Flagged=(analysis_column, "sum"),
                ).reset_index()
                summary["Not Flagged"] = summary["Transactions"] - summary["Flagged"]
                chart = summary[["Transactions", "Flagged", "Not Flagged"]].copy()
                chart.index = [f"Period {i + 1}" for i in range(len(chart))]
                st.line_chart(chart, width="stretch")
                st.caption(f"Flagged values use {analysis_type.lower()}.")
            else:
                st.line_chart(df[["Time"]].reset_index(drop=True), width="stretch")
                st.info("Generate predictions to show flagged versus non-flagged trends.")
        except Exception as exc:
            st.warning(f"Unable to display transaction trend: {exc}")
    elif dataset_available:
        st.info("The active dataset does not contain a Time column.")
    else:
        st.info("Upload a CSV in Batch Prediction to populate transaction trends.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Transaction Amount Overview")
    if dataset_available and "Amount" in df.columns:
        left, right = st.columns(2)
        with left:
            if analysis_column:
                labels = pd.to_numeric(df[analysis_column], errors="coerce")
                amounts = pd.to_numeric(df["Amount"], errors="coerce")
                amount_summary = pd.DataFrame({
                    "Status": ["Potential Fraud", "Legitimate"],
                    "Median Amount": [amounts[labels == 1].median(), amounts[labels == 0].median()],
                }).set_index("Status")
                st.bar_chart(amount_summary, width="stretch")
            else:
                st.bar_chart(df[["Amount"]].reset_index(drop=True), width="stretch")
        with right:
            recent_columns = [c for c in ["Time", "Amount", "fraud_probability", "prediction", TARGET_COLUMN] if c in df.columns]
            recent = df[recent_columns].tail(10).copy()
            if analysis_column:
                recent["Status"] = recent[analysis_column].map(status_label)
            st.dataframe(recent, width="stretch", hide_index=True)
            st.caption("Most recent rows in the active dataset.")
    elif dataset_available:
        st.info("The active dataset does not contain an Amount column.")
    else:
        st.info("Upload a CSV in Batch Prediction to populate amount summaries.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Quick Actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.button(
            "🔍 Predict a Transaction",
            use_container_width=True,
            on_click=navigate_to,
            args=("Predict Transaction",),
        )
    with a2:
        st.button(
            "📁 Batch Prediction",
            use_container_width=True,
            on_click=navigate_to,
            args=("Batch Prediction",),
        )
    with a3:
        st.button(
            "📊 Model Insights",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Performance",),
        )

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Dataset Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Dataset Shape", f"{df.shape[0]:,} × {df.shape[1]}" if dataset_available else "0")
    s2.metric("Missing Values", f"{missing_values:,}" if dataset_available else "0")
    s3.metric("Duplicate Rows", f"{duplicate_rows:,}" if dataset_available else "0")
    s4.metric("Features", f"{number_of_features:,}")
    if dataset_available:
        st.caption(f"Currently displaying: {dataset_source}")


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

elif page == "Predict Transaction":
    st.title("Predict a Transaction")
    st.write("Enter transaction features and FraudGuard will analyze them using the trained machine learning model.")
    st.warning(
        "V1–V28 are anonymized PCA-based features from the original dataset. "
        "Use values prepared in the same way as the model's training data."
    )

    with st.expander("› Load an example transaction"):
        if dataset_available:
            example_index = st.number_input(
                "Dataset row", min_value=0, max_value=max(len(df) - 1, 0),
                value=0, step=1, key="single_example_index",
            )
            if st.button("Load Example", key="load_example"):
                st.session_state["example_transaction"] = df.loc[example_index, FEATURE_COLUMNS].to_dict()
                st.success("Example transaction loaded.")
        else:
            st.info("Reference dataset is not deployed. Enter feature values manually.")

    example = st.session_state.get("example_transaction", {f: 0.0 for f in FEATURE_COLUMNS})
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Transaction Details")
    time_col, amount_col = st.columns(2)
    with time_col:
        time_value = st.number_input("Time (seconds)", value=float(example.get("Time", 0.0)), format="%.2f")
    with amount_col:
        amount_value = st.number_input(
            "Amount", min_value=0.0, value=max(0.0, float(example.get("Amount", 0.0))), format="%.2f"
        )

    st.subheader("PCA Features")
    pca_features = [f for f in FEATURE_COLUMNS if f.startswith("V")]
    feature_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(pca_features):
        with cols[i % 3]:
            feature_values[feature] = st.number_input(
                feature, value=float(example.get(feature, 0.0)),
                format="%.6f", key=f"predict_{feature}",
            )

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    if st.button("🔍 Predict Transaction", type="primary", use_container_width=True):
        try:
            payload = {"Time": time_value, **feature_values, "Amount": amount_value}
            with st.spinner("Analyzing transaction..."):
                st.session_state["last_single_prediction"] = predict_single(payload, MODEL_BUNDLE)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_single_prediction")
    if result:
        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        if result["prediction"] == 1:
            st.error("🚨 Potential Fraud Detected")
        else:
            st.success("✅ Transaction Classified as Legitimate")
        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
        r2.metric("Risk Level", result["risk_level"])
        r3.metric("Prediction", result["status"])
        st.info(f"Recommended action: **{result['recommended_action']}**")
        st.caption("This is a model-generated risk signal, not a definitive determination of fraud.")


# ============================================================
# BATCH PREDICTION
# ============================================================


elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.write("Upload a CSV containing multiple transactions and generate fraud predictions directly with the trained model.")
    st.info("Required columns: " + ", ".join(FEATURE_COLUMNS))
    st.caption("An optional Class column is retained for comparison but is not used as a model input.")

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"], key="batch_csv_uploader")
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            fingerprint = hashlib.md5(file_bytes).hexdigest()
            batch_df = pd.read_csv(uploaded_file)

            st.success(f"File uploaded: {uploaded_file.name}")
            missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
            if missing:
                st.error("The uploaded file is missing required columns: " + ", ".join(missing))
            elif batch_df.empty:
                st.warning("The uploaded CSV contains no transaction rows.")
            else:
                if fingerprint != st.session_state.get("batch_file_fingerprint"):
                    st.session_state["batch_file_fingerprint"] = fingerprint
                    st.session_state["batch_results"] = None
                    st.session_state["active_dataset"] = batch_df.copy()
                    st.session_state["active_dataset_name"] = uploaded_file.name

                st.write(f"Rows loaded: **{len(batch_df):,}**")
                st.dataframe(batch_df.head(10), width="stretch")
                if TARGET_COLUMN in batch_df.columns:
                    st.caption("The Class column is retained for reference and is not used as a model input.")

                if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Analyzing uploaded transactions..."):
                            results = predict_dataframe(batch_df, MODEL_BUNDLE)
                            st.session_state["batch_results"] = results.copy()
                            st.session_state["active_dataset"] = results.copy()
                            st.session_state["active_dataset_name"] = uploaded_file.name
                        st.success("Predictions generated. Dashboard, Transactions, and Data Insights now use this uploaded dataset.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Batch prediction failed: {exc}")
        except Exception as exc:
            st.error(f"Unable to read uploaded CSV: {exc}")

    results = st.session_state.get("batch_results")
    if results is not None and not results.empty:
        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        st.subheader("Prediction Summary")
        fraud_count = int((results["prediction"] == 1).sum())
        legitimate_count = int((results["prediction"] == 0).sum())
        total = len(results)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Processed", f"{total:,}")
        c2.metric("Potential Fraud", f"{fraud_count:,}")
        c3.metric("Legitimate", f"{legitimate_count:,}")
        c4.metric("Flagged Rate", f"{fraud_count / total:.2%}" if total else "0.00%")
        p1, p2 = st.columns(2)
        p1.metric("Average Fraud Probability", f"{results['fraud_probability'].mean():.2%}")
        p2.metric("Highest Fraud Probability", f"{results['fraud_probability'].max():.2%}")

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        st.subheader("Prediction Results")
        preferred = ["Time", "Amount", "fraud_probability", "prediction", "status",
                     "risk_level", "recommended_action", TARGET_COLUMN]
        display_columns = [c for c in preferred if c in results.columns]
        st.dataframe(results[display_columns], width="stretch", height=500)
        flagged = results[results["prediction"] == 1]
        if not flagged.empty:
            st.subheader("Flagged Transactions")
            st.warning(f"{len(flagged):,} transaction(s) were flagged as potentially fraudulent.")
            st.dataframe(flagged[display_columns], width="stretch")
        st.download_button(
            "⬇️ Download Prediction Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="fraudguard_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("This batch is now the active dataset for Dashboard, Transactions, and Data Insights in this session.")
    elif uploaded_file is None:
        st.info("Upload a transaction CSV to make it the active dataset for the rest of FraudGuard.")


# ============================================================
# TRANSACTIONS
# ============================================================


elif page == "Transactions":
    st.title("Transactions")
    st.write("Explore transactions from the active dataset.")
    if not dataset_available:
        dataset_required_message()
    else:
        label_column = get_label_column(df)
        st.caption(f"Data source: {dataset_source} • {get_label_name(df)}")
        c1, c2, c3 = st.columns(3)
        with c1:
            if label_column:
                transaction_type = st.selectbox("Transaction Type", ["All", "Potential Fraud", "Legitimate"])
            else:
                transaction_type = "All"
                st.selectbox("Transaction Type", ["All"], disabled=True)

        if "Amount" in df.columns:
            amount_values = pd.to_numeric(df["Amount"], errors="coerce")
            min_amount = float(amount_values.min())
            max_amount = float(amount_values.max())
            if not np.isfinite(min_amount) or not np.isfinite(max_amount):
                min_amount = max_amount = 0.0
            with c2:
                minimum_amount = st.number_input("Minimum Amount", min_value=min_amount, max_value=max_amount, value=min_amount, format="%.2f")
            with c3:
                maximum_amount = st.number_input("Maximum Amount", min_value=min_amount, max_value=max_amount, value=max_amount, format="%.2f")
        else:
            minimum_amount = maximum_amount = None
            st.info("Amount filters are unavailable because this dataset has no Amount column.")

        filtered = df.copy()
        if label_column and transaction_type == "Potential Fraud":
            filtered = filtered[pd.to_numeric(filtered[label_column], errors="coerce") == 1]
        elif label_column and transaction_type == "Legitimate":
            filtered = filtered[pd.to_numeric(filtered[label_column], errors="coerce") == 0]

        if minimum_amount is not None and maximum_amount is not None:
            if minimum_amount > maximum_amount:
                st.error("Minimum amount cannot be greater than maximum amount.")
                st.stop()
            values = pd.to_numeric(filtered["Amount"], errors="coerce")
            filtered = filtered[(values >= minimum_amount) & (values <= maximum_amount)]

        st.write(f"Showing **{len(filtered):,}** transactions")
        display = filtered.head(500).copy()
        if label_column:
            display["Display Status"] = display[label_column].map(status_label)
        st.dataframe(display, width="stretch", height=500)
        if len(filtered) > 500:
            st.info("Showing the first 500 matching transactions for better dashboard performance.")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":
    st.title("Model Performance")
    st.write(f"Performance of the {MODEL_NAME} fraud detection model.")
    st.info(f"Classification threshold: {THRESHOLD:.2f}")

    metric_items = [("Precision", "precision"), ("Recall", "recall"), ("F1 Score", "f1_score"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]
    metric_cols = st.columns(5)
    for col, (label, key) in zip(metric_cols, metric_items):
        col.metric(label, metric_percent(TEST_METRICS, key))
    st.caption("Metrics are read from the saved model bundle's test_metrics.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Test Set Confusion Matrix")
    candidates = [FIGURES_DIR / "confusion_matrix.png", FIGURES_DIR / "test_confusion_matrix.png", FIGURES_DIR / "confusion_matrix_test.png"]
    cm_path = next((p for p in candidates if p.exists()), None)
    tn, fp = TEST_METRICS.get("true_negatives"), TEST_METRICS.get("false_positives")
    fn, tp = TEST_METRICS.get("false_negatives"), TEST_METRICS.get("true_positives")
    if all(v is not None for v in [tn, fp, fn, tp]):
        cm_cols = st.columns(4)
        for col, label, value in zip(cm_cols, ["True Negatives", "False Positives", "False Negatives", "True Positives"], [tn, fp, fn, tp]):
            col.metric(label, f"{int(value):,}")
    if cm_path:
        st.image(str(cm_path), width="stretch")
    else:
        st.info("Confusion matrix image is not available in reports/figures.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader(f"{MODEL_NAME} Feature Importance")
    importance_candidates = [FIGURES_DIR / "feature_importance.png", FIGURES_DIR / "random_forest_feature_importance.png", FIGURES_DIR / "model_feature_importance.png"]
    importance_path = next((p for p in importance_candidates if p.exists()), None)
    if importance_path:
        st.image(str(importance_path), width="stretch")
    else:
        importance = feature_importance_frame(MODEL_BUNDLE)
        if not importance.empty:
            st.dataframe(importance, width="stretch", hide_index=True)
            st.bar_chart(importance.head(15).set_index("Feature"), width="stretch")
        else:
            st.info("Feature importance is not exposed by this model.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("SHAP Feature Importance")
    shap_candidates = [FIGURES_DIR / "shap_feature_importance.png", FIGURES_DIR / "shap_summary.png", FIGURES_DIR / "shap_summary_bar.png"]
    shap_path = next((p for p in shap_candidates if p.exists()), None)
    if shap_path:
        st.image(str(shap_path), width="stretch")
        st.caption("SHAP explains how input features contribute to model predictions.")
    else:
        st.info("A saved SHAP visualization was not found in reports/figures.")


# ============================================================
# DATA INSIGHTS
# ============================================================


elif page == "Data Insights":
    st.title("Data Insights")
    st.write("Explore transaction patterns and fraud characteristics.")
    if not dataset_available:
        dataset_required_message()
    else:
        label_column = get_label_column(df)
        st.caption(f"Data source: {dataset_source} • Analysis basis: {get_label_name(df)}")

        if label_column:
            labels = pd.to_numeric(df[label_column], errors="coerce")
            fraud_count = int((labels == 1).sum())
            legitimate_count = int((labels == 0).sum())
            st.subheader("Class Distribution")
            distribution = pd.DataFrame({
                "Status": ["Potential Fraud", "Legitimate"],
                "Transactions": [fraud_count, legitimate_count],
            }).set_index("Status")
            st.bar_chart(distribution, width="stretch")
            st.caption("Uploaded batch data uses model predictions; reference data uses actual Class labels.")
        else:
            st.info("Fraud/legitimate distribution will appear after batch predictions are generated.")

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        if "Amount" in df.columns:
            st.subheader("Transaction Amount Statistics")
            amounts = pd.to_numeric(df["Amount"], errors="coerce")
            if label_column:
                labels = pd.to_numeric(df[label_column], errors="coerce")
                legit_amounts = amounts[labels == 0]
                fraud_amounts = amounts[labels == 1]
                amount_stats = pd.DataFrame({
                    "Status": ["Legitimate", "Potential Fraud"],
                    "Count": [legit_amounts.count(), fraud_amounts.count()],
                    "Mean": [legit_amounts.mean(), fraud_amounts.mean()],
                    "Median": [legit_amounts.median(), fraud_amounts.median()],
                    "Minimum": [legit_amounts.min(), fraud_amounts.min()],
                    "Maximum": [legit_amounts.max(), fraud_amounts.max()],
                })
                st.dataframe(amount_stats.round(2), width="stretch", hide_index=True)
            else:
                st.dataframe(amounts.describe().to_frame("Amount").round(2), width="stretch")

            st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                st.subheader("Transaction Amount Distribution")
                amount_frame = pd.DataFrame({"Amount": amounts.dropna().reset_index(drop=True)})
                st.bar_chart(amount_frame, width="stretch")
            with right:
                st.subheader("Feature Correlation")
                try:
                    numeric = df.select_dtypes(include=np.number)
                    if label_column and label_column in numeric.columns:
                        corr = numeric.corr()[label_column].drop(label_column).dropna().sort_values()
                        if not corr.empty:
                            st.bar_chart(corr.to_frame(name="Correlation"), width="stretch")
                        else:
                            st.info("Not enough numeric variation to calculate correlations.")
                    else:
                        st.info("Generate predictions to calculate feature correlation with predicted fraud.")
                except Exception as exc:
                    st.warning(f"Unable to calculate correlations: {exc}")
        else:
            st.info("Amount-based insights are unavailable because the active dataset has no Amount column.")

        if "Time" in df.columns:
            st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
            st.subheader("Transaction Activity Over Time")
            time_data = pd.to_numeric(df["Time"], errors="coerce").dropna().sort_values().reset_index(drop=True)
            if not time_data.empty:
                st.line_chart(time_data, width="stretch")


# ============================================================
# DOCUMENTATION
# ============================================================

elif page == "Documentation":
    st.title("FraudGuard Documentation")
    st.write("FraudGuard is a machine learning application for identifying potentially fraudulent credit card transactions.")

    st.subheader("Project Workflow")
    workflow = [
        "1. Raw Dataset", "2. Data Validation", "3. Exploratory Data Analysis",
        "4. Preprocessing", "5. SMOTE for Class Imbalance", "6. Machine Learning",
        "7. Threshold Optimization", "8. Model Evaluation", "9. SHAP Explainability",
        "10. Model Serialization", "11. FraudGuard Streamlit Dashboard",
    ]
    for step in workflow:
        st.write(step)

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Current Model")
    st.write(f"**Model:** {MODEL_NAME}")
    st.write(f"**Classification Threshold:** {THRESHOLD:.2f}")
    st.write("**Purpose:** Identify transactions that may require additional fraud review.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Input Features")
    st.write(f"The model expects **{len(FEATURE_COLUMNS)} transaction features:**")
    fcols = st.columns(3)
    for i, feature in enumerate(FEATURE_COLUMNS):
        with fcols[i % 3]:
            st.write(f"• {feature}")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Dataset Information")
    if dataset_available:
        left, right = st.columns(2)
        with left:
            st.write(f"**Active source:** {dataset_source}")
            st.write(f"**Transactions:** {total_transactions:,}")
            if analysis_column:
                st.write(f"**Potential fraud / fraud labels:** {fraud_transactions:,}")
                st.write(f"**Legitimate labels:** {legitimate_transactions:,}")
                st.write(f"**Fraud / flagged rate:** {fraud_rate:.4f}%")
            else:
                st.write("**Fraud labels:** Not available")
        with right:
            st.write(f"**Model features:** {number_of_features}")
            st.write(f"**Missing values:** {missing_values:,}")
            st.write(f"**Duplicate rows:** {duplicate_rows:,}")
            st.write(f"**Dataset shape:** {df.shape[0]:,} × {df.shape[1]}")
        st.caption("For uploaded batch files, fraud counts are model predictions, not confirmed fraud labels.")
    else:
        st.info("No active dataset is loaded. Upload a CSV in Batch Prediction to populate the application.")


    st.subheader("Technology Stack")
    st.markdown("""
    - **Python**
    - **Pandas**
    - **NumPy**
    - **Scikit-learn**
    - **Imbalanced-learn**
    - **SHAP**
    - **Streamlit**
    - **Joblib**
    """)

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Application Architecture")
    st.code(
        """
User
  |
  v
FraudGuard Streamlit
  |
  v
Model Loader (app/backend.py)
  |
  v
Saved Random Forest Model
  |
  v
Fraud Probability
  |
  v
Threshold-based Risk Classification
        """,
        language="text",
    )

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Classification Threshold")
    st.write(
        f"The current fraud classification threshold is **{THRESHOLD:.2f}**. "
        "Transactions with a predicted fraud probability at or above this "
        "threshold are classified as potentially fraudulent."
    )
    st.caption("Predictions are decision-support signals and are not a definitive finding of fraud.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        FraudGuard • Credit Card Fraud Detection<br>
        Built with Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
