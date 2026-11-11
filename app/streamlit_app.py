# ============================================================
# FraudGuard - Streamlit Application
# Direct model inference; no separate FastAPI server required.
# ============================================================

from pathlib import Path
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
        return "N/A"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def dataset_required_message():
    st.info(
        "The original labelled dataset is not included in this deployment. "
        "Single and batch predictions remain available. To enable dataset-based "
        "charts and transaction exploration, deploy data/raw/creditcard.csv."
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

try:
    df = load_dataset()
except Exception as exc:
    df = None
    st.warning(f"Optional reference dataset could not be loaded: {exc}")

dataset_available = df is not None
if dataset_available:
    total_transactions = len(df)
    fraud_transactions = int(df[TARGET_COLUMN].sum())
    legitimate_transactions = total_transactions - fraud_transactions
    fraud_rate = fraud_transactions / total_transactions * 100 if total_transactions else 0
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    number_of_features = len([c for c in df.columns if c != TARGET_COLUMN])
else:
    total_transactions = fraud_transactions = legitimate_transactions = 0
    fraud_rate = 0.0
    missing_values = duplicate_rows = 0
    number_of_features = len(FEATURE_COLUMNS)


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
    )
    st.divider()
    st.caption("Credit Card Fraud Detection")
    st.caption("Built with Python • Scikit-learn • Streamlit")
    st.divider()
    st.success("Model: Loaded")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Threshold: {THRESHOLD:.2f}")
    st.caption("Reference dataset: " + ("Available" if dataset_available else "Not deployed"))


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.title("Welcome to FraudGuard")
    st.write("Monitor transaction data, detect potentially fraudulent activity, and explore machine learning insights.")
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reference Transactions", f"{total_transactions:,}" if dataset_available else "N/A")
    c2.metric("Fraud Transactions", f"{fraud_transactions:,}" if dataset_available else "N/A")
    c3.metric("Legitimate Transactions", f"{legitimate_transactions:,}" if dataset_available else "N/A")
    c4.metric("Model ROC-AUC", metric_percent(TEST_METRICS, "roc_auc"))
    if dataset_available:
        st.caption(f"Reference dataset fraud rate: {fraud_rate:.4f}%")
    else:
        dataset_required_message()

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Transaction Overview")
    if dataset_available:
        try:
            trend = df[["Time", TARGET_COLUMN]].copy()
            trend["Time Period"] = pd.qcut(
                trend["Time"], q=min(10, max(1, trend["Time"].nunique())),
                labels=False, duplicates="drop",
            )
            summary = trend.groupby("Time Period", observed=True).agg(
                Transactions=(TARGET_COLUMN, "size"), Fraud=(TARGET_COLUMN, "sum")
            ).reset_index()
            summary["Legitimate"] = summary["Transactions"] - summary["Fraud"]
            chart = summary[["Transactions", "Fraud", "Legitimate"]].copy()
            chart.index = [f"Period {i + 1}" for i in range(len(chart))]
            st.line_chart(chart, width="stretch")
            st.caption("Transaction activity grouped into time-based periods.")
        except Exception as exc:
            st.warning(f"Unable to display transaction trend: {exc}")
    else:
        st.info("Transaction trend will appear when the reference dataset is deployed.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Transaction Amount Overview")
    if dataset_available:
        left, right = st.columns(2)
        with left:
            amount_summary = pd.DataFrame({
                "Class": ["Fraud", "Legitimate"],
                "Median Amount": [
                    df.loc[df[TARGET_COLUMN] == 1, "Amount"].median(),
                    df.loc[df[TARGET_COLUMN] == 0, "Amount"].median(),
                ],
            }).set_index("Class")
            st.bar_chart(amount_summary, width="stretch")
            st.caption("Median transaction amount by class.")
        with right:
            recent = df[["Time", "Amount", TARGET_COLUMN]].tail(10).copy()
            recent["Status"] = np.where(recent[TARGET_COLUMN] == 1, "Fraud", "Legitimate")
            st.dataframe(recent[["Time", "Amount", "Status"]], width="stretch", hide_index=True)
            st.caption("Recent transactions from the reference dataset.")
    else:
        st.info("Amount summaries will appear when the reference dataset is deployed.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Quick Actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("🔍 Predict a Transaction", use_container_width=True):
            st.info("Select 'Predict Transaction' from the sidebar.")
    with a2:
        if st.button("📁 Batch Prediction", use_container_width=True):
            st.info("Select 'Batch Prediction' from the sidebar.")
    with a3:
        if st.button("📊 Model Insights", use_container_width=True):
            st.info("Select 'Model Performance' from the sidebar.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader("Dataset Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Dataset Shape", f"{df.shape[0]:,} × {df.shape[1]}" if dataset_available else "N/A")
    s2.metric("Missing Values", f"{missing_values:,}" if dataset_available else "N/A")
    s3.metric("Duplicate Rows", f"{duplicate_rows:,}" if dataset_available else "N/A")
    s4.metric("Features", f"{number_of_features:,}")


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
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded: {uploaded_file.name}")
            missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
            if missing:
                st.error("The uploaded file is missing required columns: " + ", ".join(missing))
            elif batch_df.empty:
                st.warning("The uploaded CSV contains no transaction rows.")
            else:
                st.write(f"Rows loaded: **{len(batch_df):,}**")
                st.dataframe(batch_df.head(10), width="stretch")
                if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Analyzing uploaded transactions..."):
                            st.session_state["batch_results"] = predict_dataframe(batch_df, MODEL_BUNDLE)
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
        c4.metric("Flagged Rate", f"{fraud_count / total:.2%}")
        p1, p2 = st.columns(2)
        p1.metric("Average Fraud Probability", f"{results['fraud_probability'].mean():.2%}")
        p2.metric("Highest Fraud Probability", f"{results['fraud_probability'].max():.2%}")

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        st.subheader("Prediction Results")
        preferred = [
            "Time", "Amount", "fraud_probability", "prediction", "status",
            "risk_level", "recommended_action", TARGET_COLUMN,
        ]
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


# ============================================================
# TRANSACTIONS
# ============================================================

elif page == "Transactions":
    st.title("Transactions")
    st.write("Explore transactions from the original labelled dataset.")
    if not dataset_available:
        dataset_required_message()
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            transaction_type = st.selectbox("Transaction Type", ["All", "Legitimate", "Fraud"])
        min_amount, max_amount = float(df["Amount"].min()), float(df["Amount"].max())
        with c2:
            minimum_amount = st.number_input("Minimum Amount", min_value=min_amount, max_value=max_amount, value=min_amount, format="%.2f")
        with c3:
            maximum_amount = st.number_input("Maximum Amount", min_value=min_amount, max_value=max_amount, value=max_amount, format="%.2f")
        if minimum_amount > maximum_amount:
            st.error("Minimum amount cannot be greater than maximum amount.")
            st.stop()
        filtered = df.copy()
        if transaction_type == "Fraud":
            filtered = filtered[filtered[TARGET_COLUMN] == 1]
        elif transaction_type == "Legitimate":
            filtered = filtered[filtered[TARGET_COLUMN] == 0]
        filtered = filtered[(filtered["Amount"] >= minimum_amount) & (filtered["Amount"] <= maximum_amount)]
        st.write(f"Showing **{len(filtered):,}** transactions")
        display = filtered.head(500).copy()
        display["Status"] = np.where(display[TARGET_COLUMN] == 1, "Fraud", "Legitimate")
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
        st.subheader("Class Distribution")
        distribution = pd.DataFrame({
            "Class": ["Fraud", "Legitimate"],
            "Transactions": [fraud_transactions, legitimate_transactions],
        }).set_index("Class")
        st.bar_chart(distribution, width="stretch")
        st.caption("The dataset is highly imbalanced when fraud represents a small proportion of transactions.")

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        st.subheader("Transaction Amount Statistics")
        legit_amounts = df.loc[df[TARGET_COLUMN] == 0, "Amount"]
        fraud_amounts = df.loc[df[TARGET_COLUMN] == 1, "Amount"]
        amount_stats = pd.DataFrame({
            "Class": ["Legitimate", "Fraud"],
            "Count": [legit_amounts.count(), fraud_amounts.count()],
            "Mean": [legit_amounts.mean(), fraud_amounts.mean()],
            "Median": [legit_amounts.median(), fraud_amounts.median()],
            "Minimum": [legit_amounts.min(), fraud_amounts.min()],
            "Maximum": [legit_amounts.max(), fraud_amounts.max()],
        })
        st.dataframe(amount_stats.round(2), width="stretch", hide_index=True)

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Fraud Amount Distribution")
            fraud_plot = df.loc[df[TARGET_COLUMN] == 1, ["Amount"]].reset_index(drop=True)
            if not fraud_plot.empty:
                fraud_plot.index = [f"Fraud {i + 1}" for i in range(len(fraud_plot))]
                st.line_chart(fraud_plot, width="stretch")
            else:
                st.info("No fraud transactions available.")
        with right:
            st.subheader("Feature Correlation with Fraud")
            try:
                corr = df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()
                st.bar_chart(corr.to_frame(name="Correlation"), width="stretch")
            except Exception as exc:
                st.warning(f"Unable to calculate correlations: {exc}")


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
            st.write(f"**Transactions:** {total_transactions:,}")
            st.write(f"**Fraud transactions:** {fraud_transactions:,}")
            st.write(f"**Legitimate transactions:** {legitimate_transactions:,}")
            st.write(f"**Fraud rate:** {fraud_rate:.4f}%")
        with right:
            st.write(f"**Features:** {number_of_features}")
            st.write(f"**Missing values:** {missing_values:,}")
            st.write(f"**Duplicate rows:** {duplicate_rows:,}")
            st.write(f"**Dataset shape:** {df.shape[0]:,} × {df.shape[1]}")
    else:
        st.info("The reference dataset is not deployed. The trained model remains available for single and batch predictions.")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
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
