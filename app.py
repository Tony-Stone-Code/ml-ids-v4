"""
Streamlit Web Application for UNSW-NB15 Intrusion Detection System.

Features:
  - Sidebar: Model selection (Logistic Regression / Decision Tree / XGBoost)
  - Main: File uploader (.csv / .parquet), evaluation metrics, confusion matrix

Usage:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-IDS | Intrusion Detection System",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add project root to path for imports
import sys
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import transform_new


# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.7);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100, 100, 255, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card h2 {
        color: #7c83ff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    .metric-card p {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: rgba(255,255,255,0.85);
    }

    /* Status badges */
    .badge-success {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-danger {
        background: linear-gradient(135deg, #e53e3e, #fc5c7d);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100, 100, 255, 0.12);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "XGBoost": "xgboost",
}


@st.cache_resource
def load_model(model_key: str):
    """Load a saved model from the models/ directory."""
    path = os.path.join(MODELS_DIR, f"{model_key}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_target_encoder():
    """Load the saved attack_cat LabelEncoder."""
    path = os.path.join(MODELS_DIR, "le_attack_cat.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def check_models_exist() -> bool:
    """Check if trained models and artifacts exist."""
    required = [
        "logistic_regression.joblib",
        "decision_tree.joblib",
        "xgboost.joblib",
        "scaler.joblib",
        "le_proto.joblib",
        "le_service.joblib",
        "le_state.joblib",
        "le_attack_cat.joblib",
        "feature_columns.joblib",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AI-IDS Control Panel")
    st.markdown("---")

    # Model selection
    st.markdown("### Select Model")
    selected_model_name = st.selectbox(
        "Choose a classifier:",
        options=list(MODEL_OPTIONS.keys()),
        index=2,  # Default to XGBoost
        help="Select the ML model to use for inference.",
    )
    selected_model_key = MODEL_OPTIONS[selected_model_name]

    st.markdown("---")

    # Model info
    st.markdown("### Model Info")
    model_descriptions = {
        "Logistic Regression": "A linear model for classification. Fast and interpretable but may underperform on complex patterns.",
        "Decision Tree": "A tree-based model that learns decision rules. Good interpretability with moderate performance.",
        "XGBoost": "Gradient-boosted ensemble of decision trees. State-of-the-art performance for tabular data.",
    }
    st.info(model_descriptions[selected_model_name])

    st.markdown("---")
    st.markdown("### Accepted Formats")
    st.markdown("- `.csv` — Comma-separated values")
    st.markdown("- `.parquet` — Apache Parquet (faster)")

    st.markdown("---")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.3); font-size:0.8rem; text-align:center;'>"
        "UNSW-NB15 IDS • Final Year Project</p>",
        unsafe_allow_html=True,
    )


# ── Main Content ─────────────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>AI-Based Intrusion Detection System</h1>
        <p>Upload network traffic data to detect and classify intrusions using machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Check if models are trained
if not check_models_exist():
    st.error(
        "Models not found. Please train the models first by running:\n\n"
        "```bash\npython src/convert_to_parquet.py\npython src/train.py\n```"
    )
    st.stop()

# File uploader
st.markdown("### Upload Test Data")
uploaded_file = st.file_uploader(
    "Drop your network traffic dataset here",
    type=["csv", "parquet"],
    help="Upload UNSW-NB15 formatted data (.csv or .parquet)",
)

if uploaded_file is not None:
    # ── Load uploaded data ───────────────────────────────────────────
    with st.spinner("Loading data..."):
        if uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            # Raw UNSW-NB15 column names (49 columns, in order)
            RAW_COLS = [
                "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
                "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
                "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
                "smean", "dmean", "trans_depth", "response_body_len", "sjit", "djit",
                "stime", "ltime", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",
                "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
                "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                "attack_cat", "label",
            ]

            # First, read a small sample to check if headers exist
            df_peek = pd.read_csv(uploaded_file, nrows=2, low_memory=False)
            col_names = set(str(c) for c in df_peek.columns)
            key_cols = {"proto", "service", "state", "attack_cat"}

            uploaded_file.seek(0)  # Reset for full read

            if key_cols.intersection(col_names):
                # File has proper headers (training-set / testing-set format)
                df = pd.read_csv(uploaded_file, low_memory=False)
            elif len(df_peek.columns) == 49:
                # Headerless raw UNSW-NB15 CSV — assign column names
                df = pd.read_csv(uploaded_file, header=None, names=RAW_COLS, low_memory=False)
                st.info("Raw UNSW-NB15 CSV detected (no headers). Auto-assigned 49 column names.")
            elif len(df_peek.columns) == 48:
                df = pd.read_csv(uploaded_file, header=None, names=RAW_COLS[:48], low_memory=False)
                st.info("Raw UNSW-NB15 CSV detected (48 columns). Auto-assigned column names.")
            else:
                df = pd.read_csv(uploaded_file, low_memory=False)
                st.warning(f"CSV has {len(df.columns)} columns. Expected 49 (UNSW-NB15). Results may be unreliable.")

            # Normalize attack_cat names (raw files use slightly different names)
            if "attack_cat" in df.columns:
                df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
                # Map raw names to training set names
                attack_rename = {
                    "Backdoors": "Backdoor",
                    "Backdoor": "Backdoor",
                }
                df["attack_cat"] = df["attack_cat"].replace(attack_rename)
                # Normalize empty/NaN to Normal
                df["attack_cat"] = df["attack_cat"].replace({"": "Normal", "nan": "Normal", " ": "Normal"})

    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Data preview
    with st.expander("Preview Uploaded Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # ── Load model & encoder ─────────────────────────────────────────
    model = load_model(selected_model_key)
    le_target = load_target_encoder()

    if model is None or le_target is None:
        st.error("Failed to load model or target encoder.")
        st.stop()

    # ── Preprocess & Predict ─────────────────────────────────────────
    with st.spinner(f"Running {selected_model_name} inference..."):
        X_scaled, y_true = transform_new(df)
        y_pred = model.predict(X_scaled)

    # Decode predictions
    pred_labels = le_target.inverse_transform(y_pred)

    st.markdown("---")

    # ── Results ──────────────────────────────────────────────────────
    has_ground_truth = y_true is not None

    if has_ground_truth:
        # ── Accuracy & Metrics ───────────────────────────────────────
        acc = accuracy_score(y_true, y_pred)

        st.markdown("### Evaluation Results")

        # Metric cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc * 100:.2f}%")
        with col2:
            attack_count = int(np.sum(pred_labels != "Normal"))
            st.metric("Attacks Detected", f"{attack_count:,}")
        with col3:
            normal_count = int(np.sum(pred_labels == "Normal"))
            st.metric("Normal Traffic", f"{normal_count:,}")

        st.markdown("---")

        # ── Two-column layout for charts ─────────────────────────────
        col_left, col_right = st.columns(2)

        # Confusion Matrix
        with col_left:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                ax=ax,
                linewidths=0.5,
                linecolor="#333",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
            ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
            ax.set_title(
                f"Confusion Matrix — {selected_model_name}",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Prediction Distribution
        with col_right:
            st.markdown("### Prediction Distribution")
            pred_counts = pd.Series(pred_labels).value_counts()
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            colors = sns.color_palette("viridis", len(pred_counts))
            bars = ax2.barh(pred_counts.index, pred_counts.values, color=colors)
            ax2.set_xlabel("Count", fontsize=12, fontweight="bold")
            ax2.set_title(
                f"Predicted Class Distribution — {selected_model_name}",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            # Add count labels on bars
            for bar, count in zip(bars, pred_counts.values):
                ax2.text(
                    bar.get_width() + max(pred_counts.values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count:,}",
                    va="center",
                    fontsize=10,
                )
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Precision / Recall / F1 Summary ─────────────────────────
        st.markdown("---")
        st.markdown("### Precision / Recall / F1 Summary")

        label_indices = np.arange(len(le_target.classes_))
        per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=label_indices,
            zero_division=0,
        )

        per_class_df = pd.DataFrame(
            {
                "class": le_target.classes_,
                "precision": per_class_prec,
                "recall": per_class_rec,
                "f1": per_class_f1,
                "support": per_class_support,
            }
        )

        show_micro = st.checkbox(
            "Show micro average (approx equals accuracy for single-label multiclass)",
            value=False,
        )

        avg_order = ["macro", "weighted"]
        if show_micro:
            avg_order = ["micro"] + avg_order

        agg_rows = []
        for avg in avg_order:
            agg_rows.append(
                {
                    "average": avg,
                    "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
                    "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
                    "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
                }
            )

        agg_df = pd.DataFrame(agg_rows).set_index("average")

        st.dataframe(
            agg_df.style.format("{:.4f}").background_gradient(cmap="Blues"),
            use_container_width=True,
        )

        st.markdown("#### Aggregate Metrics Chart")
        st.bar_chart(agg_df)

        with st.expander("Per-Class Metrics Chart", expanded=False):
            per_class_chart_df = per_class_df.set_index("class")[["precision", "recall", "f1"]]
            st.bar_chart(per_class_chart_df)

        # Classification Report
        st.markdown("---")
        st.markdown("### Classification Report")
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=le_target.classes_,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.round(4)

        # Style the dataframe
        st.dataframe(
            report_df.style.format(
                {
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}",
                }
            ).background_gradient(
                cmap="Blues", subset=["precision", "recall", "f1-score"]
            ),
            use_container_width=True,
            height=450,
        )

    else:
        # ── No ground truth — just show predictions ─────────────────
        st.markdown("### Prediction Results")
        st.warning(
            "No `attack_cat` column found in uploaded data. "
            "Showing predictions only (no accuracy metrics)."
        )

        col1, col2 = st.columns(2)
        with col1:
            attack_count = int(np.sum(pred_labels != "Normal"))
            st.metric("Attacks Detected", f"{attack_count:,}")
        with col2:
            normal_count = int(np.sum(pred_labels == "Normal"))
            st.metric("Normal Traffic", f"{normal_count:,}")

        # Prediction distribution
        st.markdown("### Prediction Distribution")
        pred_counts = pd.Series(pred_labels).value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(pred_counts))
        bars = ax.barh(pred_counts.index, pred_counts.values, color=colors)
        ax.set_xlabel("Count", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Predicted Classes — {selected_model_name}",
            fontsize=14,
            fontweight="bold",
        )
        for bar, count in zip(bars, pred_counts.values):
            ax.text(
                bar.get_width() + max(pred_counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}",
                va="center",
                fontsize=10,
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Detailed Predictions Table ───────────────────────────────────
    st.markdown("---")
    st.markdown("### Sample Predictions")
    result_df = df.copy()
    result_df["Predicted_Class"] = pred_labels

    # Color-code predictions
    st.dataframe(
        result_df[
            ["Predicted_Class"]
            + [c for c in ["proto", "service", "state", "dur", "sbytes", "dbytes"] if c in result_df.columns]
        ].head(50),
        use_container_width=True,
    )

    # ── Download Predictions ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Download Results")
    export_cols = ["Predicted_Class"] + [c for c in ["proto", "service", "state", "attack_cat", "dur", "sbytes", "dbytes"] if c in result_df.columns]
    export_df = result_df[export_cols]
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_data,
        file_name="ids_predictions.csv",
        mime="text/csv",
        type="primary",
    )

else:
    # ── Landing state ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem; 
             background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
             border-radius: 16px; border: 1px dashed rgba(100,100,255,0.3);
             margin: 2rem 0;">
            <h2 style="color: #7c83ff; margin-bottom: 1rem;">Upload Your Data</h2>
            <p style="color: rgba(255,255,255,0.6); font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                Drop a <strong>.csv</strong> or <strong>.parquet</strong> file above to begin 
                intrusion detection analysis using the selected ML model.
            </p>
            <p style="color: rgba(255,255,255,0.35); font-size: 0.85rem; margin-top: 1.5rem;">
                Compatible with UNSW-NB15 dataset format
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
