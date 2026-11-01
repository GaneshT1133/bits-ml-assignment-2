"""Interactive evaluation dashboard for the trained wine-quality classifiers."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
TARGET_COLUMN = "quality_label"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(
    page_title="Wine Quality · Model Lab",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#17231f; --muted:#66746d; --wine:#8c3047; --sage:#dce8df; --paper:#f7f8f4; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp { background:linear-gradient(180deg,#f7f8f4 0%,#f2f5f0 100%); color:var(--ink); }
      h1,h2,h3 { font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-.03em; }
      .hero { padding:1.55rem 1.8rem; border-radius:22px; color:#fff; background:linear-gradient(115deg,#263f35 0%,#3f6654 58%,#8c3047 145%); margin:.4rem 0 1.25rem 0; box-shadow:0 12px 30px rgba(34,58,47,.12); }
      .hero-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; color:#d5e5d8; font-weight:700; }
      .hero h1 { color:#fff; font-size:2.1rem; margin:.35rem 0; }
      .hero p { color:#e3ece5; margin:0; max-width:760px; }
      .section-label { color:#7a3547; font-weight:700; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #e6ebe5; padding:1rem 1.05rem; border-radius:15px; box-shadow:0 4px 14px rgba(31,56,43,.04); }
      [data-testid="stSidebar"] { background:#eef2ec; }
      div[data-testid="stDataFrame"] { border:1px solid #e3e9e2; border-radius:14px; overflow:hidden; }
      .stAlert { border-radius:12px; }
    </style>
    <div class="hero">
      <div class="hero-kicker">Machine Learning · Evaluation Studio</div>
      <h1>Red Wine Quality Classifier Lab</h1>
      <p>Explore five trained classifiers on labeled test data. Compare metrics, inspect the confusion matrix, and review class-level performance—without retraining.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the saved models and scaler…")
def load_artifacts():
    """Load persisted training artifacts once; never fit a model in the app."""
    scaler_path = MODEL_DIR / "scaler.pkl"
    missing_paths = [
        path
        for path in [scaler_path, *(MODEL_DIR / name for name in MODEL_FILES.values())]
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Required model artifacts are missing: "
            + ", ".join(path.name for path in missing_paths)
        )

    scaler = joblib.load(scaler_path)
    models = {
        model_name: joblib.load(MODEL_DIR / filename)
        for model_name, filename in MODEL_FILES.items()
    }
    return scaler, models


def load_test_data(uploaded_file, use_sample_data: bool) -> tuple[pd.DataFrame, str]:
    """Read uploaded CSV or repository sample test data."""
    if use_sample_data:
        if not TEST_DATA_PATH.is_file():
            raise FileNotFoundError("Repository sample test_data.csv was not found.")
        return pd.read_csv(TEST_DATA_PATH), "Repository sample · test_data.csv"
    if uploaded_file is None:
        raise ValueError("Upload a labeled CSV or enable the repository sample dataset.")
    return pd.read_csv(uploaded_file), uploaded_file.name


def validate_dataset(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Check schema and values, then return features and binary labels."""
    if frame.empty:
        raise ValueError("The selected CSV contains no data rows.")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"The CSV must include the target column '{TARGET_COLUMN}'.")

    expected_columns = set(feature_columns) | {TARGET_COLUMN}
    received_columns = set(frame.columns)
    missing = sorted(expected_columns - received_columns)
    extra = sorted(received_columns - expected_columns)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError("CSV columns do not match the model schema (" + "; ".join(details) + ").")

    ordered = frame.loc[:, [*feature_columns, TARGET_COLUMN]].copy()
    for column in feature_columns:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    ordered[TARGET_COLUMN] = pd.to_numeric(ordered[TARGET_COLUMN], errors="coerce")
    if ordered.isna().any().any():
        raise ValueError("The CSV contains missing or non-numeric feature/target values.")
    if not np.isfinite(ordered[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("The CSV contains infinite feature values.")
    if not ordered[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("quality_label must contain only binary values 0 and 1.")

    return ordered[feature_columns], ordered[TARGET_COLUMN].astype(int)


def evaluate_model(model, X_scaled: pd.DataFrame, y_true: pd.Series) -> dict:
    """Compute the six required metrics from predictions on the selected data."""
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else float("nan")
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": {
            "Accuracy": accuracy_score(y_true, predictions),
            "AUC": auc,
            "Precision": precision_score(y_true, predictions, zero_division=0),
            "Recall": recall_score(y_true, predictions, zero_division=0),
            "F1": f1_score(y_true, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y_true, predictions),
        },
    }


def render_model_details(
    model_name: str,
    evaluation: dict,
    y_true: pd.Series,
) -> None:
    """Display one model's metric cards, confusion matrix, and class report."""
    st.markdown(f"<div class='section-label'>Model report · {model_name}</div>", unsafe_allow_html=True)
    metric_columns = st.columns(6)
    for column, metric_name in zip(metric_columns, METRIC_NAMES):
        value = evaluation["metrics"][metric_name]
        display_value = "N/A" if pd.isna(value) else f"{value:.3f}"
        column.metric(metric_name, display_value)

    chart_column, report_column = st.columns([1.0, 1.15], gap="large")
    with chart_column:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(y_true, evaluation["predictions"], labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5.4, 4.2))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="crest",
            cbar=False,
            linewidths=1,
            linecolor="white",
            xticklabels=["Quality < 6", "Quality ≥ 6"],
            yticklabels=["Quality < 6", "Quality ≥ 6"],
            ax=axis,
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with report_column:
        st.subheader("Classification report")
        report = classification_report(
            y_true,
            evaluation["predictions"],
            labels=[0, 1],
            target_names=["Quality < 6", "Quality ≥ 6"],
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).T
        st.dataframe(report_frame.round(3), width="stretch")


with st.sidebar:
    st.markdown("### Evaluation controls")
    uploaded_file = st.file_uploader(
        "Upload a labeled test CSV",
        type=["csv"],
        help="Include all 14 model input features and the quality_label target column.",
    )
    prefer_sample_data = st.toggle(
        "Use repository sample test data",
        value=False,
        help="Loads test_data.csv shipped with this repository for quick evaluation.",
    )
    st.divider()
    st.caption("With no upload, the sample CSV loads automatically. The app uses saved models and never retrains.")

try:
    scaler, trained_models = load_artifacts()
except (OSError, ValueError, EOFError) as error:
    st.error(f"Could not load model artifacts: {error}")
    st.stop()

feature_columns = list(scaler.feature_names_in_)
if len(feature_columns) != 14:
    st.error(f"Expected 14 trained features, but the saved scaler has {len(feature_columns)}.")
    st.stop()

try:
    use_sample_data = prefer_sample_data or uploaded_file is None
    raw_data, data_source = load_test_data(uploaded_file, use_sample_data)
    X, y_true = validate_dataset(raw_data, feature_columns)
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

X_scaled = scaler.transform(X)
evaluations = {
    model_name: evaluate_model(model, X_scaled, y_true)
    for model_name, model in trained_models.items()
}

left, right = st.columns([1.2, 1])
with left:
    st.markdown("<div class='section-label'>Current evaluation</div>", unsafe_allow_html=True)
    st.subheader(data_source)
with right:
    st.markdown(
        f"<div style='text-align:right;padding-top:1.45rem;color:#66746d'>"
        f"{len(raw_data):,} labeled rows&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"{len(feature_columns)} input features</div>",
        unsafe_allow_html=True,
    )

selection_options = ["All Models Comparison", *trained_models.keys()]
selected_model = st.selectbox("Choose a model view", selection_options)

if selected_model == "All Models Comparison":
    st.markdown("### Held-out performance overview")
    comparison = pd.DataFrame(
        [
            {"ML Model Name": name, **result["metrics"]}
            for name, result in evaluations.items()
        ]
    )
    st.dataframe(
        comparison.style.format(
            {
                metric: lambda value: "N/A" if pd.isna(value) else f"{value:.3f}"
                for metric in METRIC_NAMES
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("Metrics are calculated dynamically from the selected labeled dataset.")
    tabs = st.tabs(list(evaluations.keys()))
    for tab, (model_name, evaluation) in zip(tabs, evaluations.items()):
        with tab:
            render_model_details(model_name, evaluation, y_true)
else:
    render_model_details(
        selected_model,
        evaluations[selected_model],
        y_true,
    )
