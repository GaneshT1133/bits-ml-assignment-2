"""Train the five assignment classifiers and evaluate them on the held-out split."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("wine_dataset_cleaned.csv")
MODEL_DIR = Path("model")
TARGET_COLUMN = "quality_label"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    if list(scaler.feature_names_in_) != feature_columns:
        raise ValueError("Loaded scaler feature order does not match the dataset.")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (
            "logistic_regression.pkl",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": (
            "decision_tree.pkl",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
        "KNN": ("knn.pkl", KNeighborsClassifier()),
        "Naive Bayes": ("naive_bayes.pkl", GaussianNB()),
        "Support Vector Machine": (
            "svm.pkl",
            SVC(probability=True, random_state=RANDOM_STATE),
        ),
    }

    metric_rows = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, (filename, model) in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        positive_probabilities = model.predict_proba(X_test_scaled)[:, 1]

        metric_rows.append(
            {
                "ML Model Name": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, positive_probabilities),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "MCC": matthews_corrcoef(y_test, predictions),
            }
        )
        joblib.dump(model, MODEL_DIR / filename)
        print(f"Saved {model_name}: {MODEL_DIR / filename}")

    metrics = pd.DataFrame(metric_rows)
    print("\nHeld-out test metrics:")
    print(metrics.round(4).to_string(index=False))

    if len(feature_columns) != 14 or len(X_test) != 320:
        raise ValueError("Unexpected feature count or test-set size.")
    if len(metric_rows) != 5 or not metrics.iloc[:, 1:].notna().all().all():
        raise ValueError("Expected five models with complete numeric metrics.")


if __name__ == "__main__":
    main()
