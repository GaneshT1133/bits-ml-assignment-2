"""Create the stratified train/test split and persist the training scaler."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("wine_dataset_cleaned.csv")
TEST_OUTPUT_PATH = Path("test_data.csv")
SCALER_OUTPUT_PATH = Path("model") / "scaler.pkl"
TARGET_COLUMN = "quality_label"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
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

    # Export the test partition before scaling for app-side preprocessing.
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(TEST_OUTPUT_PATH, index=False)

    # Learn scaling parameters from training features only.
    scaler = StandardScaler()
    scaler.fit(X_train)
    SCALER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Input features: {len(feature_columns)}")
    print("Training target proportions:")
    print(y_train.value_counts(normalize=True).sort_index().to_string())
    print("Test target proportions:")
    print(y_test.value_counts(normalize=True).sort_index().to_string())
    print(f"Exported unscaled test data: {test_data.shape}")
    print(f"Scaler fitted on training rows only: {scaler.n_samples_seen_ == len(X_train)}")
    print(f"Saved scaler: {SCALER_OUTPUT_PATH}")

    if len(X_train) != 1279 or len(X_test) != 320:
        raise ValueError("Unexpected split size; expected 1279 train and 320 test rows.")
    if len(feature_columns) != 14 or scaler.n_samples_seen_ != len(X_train):
        raise ValueError("Unexpected feature count or scaler training sample count.")
    if test_data.isna().sum().sum() != 0:
        raise ValueError("The exported test data contains missing values.")


if __name__ == "__main__":
    main()
