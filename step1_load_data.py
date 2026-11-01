"""Load and prepare the UCI Red Wine Quality dataset for Step 1."""

import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
OUTPUT_PATH = "wine_dataset_cleaned.csv"


def main() -> None:
    df = pd.read_csv(DATASET_URL, sep=";")
    df.columns = df.columns.str.replace(" ", "_", regex=False)

    df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"]
    df["bound_sulfur_dioxide"] = (
        df["total_sulfur_dioxide"] - df["free_sulfur_dioxide"]
    )
    df["sugar_to_acidity_ratio"] = df["residual_sugar"] / (
        df["total_acidity"] + 1e-5
    )
    df["quality_label"] = (df["quality"] >= 6).astype(int)
    df = df.drop(columns=["quality"])

    feature_columns = [column for column in df.columns if column != "quality_label"]
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Total input features: {len(feature_columns)}")
    print(f"Missing value count: {int(df.isna().sum().sum())}")
    print("Feature names:")
    for feature in feature_columns:
        print(f"- {feature}")
    print("Target class counts:")
    print(df["quality_label"].value_counts().sort_index().to_string())
    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")

    if len(df) != 1599 or len(feature_columns) != 14:
        raise ValueError(
            f"Unexpected dataset dimensions: {len(df)} rows, "
            f"{len(feature_columns)} input features."
        )


if __name__ == "__main__":
    main()
