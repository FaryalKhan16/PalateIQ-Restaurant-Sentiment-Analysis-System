from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "three_class"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "restaurant_sentiment_pipeline.joblib"

LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def main():
    selection_path = RESULTS_DIR / "selected_model.json"

    if not selection_path.exists():
        raise FileNotFoundError(
            f"Model selection record not found: {selection_path}"
        )

    with selection_path.open(encoding="utf-8") as file:
        selection = json.load(file)

    if selection.get("selected_model") != "logistic_regression":
        raise ValueError(
            "The selected model is not logistic_regression. "
            "Review the selection record before continuing."
        )

    # Intentionally load only training and validation data.
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    required = {"text", "rating_review"}

    for name, df in [
        ("train.csv", train_df),
        ("validation.csv", validation_df),
    ]:
        if not required.issubset(df.columns):
            raise ValueError(f"{name} is missing required columns.")
        if df["text"].isna().any() or df["rating_review"].isna().any():
            raise ValueError(f"{name} contains missing text or labels.")

    development_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    X = development_df["text"].astype(str)
    y = development_df["rating_review"].astype(str)

    if set(y.unique()) != set(LABEL_ORDER):
        raise ValueError("Development data must contain all three classes.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )),
    ])

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Combined development rows: {len(development_df)}")
    print("Final test data will not be loaded.")

    pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model": "logistic_regression",
        "features": "TF-IDF word unigrams and bigrams",
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "fitted_rows": len(development_df),
        "classes": [str(label) for label in pipeline.classes_],
        "final_test_used_for_training": False,
    }

    with (MODEL_DIR / "model_info.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nDeployment model saved successfully:")
    print(MODEL_PATH)
    print("Metadata saved to:", MODEL_DIR / "model_info.json")
    print("No final test evaluation was performed.")


if __name__ == "__main__":
    main()
