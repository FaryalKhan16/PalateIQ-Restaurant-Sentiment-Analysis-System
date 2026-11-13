
from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/three_class")
LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def load_split(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_dataset.py first."
        )

    df = pd.read_csv(path)

    if df["text"].isna().any() or df["rating_review"].isna().any():
        raise ValueError(f"Missing text or labels in {filename}.")

    return df


def build_pipeline(model_name):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )

    if model_name == "linear_svc":
        classifier = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )
    else:
        raise ValueError(
            f"Unsupported selected model: {model_name}"
        )

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def main():
    selection_path = RESULTS_DIR / "selected_model.json"

    if not selection_path.exists():
        raise FileNotFoundError(
            "Run compare_models.py first to select a model."
        )

    with open(selection_path, encoding="utf-8") as file:
        selection = json.load(file)

    model_name = selection["selected_model"]

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_df = load_split("test.csv")

    # Combine train and validation only after model selection.
    development_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    X_development = development_df["text"].astype(str)
    y_development = development_df["rating_review"].astype(str)

    X_test = test_df["text"].astype(str)
    y_test = test_df["rating_review"].astype(str)

    if set(y_development.unique()) != set(LABEL_ORDER):
        raise ValueError("Development data does not contain all classes.")

    if set(y_test.unique()) != set(LABEL_ORDER):
        raise ValueError("Test data does not contain all classes.")

    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Selected model: {model_name}")
    print(f"Training + validation rows: {len(development_df)}")
    print(f"Final test rows: {len(test_df)}")

    model = build_pipeline(model_name)
    model.fit(X_development, y_development)

    predictions = model.predict(X_test)

    metrics = {
        "model": model_name,
        "training_plus_validation_rows": len(development_df),
        "test_rows": len(test_df),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(
            precision_score(
                y_test, predictions,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test, predictions,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test, predictions,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test, predictions, labels=LABEL_ORDER
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{x}" for x in LABEL_ORDER],
        columns=[f"predicted_{x}" for x in LABEL_ORDER],
    )

    # Save final test results separately from validation results.
    with open(
        RESULTS_DIR / "final_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        RESULTS_DIR / "final_test_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    matrix_df.to_csv(
        RESULTS_DIR / "final_test_confusion_matrix.csv"
    )

    predictions_df = pd.DataFrame({
        "text": X_test.to_numpy(),
        "actual_sentiment": y_test.to_numpy(),
        "predicted_sentiment": predictions,
    })

    predictions_df.to_csv(
        RESULTS_DIR / "final_test_predictions.csv",
        index=False,
    )

    print("\nFinal test metrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0,
    ))

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(matrix_df.to_string())

    print("\nResults saved in:", RESULTS_DIR.resolve())
    print("Keep these final test results for reporting.")
    print("Do not tune the model against this test set.")


if __name__ == "__main__":
    main()
    