
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text"
LABEL_COL = "rating_review"
LABEL_ORDER = ["Negative", "Neutral", "Positive"]
RANDOM_STATE = 42


def load_split(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_dataset.py first."
        )

    df = pd.read_csv(path)

    if TEXT_COL not in df or LABEL_COL not in df:
        raise ValueError(
            f"{filename} must contain {TEXT_COL!r} and {LABEL_COL!r}."
        )

    if df[TEXT_COL].isna().any() or df[LABEL_COL].isna().any():
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
            random_state=RANDOM_STATE,
        )

    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def main():
    print("=" * 70)
    print("THREE-CLASS SENTIMENT MODEL COMPARISON")
    print("=" * 70)

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")

    X_train = train_df[TEXT_COL].astype(str)
    y_train = train_df[LABEL_COL].astype(str)

    X_val = validation_df[TEXT_COL].astype(str)
    y_val = validation_df[LABEL_COL].astype(str)

    if set(y_train.unique()) != set(LABEL_ORDER):
        raise ValueError("Training split does not contain all 3 classes.")

    if set(y_val.unique()) != set(LABEL_ORDER):
        raise ValueError("Validation split does not contain all 3 classes.")

    print(f"\nTraining rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    # Baseline: predict the most frequent training label for every review.
    majority_label = y_train.value_counts().idxmax()
    baseline_predictions = [majority_label] * len(y_val)

    models = {
        "linear_svc": build_pipeline("linear_svc"),
        "logistic_regression": build_pipeline("logistic_regression"),
    }

    results = []
    reports = {}
    confusion_matrices = {}

    # Evaluate baseline and both models on the same validation examples.
    candidates = [
        ("majority_baseline", None, baseline_predictions),
    ]

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        candidates.append((name, model, predictions))

    for name, model, predictions in candidates:
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_val, predictions),
            "macro_precision": precision_score(
                y_val, predictions,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            ),
            "macro_recall": recall_score(
                y_val, predictions,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            ),
            "macro_f1": f1_score(
                y_val, predictions,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            ),
        }

        results.append(metrics)

        reports[name] = classification_report(
            y_val,
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_val, predictions, labels=LABEL_ORDER
        )

        confusion_matrices[name] = pd.DataFrame(
            matrix,
            index=[f"actual_{x}" for x in LABEL_ORDER],
            columns=[f"predicted_{x}" for x in LABEL_ORDER],
        )

        print(f"\n--- {name} ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro precision: {metrics['macro_precision']:.4f}")
        print(f"Macro recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print("\nClassification report:")
        print(classification_report(
            y_val,
            predictions,
            labels=LABEL_ORDER,
            zero_division=0,
        ))

    results_df = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )

    results_path = RESULTS_DIR / "validation_comparison.csv"
    results_df.to_csv(results_path, index=False)

    for name, matrix_df in confusion_matrices.items():
        matrix_df.to_csv(
            RESULTS_DIR / f"{name}_validation_confusion_matrix.csv"
        )

    with open(
        RESULTS_DIR / "validation_classification_reports.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(reports, file, indent=4)

    # Select between the two actual ML models using validation macro F1.
    # The majority baseline is a reference, not a model candidate.
    ml_results = results_df[
        results_df["model"].isin(
            ["linear_svc", "logistic_regression"]
        )
    ]

    selected_name = ml_results.iloc[0]["model"]

    selection = {
        "selected_model": selected_name,
        "selection_metric": "validation_macro_f1",
        "selected_validation_macro_f1": float(
            ml_results.iloc[0]["macro_f1"]
        ),
        "test_set_used_for_selection": False,
    }

    with open(
        RESULTS_DIR / "selected_model.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selection, file, indent=4)

    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {selected_name}")
    print("Selection criterion: highest validation macro F1")
    print("\nResults saved in:", RESULTS_DIR.resolve())
    print("The final test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
    