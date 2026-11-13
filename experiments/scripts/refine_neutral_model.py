
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "neutral_refinement"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

CLASSES = ["Negative", "Neutral", "Positive"]
NEUTRAL_WEIGHTS = [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, candidates, description):
    columns = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]

    raise ValueError(
        f"Cannot find {description}. Available columns: {list(df.columns)}"
    )


def normalize_label(value):
    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    mapping = {
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        "positive": "Positive",
        "pos": "Positive",
    }

    # Also support datasets with original 1–5 star labels.
    star_mapping = {
        "1": "Negative",
        "1.0": "Negative",
        "2": "Negative",
        "2.0": "Negative",
        "3": "Neutral",
        "3.0": "Neutral",
        "4": "Positive",
        "4.0": "Positive",
        "5": "Positive",
        "5.0": "Positive",
    }

    return mapping.get(label, star_mapping.get(label))


def load_split(path):
    df = pd.read_csv(path)

    text_col = find_column(
        df,
        ["text", "review", "review_text", "review_text_clean"],
        "review text column",
    )
    label_col = find_column(
        df,
        ["sentiment", "label", "actual_sentiment", "target", "rating_review"],
        "sentiment label column",
    )

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "sentiment"]

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].apply(normalize_label)

    df = df[
        (df["text"] != "") & df["sentiment"].notna()
    ].copy()

    if not set(df["sentiment"].unique()).issubset(set(CLASSES)):
        raise ValueError(f"Unexpected labels in {path}")

    return df.reset_index(drop=True)


train = load_split(TRAIN_PATH)
validation = load_split(VALIDATION_PATH)

X_train = train["text"]
y_train = train["sentiment"]
X_val = validation["text"]
y_val = validation["sentiment"]

if set(y_train.unique()) != set(CLASSES):
    raise ValueError("Training data must contain all three classes.")

if set(y_val.unique()) != set(CLASSES):
    raise ValueError("Validation data must contain all three classes.")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array(CLASSES),
    y=y_train,
)
base_weights = dict(zip(CLASSES, class_weights_array))

print("=" * 70)
print("REFINING NEUTRAL SENTIMENT WEIGHT")
print("=" * 70)
print(f"Training reviews: {len(train)}")
print(f"Validation reviews: {len(validation)}")
print(f"Base balanced weights: {base_weights}")
print(f"Neutral multipliers: {NEUTRAL_WEIGHTS}")

results = []
reports = {}

for multiplier in NEUTRAL_WEIGHTS:
    weights = base_weights.copy()
    weights["Neutral"] *= multiplier

    model = Pipeline([
        (
            "tfidf",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=80000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=60000,
                        sublinear_tf=True,
                    ),
                ),
            ]),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight=weights,
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])

    name = f"word_char_neutral_weight_{multiplier}"
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    report = classification_report(
        y_val,
        predictions,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=CLASSES,
    )

    neutral = report["Neutral"]
    macro = report["macro avg"]

    results.append({
        "experiment": name,
        "neutral_weight_multiplier": multiplier,
        "validation_accuracy": accuracy_score(y_val, predictions),
        "validation_macro_f1": macro["f1-score"],
        "validation_macro_precision": macro["precision"],
        "validation_macro_recall": macro["recall"],
        "neutral_precision": neutral["precision"],
        "neutral_recall": neutral["recall"],
        "neutral_f1": neutral["f1-score"],
        "negative_f1": report["Negative"]["f1-score"],
        "positive_f1": report["Positive"]["f1-score"],
    })

    reports[name] = {
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": CLASSES,
        "class_weights": {
            key: float(value) for key, value in weights.items()
        },
    }

    print(
        f"Accuracy={accuracy_score(y_val, predictions):.4f} | "
        f"Macro F1={macro['f1-score']:.4f} | "
        f"Neutral P={neutral['precision']:.4f} | "
        f"Neutral R={neutral['recall']:.4f} | "
        f"Neutral F1={neutral['f1-score']:.4f}"
    )

results_df = pd.DataFrame(results).sort_values(
    by=["validation_macro_f1", "neutral_f1"],
    ascending=False,
)

results_df.to_csv(
    OUTPUT_DIR / "validation_comparison.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "validation_reports.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(reports, file, indent=2)

print("\n" + "=" * 90)
print("RANKED VALIDATION RESULTS")
print("=" * 90)

print(
    results_df[
        [
            "experiment",
            "validation_accuracy",
            "validation_macro_f1",
            "neutral_precision",
            "neutral_recall",
            "neutral_f1",
            "negative_f1",
            "positive_f1",
        ]
    ].round(4).to_string(index=False)
)

print(f"\nResults saved to: {OUTPUT_DIR}")
print("Final test data was not read or modified.")
