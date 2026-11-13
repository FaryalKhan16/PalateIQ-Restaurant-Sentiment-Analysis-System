
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# This script reads ONLY training and validation data.
# It never reads or modifies the final test files.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "neutral_improvement"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

CLASS_ORDER = ["Negative", "Neutral", "Positive"]
NEUTRAL_CLASS = "Neutral"

# Larger values give Neutral more weight relative to the
# automatically calculated balanced class weights.
NEUTRAL_WEIGHT_MULTIPLIERS = [1.0, 1.25, 1.5]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def normalize_label(value):
    """Standardize common three-class sentiment labels."""

    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    mapping = {
        "negative": "Negative",
        "neg": "Negative",
        "0": "Negative",
        "1": "Negative",
        "1.0": "Negative",
        "2": "Negative",
        "2.0": "Negative",

        "neutral": "Neutral",
        "neu": "Neutral",
        "3": "Neutral",
        "3.0": "Neutral",

        "positive": "Positive",
        "pos": "Positive",
        "4": "Positive",
        "4.0": "Positive",
        "5": "Positive",
        "5.0": "Positive",
    }

    return mapping.get(label)


def find_column(df, candidates, description):
    """Find a column using common column-name alternatives."""

    columns = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]

    raise ValueError(
        f"Could not find the {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_dataset(path):
    df = pd.read_csv(path)

    text_col = find_column(
        df,
        ["text", "review", "review_text", "review_text_clean"],
        "review text",
    )

    label_col = find_column(
        df,
        [
            "sentiment",
            "label",
            "actual_sentiment",
            "target",
            "rating_review",
        ],
        "sentiment label",
    )

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "sentiment"]

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].apply(normalize_label)

    df = df[
        (df["text"] != "")
        & (df["sentiment"].notna())
    ].copy()

    unknown = set(df["sentiment"].unique()) - set(CLASS_ORDER)

    if unknown:
        raise ValueError(f"Unexpected sentiment labels: {unknown}")

    return df.reset_index(drop=True)


train_df = load_dataset(TRAIN_PATH)
val_df = load_dataset(VALIDATION_PATH)

X_train = train_df["text"]
y_train = train_df["sentiment"]

X_val = val_df["text"]
y_val = val_df["sentiment"]

if set(y_train.unique()) != set(CLASS_ORDER):
    raise ValueError(
        "Training data must contain Negative, Neutral, and Positive."
    )

if set(y_val.unique()) != set(CLASS_ORDER):
    raise ValueError(
        "Validation data must contain all three sentiment classes."
    )

print("=" * 65)
print("NEUTRAL SENTIMENT IMPROVEMENT EXPERIMENTS")
print("=" * 65)
print(f"Training reviews:   {len(train_df)}")
print(f"Validation reviews: {len(val_df)}")
print("\nTraining class distribution:")
print(y_train.value_counts().reindex(CLASS_ORDER))
print("\nValidation class distribution:")
print(y_val.value_counts().reindex(CLASS_ORDER))


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.array(CLASS_ORDER)

balanced_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

balanced_weights = dict(zip(classes, balanced_values))

print("\nAutomatically calculated balanced weights:")
print(balanced_weights)


def make_class_weights(neutral_multiplier):
    weights = balanced_weights.copy()

    weights[NEUTRAL_CLASS] *= neutral_multiplier

    return weights


# ============================================================
# TEXT FEATURE CONFIGURATIONS
# ============================================================

def make_word_features():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
    )


def make_word_and_character_features():
    """
    Word features capture words and short phrases.
    Character features can capture spelling variations,
    informal language, and some slang.
    """

    return FeatureUnion([
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
    ])


# ============================================================
# EXPERIMENTS
# ============================================================

def build_model(model_name, feature_name, neutral_multiplier):
    weights = make_class_weights(neutral_multiplier)

    if model_name == "LogisticRegression":
        classifier = LogisticRegression(
            C=1.0,
            class_weight=weights,
            max_iter=3000,
            random_state=42,
        )

    elif model_name == "LinearSVC":
        classifier = LinearSVC(
            C=1.0,
            class_weight=weights,
            random_state=42,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if feature_name == "word":
        features = make_word_features()
    elif feature_name == "word_char":
        features = make_word_and_character_features()
    else:
        raise ValueError(f"Unknown feature configuration: {feature_name}")

    return Pipeline([
        ("tfidf", features),
        ("classifier", classifier),
    ])


experiments = []

for model_name in ["LogisticRegression", "LinearSVC"]:
    for feature_name in ["word", "word_char"]:
        for multiplier in NEUTRAL_WEIGHT_MULTIPLIERS:
            experiments.append({
                "model_name": model_name,
                "feature_name": feature_name,
                "neutral_multiplier": multiplier,
            })


# ============================================================
# TRAIN ON TRAINING DATA; SELECT/COMPARE ON VALIDATION DATA
# ============================================================

results = []
reports = {}

for number, config in enumerate(experiments, start=1):
    model_name = config["model_name"]
    feature_name = config["feature_name"]
    multiplier = config["neutral_multiplier"]

    experiment_name = (
        f"{model_name}__{feature_name}__neutral_weight_{multiplier}"
    )

    print("\n" + "-" * 65)
    print(f"Experiment {number}/{len(experiments)}: {experiment_name}")
    print("-" * 65)

    model = build_model(
        model_name,
        feature_name,
        multiplier,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    report = classification_report(
        y_val,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=CLASS_ORDER,
    )

    neutral_metrics = report[NEUTRAL_CLASS]
    macro_metrics = report["macro avg"]

    result = {
        "experiment": experiment_name,
        "model": model_name,
        "features": feature_name,
        "neutral_weight_multiplier": multiplier,
        "validation_accuracy": accuracy_score(y_val, predictions),
        "validation_macro_precision": macro_metrics["precision"],
        "validation_macro_recall": macro_metrics["recall"],
        "validation_macro_f1": macro_metrics["f1-score"],
        "neutral_precision": neutral_metrics["precision"],
        "neutral_recall": neutral_metrics["recall"],
        "neutral_f1": neutral_metrics["f1-score"],
        "negative_f1": report["Negative"]["f1-score"],
        "positive_f1": report["Positive"]["f1-score"],
    }

    results.append(result)

    reports[experiment_name] = {
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": CLASS_ORDER,
        "class_weights": {
            key: float(value)
            for key, value in make_class_weights(multiplier).items()
        },
    }

    print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    print(f"Macro F1:            {result['validation_macro_f1']:.4f}")
    print(f"Neutral precision:   {result['neutral_precision']:.4f}")
    print(f"Neutral recall:      {result['neutral_recall']:.4f}")
    print(f"Neutral F1:          {result['neutral_f1']:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

# Sort by overall macro F1 first, then neutral F1.
# This exposes the trade-off instead of optimizing only
# one class at the expense of the others.
results_df = results_df.sort_values(
    by=["validation_macro_f1", "neutral_f1"],
    ascending=False,
)

comparison_path = OUTPUT_DIR / "validation_comparison.csv"
reports_path = OUTPUT_DIR / "validation_reports.json"

results_df.to_csv(comparison_path, index=False)

with open(reports_path, "w", encoding="utf-8") as file:
    json.dump(reports, file, indent=2)

print("\n" + "=" * 80)
print("VALIDATION RESULTS — NO TEST DATA USED")
print("=" * 80)

display_columns = [
    "experiment",
    "validation_accuracy",
    "validation_macro_f1",
    "neutral_precision",
    "neutral_recall",
    "neutral_f1",
    "negative_f1",
    "positive_f1",
]

print(
    results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nSaved comparison to:")
print(comparison_path)

print("\nSaved detailed reports to:")
print(reports_path)

print("\nNo final test files were read, changed, or regenerated.")
print("Review the validation trade-offs before choosing a candidate.")
