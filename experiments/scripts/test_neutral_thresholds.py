
from pathlib import Path
import json

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

# ============================================================
# CONFIGURATION
# Uses train.csv and validation.csv ONLY.
# Never reads or modifies final test data.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "neutral_thresholds"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

CLASSES = ["Negative", "Neutral", "Positive"]
NEUTRAL_WEIGHT_MULTIPLIER = 1.25

# Neutral probability thresholds to test.
THRESHOLDS = np.round(np.arange(0.15, 0.601, 0.025), 3)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def find_column(df, candidates, description):
    columns = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]

    raise ValueError(
        f"Cannot find {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_label(value):
    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    label_mapping = {
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        "positive": "Positive",
        "pos": "Positive",
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

    return label_mapping.get(label)


def load_split(path):
    df = pd.read_csv(path)

    text_col = find_column(
        df,
        ["text", "review", "review_text", "review_text_clean"],
        "review text",
    )

    label_col = find_column(
        df,
        ["sentiment", "label", "actual_sentiment", "target", "rating_review"],
        "sentiment",
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


# ============================================================
# TRAIN THE CURRENT LEADING CANDIDATE
# Logistic Regression + word/character TF-IDF
# Neutral class weight multiplier = 1.25
# ============================================================

weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array(CLASSES),
    y=y_train,
)

class_weights = dict(zip(CLASSES, weights_array))
class_weights["Neutral"] *= NEUTRAL_WEIGHT_MULTIPLIER

print("=" * 72)
print("NEUTRAL-AWARE DECISION THRESHOLD EXPERIMENT")
print("=" * 72)
print(f"Training rows:   {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Class weights:   {class_weights}")

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
            class_weight=class_weights,
            max_iter=3000,
            random_state=42,
        ),
    ),
])

print("\nTraining model on training data...")
model.fit(X_train, y_train)

# Columns of predict_proba correspond to model.classes_.
probabilities = model.predict_proba(X_val)
model_classes = list(model.named_steps["classifier"].classes_)

class_index = {
    label: model_classes.index(label)
    for label in CLASSES
}

p_negative = probabilities[:, class_index["Negative"]]
p_neutral = probabilities[:, class_index["Neutral"]]
p_positive = probabilities[:, class_index["Positive"]]

# Ordinary probability argmax: baseline rule.
baseline_predictions = np.array(model_classes)[
    np.argmax(probabilities, axis=1)
]


# ============================================================
# EVALUATION HELPERS
# ============================================================

def evaluate_predictions(name, predictions, threshold=None):
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

    result = {
        "rule": name,
        "neutral_threshold": threshold,
        "validation_accuracy": accuracy_score(y_val, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "neutral_precision": report["Neutral"]["precision"],
        "neutral_recall": report["Neutral"]["recall"],
        "neutral_f1": report["Neutral"]["f1-score"],
        "negative_f1": report["Negative"]["f1-score"],
        "positive_f1": report["Positive"]["f1-score"],
        "neutral_predicted_count": int(
            np.sum(np.asarray(predictions) == "Neutral")
        ),
    }

    details = {
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": CLASSES,
    }

    return result, details


results = []
details = {}

# Baseline: choose the class with the highest probability.
baseline_result, baseline_details = evaluate_predictions(
    "argmax_baseline",
    baseline_predictions,
)

results.append(baseline_result)
details["argmax_baseline"] = baseline_details

print("\nBaseline argmax:")
print(
    f"Accuracy={baseline_result['validation_accuracy']:.4f} | "
    f"Macro F1={baseline_result['macro_f1']:.4f} | "
    f"Neutral F1={baseline_result['neutral_f1']:.4f} | "
    f"Neutral recall={baseline_result['neutral_recall']:.4f}"
)


# ============================================================
# THRESHOLD EXPERIMENT
#
# If P(Neutral) >= threshold, predict Neutral.
# Otherwise, choose whichever is stronger: Negative or Positive.
#
# Thresholds are evaluated on validation data only.
# ============================================================

for threshold in THRESHOLDS:
    non_neutral_predictions = np.where(
        p_negative >= p_positive,
        "Negative",
        "Positive",
    )

    predictions = np.where(
        p_neutral >= threshold,
        "Neutral",
        non_neutral_predictions,
    )

    name = f"neutral_probability_at_least_{threshold:.3f}"

    result, detail = evaluate_predictions(
        name,
        predictions,
        threshold=float(threshold),
    )

    results.append(result)
    details[name] = detail


# ============================================================
# SAVE AND DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["macro_f1", "neutral_f1"],
    ascending=False,
)

comparison_path = OUTPUT_DIR / "validation_threshold_comparison.csv"
reports_path = OUTPUT_DIR / "validation_threshold_reports.json"

results_df.to_csv(comparison_path, index=False)

with open(reports_path, "w", encoding="utf-8") as file:
    json.dump(details, file, indent=2)

display_columns = [
    "rule",
    "neutral_threshold",
    "validation_accuracy",
    "macro_f1",
    "neutral_precision",
    "neutral_recall",
    "neutral_f1",
    "negative_f1",
    "positive_f1",
    "neutral_predicted_count",
]

print("\n" + "=" * 120)
print("VALIDATION RESULTS — SORTED BY MACRO F1")
print("=" * 120)

print(
    results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

# Report the best threshold under two different objectives.
# Neither choice is evaluated against final test data.
threshold_results = results_df[
    results_df["neutral_threshold"].notna()
]

best_macro = threshold_results.sort_values(
    ["macro_f1", "neutral_f1"],
    ascending=False,
).iloc[0]

best_neutral_f1 = threshold_results.sort_values(
    ["neutral_f1", "macro_f1"],
    ascending=False,
).iloc[0]

print("\n" + "=" * 72)
print("TOP THRESHOLD BY MACRO F1")
print("=" * 72)
print(best_macro[display_columns].to_string())

print("\n" + "=" * 72)
print("TOP THRESHOLD BY NEUTRAL F1")
print("=" * 72)
print(best_neutral_f1[display_columns].to_string())

print("\nFiles saved:")
print(comparison_path)
print(reports_path)

print("\nSafety check:")
print("- Only train.csv and validation.csv were read.")
print("- Final test data was not read.")
print("- Existing models and final test results were not modified.")
