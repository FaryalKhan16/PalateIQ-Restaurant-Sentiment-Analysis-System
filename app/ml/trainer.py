
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from .preprocess import clean_text
from .vectorize import build_count_vectorizer


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "Restaurant_Reviews.tsv"


def load_and_prepare():
    """Load and validate the restaurant review dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        on_bad_lines="skip"
    )

    required_columns = {"Review", "Liked"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'Review' and 'Liked' columns."
        )

    df = df.dropna(subset=["Review", "Liked"]).copy()

    df["Review"] = df["Review"].astype(str).str.strip()
    df["Liked"] = pd.to_numeric(
        df["Liked"], errors="coerce"
    )

    df = df.dropna(subset=["Liked"])
    df = df[df["Review"] != ""]
    df = df[df["Liked"].isin([0, 1])].copy()
    df["Liked"] = df["Liked"].astype(int)

    if df["Liked"].nunique() != 2:
        raise ValueError(
            "Training requires both labels: 0 and 1."
        )

    df["clean"] = df["Review"].apply(clean_text)

    # Remove reviews that become empty after preprocessing.
    df = df[df["clean"].str.strip() != ""].copy()

    if df["Liked"].value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two reviews for a stratified split."
        )

    return df


def train_all_models():
    """Train three models and return metrics and fitted objects."""
    df = load_and_prepare()

    X_text = df["clean"].values
    y = df["Liked"].values

    X_train_text, X_test_text, y_train, y_test = (
        train_test_split(
            X_text,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    # Fit the vocabulary on training data only.
    vectorizer = build_count_vectorizer()

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "naive_bayes": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced"
        ),
    }

    trained_models = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1_weighted"
        )

        results[name] = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                )
            ),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Negative", "Positive"],
                output_dict=True,
                zero_division=0
            ),
        }

        trained_models[name] = model

    return trained_models, vectorizer, results
