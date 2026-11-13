
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DATA_PATH = Path("data/yelp_sentiment_master_dataset.csv")
OUTPUT_DIR = Path("data/processed")

TEXT_COL = "text"
LABEL_COL = "rating_review"
STARS_COL = "stars_review"

RANDOM_STATE = 42
EXPECTED_LABELS = {"Positive", "Neutral", "Negative"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("RESTAURANT SENTIMENT DATASET PREPARATION")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    # Read the original dataset without modifying it.
    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)

    required = {TEXT_COL, LABEL_COL}
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print(f"\nOriginal rows: {original_rows}")
    print(f"Original columns: {len(df.columns)}")

    # Remove rows without usable text or a sentiment label.
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].fillna("").astype(str).str.strip()

    valid_text = df[TEXT_COL].ne("")
    valid_label = df[LABEL_COL].ne("")

    invalid_rows = int((~(valid_text & valid_label)).sum())
    df = df.loc[valid_text & valid_label].copy()

    print(f"Rows removed for empty text/label: {invalid_rows}")

    # Reject unexpected labels rather than guessing their meaning.
    found_labels = set(df[LABEL_COL].unique())
    unexpected = found_labels - EXPECTED_LABELS
    missing_labels = EXPECTED_LABELS - found_labels

    if unexpected:
        raise ValueError(
            f"Unexpected sentiment labels: {sorted(unexpected)}"
        )

    if missing_labels:
        raise ValueError(
            f"Expected sentiment classes are missing: "
            f"{sorted(missing_labels)}"
        )

    # Validate the known star-to-sentiment mapping when available.
    if STARS_COL in df.columns:
        expected_by_stars = {
            1: "Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Positive",
        }

        stars = pd.to_numeric(df[STARS_COL], errors="coerce")
        expected_labels = stars.map(expected_by_stars)
        mismatch = (
            expected_labels.notna()
            & expected_labels.ne(df[LABEL_COL])
        )

        mismatch_count = int(mismatch.sum())

        print(f"Star/label inconsistencies: {mismatch_count}")

        if mismatch_count:
            examples = df.loc[
                mismatch, [STARS_COL, LABEL_COL, TEXT_COL]
            ].head(10)

            print(examples.to_string(index=False))

            raise ValueError(
                "Star ratings conflict with sentiment labels. "
                "Review the examples above before proceeding."
            )

    # Create a normalized key for duplicate detection only.
    # The original review wording is retained in the text column.
    df["_dedup_key"] = (
        df[TEXT_COL]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # The same normalized review must not have conflicting labels.
    labels_per_text = df.groupby("_dedup_key")[LABEL_COL].nunique()
    conflicting_keys = labels_per_text[labels_per_text > 1]

    if len(conflicting_keys):
        examples = df[
            df["_dedup_key"].isin(conflicting_keys.index)
        ][["_dedup_key", LABEL_COL]]

        print("\nExamples of duplicate text with conflicting labels:")
        print(examples.head(20).to_string(index=False))

        raise ValueError(
            "Identical normalized review text has conflicting labels. "
            "Resolve these cases before splitting."
        )

    before_dedup = len(df)
    df = df.drop_duplicates(subset="_dedup_key", keep="first").copy()
    duplicates_removed = before_dedup - len(df)

    print(f"Duplicate review texts removed: {duplicates_removed}")
    print(f"Clean rows: {len(df)}")

    # Save the cleaned dataset without the internal deduplication key.
    cleaned = df.drop(columns=["_dedup_key"])
    cleaned_path = OUTPUT_DIR / "cleaned_dataset.csv"
    cleaned.to_csv(cleaned_path, index=False)

    # First reserve 30%, then split that portion equally into
    # 15% validation and 15% final test.
    train_df, temporary_df = train_test_split(
        cleaned,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=cleaned[LABEL_COL],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_df[LABEL_COL],
    )

    splits = {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }

    # Confirm no normalized review text crosses split boundaries.
    def text_keys(frame):
        return set(
            frame[TEXT_COL]
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    train_keys = text_keys(train_df)
    validation_keys = text_keys(validation_df)
    test_keys = text_keys(test_df)

    assert train_keys.isdisjoint(validation_keys)
    assert train_keys.isdisjoint(test_keys)
    assert validation_keys.isdisjoint(test_keys)

    # Save each split and record its label distribution.
    summary = {
        "source_file": str(DATA_PATH),
        "original_rows": original_rows,
        "rows_removed_empty_text_or_label": invalid_rows,
        "duplicate_texts_removed": duplicates_removed,
        "clean_rows": len(cleaned),
        "random_state": RANDOM_STATE,
        "split_percentages": {
            "train": 70,
            "validation": 15,
            "test": 15,
        },
        "text_column": TEXT_COL,
        "label_column": LABEL_COL,
        "classes": sorted(EXPECTED_LABELS),
        "splits": {},
    }

    for filename, frame in splits.items():
        output_path = OUTPUT_DIR / filename
        frame.to_csv(output_path, index=False)

        split_name = filename.removesuffix(".csv")

        counts = (
            frame[LABEL_COL]
            .value_counts()
            .reindex(
                ["Negative", "Neutral", "Positive"],
                fill_value=0,
            )
        )

        summary["splits"][split_name] = {
            "rows": len(frame),
            "percentage": round(100 * len(frame) / len(cleaned), 2),
            "class_counts": {
                label: int(count)
                for label, count in counts.items()
            },
        }

        print(f"\n{split_name.upper()}: {len(frame)} rows")
        print(counts.to_string())

    summary_path = OUTPUT_DIR / "dataset_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUCCESSFUL")
    print("=" * 70)
    print(f"Cleaned dataset: {cleaned_path.resolve()}")
    print(f"Training split:  {(OUTPUT_DIR / 'train.csv').resolve()}")
    print(f"Validation split: {(OUTPUT_DIR / 'validation.csv').resolve()}")
    print(f"Final test split: {(OUTPUT_DIR / 'test.csv').resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print("\nNo original files were overwritten.")


if __name__ == "__main__":
    main()