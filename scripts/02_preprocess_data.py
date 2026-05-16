"""
Preprocess IMDB reviews: deduplicate, drop nulls, clean text, encode labels.
Saves cleaned CSV and a preprocessing summary table. Run from the repository root.
"""

import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove HTML tags, non-alphabetic characters, lower-case, and normalize
    whitespace.  This is the *aggressive* cleaning variant (used as the
    primary feature for bag-of-words / TF-IDF models).
    """
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_light(text: str) -> str:
    """
    Light cleaning: strips HTML but *keeps* punctuation and digits.
    Useful for models that can leverage exclamation marks, question marks,
    numbers (e.g., ratings mentioned in text), etc.
    This column supports Research Question 2 (effect of aggressive vs.
    light preprocessing on sentiment classification).
    """
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_data(
    input_path: str,
    output_path: str,
    tables_dir: Optional[str] = None,
) -> None:
    print("--- Preprocessing Data ---")

    if not os.path.exists(input_path):
        print(f"Raw data not found at {input_path}. Place IMDB Dataset.csv in data/raw/.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    raw_rows = int(df.shape[0])

    # --- Deduplication ---
    initial_shape = df.shape
    df = df.drop_duplicates(subset=["review"])
    duplicates_dropped = int(initial_shape[0] - df.shape[0])
    print(f"Dropped {duplicates_dropped} duplicate rows.")

    # --- Missing values ---
    nulls_before = int(df.isnull().any(axis=1).sum())
    df = df.dropna()
    nulls_dropped = nulls_before  # (rows that had any null)
    print(f"Dropped {nulls_dropped} rows with missing values.")

    # --- Text cleaning (two variants) ---
    print("Cleaning text (aggressive: removing HTML, punctuation, lowercasing)...")
    df["cleaned_review"] = df["review"].apply(clean_text)

    print("Cleaning text (light: removing HTML only, keeping punctuation)...")
    df["cleaned_review_light"] = df["review"].apply(clean_text_light)

    # --- Drop empty strings that may result from aggressive cleaning ---
    empty_mask = df["cleaned_review"].str.strip().eq("")
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        print(f"Warning: {empty_count} reviews became empty after cleaning — dropping them.")
        df = df[~empty_mask]

    # --- Label encoding ---
    df["sentiment_label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    if df["sentiment_label"].isnull().any():
        bad = df.loc[df["sentiment_label"].isnull(), "sentiment"].unique().tolist()
        raise ValueError(f"Unexpected sentiment labels: {bad}")

    # --- Derived features (useful for EDA and Research Question 1) ---
    df["word_count"] = df["cleaned_review"].apply(lambda x: len(x.split()))
    df["char_count"] = df["cleaned_review"].apply(len)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data successfully saved to {output_path}")

    # --- Summary table ---
    if tables_dir:
        os.makedirs(tables_dir, exist_ok=True)

        pos_count = int((df["sentiment_label"] == 1).sum())
        neg_count = int((df["sentiment_label"] == 0).sum())

        summary = pd.DataFrame(
            [
                {"metric": "raw_rows", "value": raw_rows},
                {"metric": "duplicate_rows_dropped", "value": duplicates_dropped},
                {"metric": "null_rows_dropped", "value": nulls_dropped},
                {"metric": "empty_after_cleaning_dropped", "value": empty_count},
                {"metric": "processed_rows", "value": int(df.shape[0])},
                {"metric": "processed_columns", "value": int(df.shape[1])},
                {"metric": "positive_count", "value": pos_count},
                {"metric": "negative_count", "value": neg_count},
                {"metric": "class_balance_ratio", "value": round(pos_count / max(neg_count, 1), 4)},
                {"metric": "mean_word_count", "value": round(float(df["word_count"].mean()), 2)},
                {"metric": "median_word_count", "value": round(float(df["word_count"].median()), 2)},
                {"metric": "std_word_count", "value": round(float(df["word_count"].std()), 2)},
                {"metric": "min_word_count", "value": int(df["word_count"].min())},
                {"metric": "max_word_count", "value": int(df["word_count"].max())},
                {"metric": "mean_char_count", "value": round(float(df["char_count"].mean()), 2)},
                {"metric": "median_char_count", "value": round(float(df["char_count"].median()), 2)},
            ]
        )
        out_path = os.path.join(tables_dir, "02_preprocess_summary.csv")
        summary.to_csv(out_path, index=False)
        print(f"Saved table: {out_path}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    input_file = os.path.join("data", "raw", "IMDB Dataset.csv")
    output_file = os.path.join("data", "processed", "cleaned_imdb_reviews.csv")
    tables_dir = os.path.join("outputs", "tables")
    preprocess_data(input_file, output_file, tables_dir=tables_dir)
