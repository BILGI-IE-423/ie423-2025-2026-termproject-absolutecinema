"""
Preprocess IMDB reviews: deduplicate, drop nulls, clean text, encode labels.
Saves cleaned CSV and a preprocessing summary table. Run from the repository root.
"""

import os
import re
import sys
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Remove HTML tags, non-alphabetic characters, lower-case, and normalize whitespace.
    """
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

    initial_shape = df.shape
    df = df.drop_duplicates(subset=["review"])
    duplicates_dropped = int(initial_shape[0] - df.shape[0])
    print(f"Dropped {duplicates_dropped} duplicate rows.")

    df = df.dropna()

    print("Cleaning text (removing HTML, punctuation, and lowercasing)...")
    df["cleaned_review"] = df["review"].apply(clean_text)

    df["sentiment_label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    if df["sentiment_label"].isnull().any():
        bad = df.loc[df["sentiment_label"].isnull(), "sentiment"].unique().tolist()
        raise ValueError(f"Unexpected sentiment labels: {bad}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data successfully saved to {output_path}")

    if tables_dir:
        os.makedirs(tables_dir, exist_ok=True)
        summary = pd.DataFrame(
            [
                {"metric": "raw_rows", "value": raw_rows},
                {"metric": "duplicate_rows_dropped", "value": duplicates_dropped},
                {"metric": "processed_rows", "value": int(df.shape[0])},
                {"metric": "processed_columns", "value": int(df.shape[1])},
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
