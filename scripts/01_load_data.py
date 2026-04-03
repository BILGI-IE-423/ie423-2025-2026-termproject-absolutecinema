"""
Load the raw IMDB reviews CSV, print basic diagnostics, and save a load summary table.
Run from the repository root.
"""

import os
import sys
from typing import Optional

import pandas as pd


def load_and_inspect_data(file_path: str, tables_dir: Optional[str] = None) -> pd.DataFrame:
    print(f"--- Loading Data from {file_path} ---")

    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        print("Please download the dataset and place it in the 'data/raw/' directory.")
        sys.exit(1)

    df = pd.read_csv(file_path)

    print(f"Dataset Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nFirst 3 rows:\n", df.head(3))

    if tables_dir:
        os.makedirs(tables_dir, exist_ok=True)
        missing = df.isnull().sum()
        summary_rows = [
            {"metric": "rows", "value": int(df.shape[0])},
            {"metric": "columns", "value": int(df.shape[1])},
        ]
        for col in df.columns:
            summary_rows.append(
                {"metric": f"missing__{col}", "value": int(missing[col])}
            )
        summary_df = pd.DataFrame(summary_rows)
        out_path = os.path.join(tables_dir, "01_load_summary.csv")
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved table: {out_path}")

    return df


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    raw_data_path = os.path.join("data", "raw", "IMDB Dataset.csv")
    tables_dir = os.path.join("outputs", "tables")
    load_and_inspect_data(raw_data_path, tables_dir=tables_dir)
