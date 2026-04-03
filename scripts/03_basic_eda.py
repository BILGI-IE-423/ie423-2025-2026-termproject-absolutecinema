"""
EDA on processed IMDB data: sentiment counts, review length distribution plots.
Saves figures and a small summary table. Run from the repository root.
"""

import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def perform_eda(
    input_path: str,
    figures_dir: str,
    tables_dir: Optional[str] = None,
) -> None:
    print("--- Performing Exploratory Data Analysis ---")

    if not os.path.exists(input_path):
        print(f"Processed data not found at {input_path}. Run 02_preprocess_data.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Sentiment distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment", hue="sentiment", palette="Set2", legend=False)
    plt.title("Distribution of Sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    dist_plot_path = os.path.join(figures_dir, "sentiment_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"Saved figure: {dist_plot_path}")

    # 2. Review word counts by sentiment
    df["review_length"] = df["cleaned_review"].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="review_length",
        hue="sentiment",
        bins=50,
        kde=True,
        palette="Set2",
        element="step",
    )
    plt.title("Review Word Count Distribution by Sentiment")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.xlim(0, 1000)
    length_plot_path = os.path.join(figures_dir, "review_length_distribution.png")
    plt.tight_layout()
    plt.savefig(length_plot_path, dpi=150)
    plt.close()
    print(f"Saved figure: {length_plot_path}")

    if tables_dir:
        os.makedirs(tables_dir, exist_ok=True)
        counts = df["sentiment"].value_counts().sort_index()
        stats_rows = [{"metric": f"count__{idx}", "value": int(val)} for idx, val in counts.items()]
        stats_rows.extend(
            [
                {
                    "metric": "mean_review_length_words",
                    "value": float(df["review_length"].mean()),
                },
                {
                    "metric": "median_review_length_words",
                    "value": float(df["review_length"].median()),
                },
            ]
        )
        pd.DataFrame(stats_rows).to_csv(
            os.path.join(tables_dir, "03_eda_summary.csv"),
            index=False,
        )
        print(f"Saved table: {os.path.join(tables_dir, '03_eda_summary.csv')}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    processed_file = os.path.join("data", "processed", "cleaned_imdb_reviews.csv")
    figures_folder = os.path.join("outputs", "figures")
    tables_dir = os.path.join("outputs", "tables")
    perform_eda(processed_file, figures_folder, tables_dir=tables_dir)
