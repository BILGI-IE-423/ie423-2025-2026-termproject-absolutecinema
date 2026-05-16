# Data Folder Instructions

Raw and processed CSVs are stored in this repo using **Git LFS** (large files).

After cloning, from the repository root run:
```bash
git lfs install
git lfs pull
```
You should then see full-size files at `data/raw/IMDB Dataset.csv` and `data/processed/cleaned_imdb_reviews.csv`.

**Fallback:** If LFS objects are missing or you prefer Kaggle, download **IMDB Dataset.csv** from [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in `data/raw/` with exactly that name. Regenerate processed data with `scripts/02_preprocess_data.py` if needed.

## File naming
- **Raw:** `data/raw/IMDB Dataset.csv`
- **Processed:** `data/processed/cleaned_imdb_reviews.csv` (included via LFS, or produced by the preprocessing script)
