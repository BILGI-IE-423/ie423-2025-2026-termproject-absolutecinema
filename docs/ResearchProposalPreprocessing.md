# IE 423 Project Proposal — Sentiment Analysis on IMDB Movie Reviews

## Team Information

**Team Name:** AbsoluteCinema

- Semih Yavuz
- Mert Ata Tekçe
- Çağan Göktaş
- Emir Türkseven

## Dataset Description

This project uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) obtained from Kaggle.com (50,000 IMDB movie reviews).  
This dataset has 50,000 movie reviews with positive and negative labels for Natural Language Processing (NLP) and text analysis.
The reason for choosing this dataset: it is suitable for binary sentiment classification; it also allows us to explore patterns such as sarcasm, strong emotions expressed with punctuation, etc.

## Accessing the Dataset

The raw dataset may not always be accessible from the repository because of file size or licensing issues. It should be placed at this filepath after downloading:

`data/raw/IMDB Dataset.csv`

For more info take a look at: `data/README.md`.

## Research Questions

### Research Question 1: Review Length and Model Behaviour

How is the review length related to correctly classifying the sentiment of the review? How reliable is the distinction based on text length when positive and negative reviews have a similar word-count distribution?

**Explanation:**  
`outputs/figures/sentiment_distribution.png` shows that after preprocessing, the classes are **almost balanced** (the numerical summary is visible at `outputs/tables/03_eda_summary.csv`; negative: 24,698, positive: 24,884, class balance ratio ≈ 1.0075). This confirms no dominance between the two groups and therefore no reason for oversampling/undersampling. On the other hand, `outputs/figures/review_length_distribution.png` shows that the word-count distributions for both groups are **quite similar** and **right-skewed**, with a density peak around **100–200 words**; most reviews consist of **50–300 words**, and even though it is rare, the tail extends up to **1,000+ words**. The boxplot (`outputs/figures/word_count_boxplot.png`) confirms that the median and interquartile range are nearly identical across both classes.

These findings suggest that sentiment is more likely to be driven by **which words or contexts** are used rather than how many words are in the review. Text length alone cannot serve as the **main predictor**, but it may be a useful auxiliary feature. We plan to categorize reviews into length buckets (short: 0–100, medium: 101–300, long: 300+ words) and perform error analysis per bucket to see where our model struggles the most.

| Statistic | Overall | Positive | Negative |
|-----------|---------|----------|----------|
| Count | 49,582 | 24,884 | 24,698 |
| Mean word count | 225.39 | 227.36 | 223.40 |
| Median word count | 169 | 168 | 170 |
| Std word count | 167.13 | 173.38 | 160.57 |
| Min word count | 4 | — | — |
| Max word count | 2,441 | — | — |
| Q1 word count | 123 | — | — |
| Q3 word count | 273 | — | — |
| Vocabulary size | 214,623 unique tokens | — | — |
| Peak range | 101–200 words (47.2% of all reviews) | — | — |

#### Review Length Bucket Distribution

| Bucket | Count | Percentage |
|--------|-------|------------|
| 0–50 words | 1,344 | 2.71% |
| 51–100 words | 5,252 | 10.59% |
| 101–200 words | 23,408 | 47.21% |
| 201–300 words | 9,054 | 18.26% |
| 301–500 words | 6,874 | 13.86% |
| 500+ words | 3,650 | 7.36% |

### Research Question 2: The Effect of Preprocessing Intensity on Interpretability

What is the impact of removing stop-words (e.g., "the", "a"), deleting punctuation, and removing numbers on sentiment classification accuracy?

**Explanation:**  
Standard NLP preprocessing pipelines usually remove punctuation and filler words. However, these elements can carry sentimental signal in movie reviews. For example, repeated exclamation marks (`!!!`) or question marks convey strong emotion; negation particles like "not" are stop-words but radically change meaning. This research question aims to determine whether aggressive cleaning disrupts the naturalness of real-world text and degrades model performance.

**Methodology:** Our preprocessing pipeline generates two cleaned versions of each review:
1. `cleaned_review` — aggressive cleaning (HTML removal + punctuation/number removal + lowercasing)
2. `cleaned_review_light` — light cleaning (HTML removal + lowercasing only)

We will train the same models (Logistic Regression, Naive Bayes) on TF-IDF features extracted from each version and compare their F1-scores, precision, and recall to measure the information loss.

### Research Question 3: Linguistic Patterns in Misclassifications

In cases where the baseline model systematically misclassifies reviews, are there common linguistic patterns?

**Explanation:**  
The same word may carry different sentiment in different movie genres ("scary" can be a positive signal for a horror movie but negative for a romantic comedy). If the model fails to distinguish between these scenarios, it may repeatedly fail on certain words or phrases. We aim to identify those mistakes through error analysis in order to understand where the model struggles the most. This will enable us to have a more accurate interpretation of aggregate metrics (F1, ROC-AUC, etc.).

**Methodology:**
1. Train the baseline model and collect all misclassified reviews from the test set.
2. Compare the most frequent n-grams in correctly vs. incorrectly classified reviews.
3. Analyze whether certain review-length buckets or vocabulary patterns are over-represented among errors.

## Project Proposal

In this project we aim to perform binary sentiment analysis on movie reviews using machine learning models.

### What has been completed

- **Data Loading & Inspection:** Loaded the raw dataset (50,000 rows × 2 columns), verified zero missing values, and generated a load summary table (`scripts/01_load_data.py`).
- **Preprocessing:** Removed 418 duplicate reviews, applied two-level text cleaning (aggressive + light), computed derived features (word count, character count), encoded sentiment labels, and generated a comprehensive summary with 16 metrics (`scripts/02_preprocess_data.py`).
- **Exploratory Data Analysis:** Generated 7 visualizations including sentiment distribution, word-count histograms, character-count distributions, boxplots, word clouds, and top-25 frequent words per class. Produced detailed summary statistics with per-class breakdowns and length-bucket analysis (`scripts/03_basic_eda.py`).

### Key Preprocessing Statistics

| Metric | Value |
|--------|-------|
| Raw rows | 50,000 |
| Duplicates dropped | 418 |
| Final processed rows | 49,582 |
| Positive reviews | 24,884 |
| Negative reviews | 24,698 |
| Class balance ratio | 1.0075 |
| Mean word count | 225.39 |
| Median word count | 169 |
| Columns in processed dataset | review, sentiment, cleaned_review, cleaned_review_light, sentiment_label, word_count, char_count |

### What will be done next

1. **Feature Extraction**
   - TF-IDF vectorization with `max_features=10,000`, `ngram_range=(1,2)`, `min_df=5`, `max_df=0.95`
   - Apply to both `cleaned_review` (aggressive) and `cleaned_review_light` columns separately

2. **Train/Test Split**
   - 80/20 stratified split (`random_state=42`) to preserve class proportions
   - Training set: ~39,666 reviews; Test set: ~9,916 reviews

3. **Modeling**
   - **Baseline:** Logistic Regression (L2 regularization, `C=1.0`)
   - **Alternative:** Multinomial Naive Bayes (`alpha=1.0` Laplace smoothing)
   - **Stretch goal:** Linear SVM (`LinearSVC`) if time permits

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score (macro & per-class), ROC-AUC
   - Confusion matrix visualization
   - ROC curve comparison across models
   - Cross-validation: 5-fold stratified CV on the training set

5. **Error Analysis** (Research Question 3)
   - Collect all false positives and false negatives
   - Analyze word-count distribution of misclassified vs. correctly classified reviews
   - Extract top distinguishing n-grams from errors
   - Length-bucket breakdown of error rates

6. **Preprocessing Comparison** (Research Question 2)
   - Re-run the full pipeline with light-cleaned text
   - Compare metrics side-by-side in a summary table

### Timeline

| Week | Task |
|------|------|
| Week 1 | Feature extraction (TF-IDF), train/test split |
| Week 2 | Logistic Regression + Naive Bayes training & evaluation |
| Week 3 | Error analysis, preprocessing comparison experiment |
| Week 4 | Final report, visualizations, and documentation |

### Potential Challenges

- **Sarcastic Reviews:** Sarcasm inverts the literal meaning; a bag-of-words model may struggle. We will flag potential sarcasm cases in the error analysis.
- **Efficient Use of Resources:** TF-IDF with large vocabulary can be memory-intensive. We will cap `max_features` at 10,000 to keep memory usage reasonable.
- **Class Overlap in Language:** Positive and negative reviews share many common words (as shown in the top-N analysis). Discriminative power may come from relatively rare terms.

## Preprocessing Steps

### Step 1 — Loading Data

`scripts/01_load_data.py` reads the raw CSV, prints the shape and missing summary to the console, and saves the summary table into `outputs/tables/01_load_summary.csv`.

### Step 2 — Inspection and Cleaning

With `scripts/02_preprocess_data.py`:

- Duplicate rows are dropped based on the `review` field (numerical summary is in `outputs/tables/02_preprocess_summary.csv`).
- Rows with missing values are removed (no missing values were observed in the raw stage of this dataset; summary can be traced via `01_load_summary.csv`).
- **Aggressive cleaning** (`cleaned_review`): HTML tags (e.g., `<br />`) are removed using BeautifulSoup; non-alphabetic and non-space characters are removed with regular expressions; text is converted to lowercase and extra spaces are simplified.
- **Light cleaning** (`cleaned_review_light`): Only HTML tags are removed and text is lowercased — punctuation and numbers are preserved. This supports the Research Question 2 comparison.
- Reviews that become empty after aggressive cleaning are detected and dropped.
- `sentiment` labels are converted to numeric (`sentiment_label`: negative → 0, positive → 1).
- Derived features `word_count` and `char_count` are computed for EDA and modeling.

### Step 3 — Saving Processed Data

The clean data is written to:  
`data/processed/cleaned_imdb_reviews.csv`  
(Columns: `review`, `sentiment`, `cleaned_review`, `cleaned_review_light`, `sentiment_label`, `word_count`, `char_count`).

---

## Initial Outputs

### Raw and Processed Data Summary

- Raw data: **50,000** rows, **2** columns; missing values: **0** (verified via `outputs/tables/01_load_summary.csv`).
- After deleting **418** duplicate rows: **49,582** rows, **7** columns remain (verified via `outputs/tables/02_preprocess_summary.csv`).

These numbers are designed to be verified through the tables written by the scripts, not just as static text (transparency and reproducibility).

### Visualizations

The following visualizations are produced by `scripts/03_basic_eda.py`:

**Sentiment distribution:** The bars are almost the same height; even after deleting duplicate rows, classes are approximately balanced (exact numbers are in `outputs/tables/03_eda_summary.csv`). Therefore, aggressive class balancing is not necessary.

**Word-count distribution:** Positive and negative curves overlap; the peak is around 100–200 words, the distribution is right-skewed, and most reviews are in the 50–300 word band.

**Word-count boxplot:** Confirms nearly identical median and IQR for both classes, with outliers extending beyond 1,000 words.

**Character-count distribution:** Mirrors the word-count pattern, confirming reviews are consistently similar in length across classes.

**Top-25 frequent words:** Shows high overlap in the most common words between positive and negative reviews (e.g., "the", "and", "of"), suggesting that discriminative power comes from less frequent, more sentiment-specific terms.

**Word clouds:** Visual summary of the most prominent words for each class.

![Sentiment distribution](../outputs/figures/sentiment_distribution.png)

![Review length distribution](../outputs/figures/review_length_distribution.png)

![Word count boxplot](../outputs/figures/word_count_boxplot.png)

![Top words by sentiment](../outputs/figures/top_words_by_sentiment.png)

![Word cloud — positive](../outputs/figures/wordcloud_positive.png)

![Word cloud — negative](../outputs/figures/wordcloud_negative.png)

Numeric EDA summary: `outputs/tables/03_eda_summary.csv`

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-absolutecinema.git
cd ie423-2025-2026-termproject-absolutecinema
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Place the data file

Put `IMDB Dataset.csv` in `data/raw/` (see `data/README.md`).

### 4. Run the scripts in order

```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```

Expected outputs: `data/processed/cleaned_imdb_reviews.csv`, `outputs/figures/*.png`, `outputs/tables/*.csv`.

## Transparency and Traceability

The figures in this document are located in `outputs/figures/`. The summary tables are stored in `outputs/tables/`. All of them are produced by the Python scripts in `scripts/`:

| Output | Script |
|--------|--------|
| `outputs/tables/01_load_summary.csv` | `scripts/01_load_data.py` |
| `data/processed/cleaned_imdb_reviews.csv`, `outputs/tables/02_preprocess_summary.csv` | `scripts/02_preprocess_data.py` |
| `outputs/figures/*.png`, `outputs/tables/03_eda_summary.csv`, `outputs/tables/03_top_words_*.csv` | `scripts/03_basic_eda.py` |

Someone else can install the packages, use the same raw data, and run the scripts in this order to reproduce the same outputs.
