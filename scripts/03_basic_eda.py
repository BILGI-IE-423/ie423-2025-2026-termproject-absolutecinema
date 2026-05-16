"""
Advanced EDA on processed IMDB data.
Produces 15+ publication-quality figures and detailed summary tables.
Run from the repository root.
"""

import os
import re
import sys
from collections import Counter
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except ImportError:
    _HAS_WORDCLOUD = False

try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw
    STOPWORDS: set = set(_sw.words("english"))
except ImportError:
    STOPWORDS = {
        "the","a","an","and","or","but","if","in","on","at","to","for","of",
        "with","by","is","was","are","were","be","been","am","do","does","did",
        "has","have","had","it","its","i","he","she","we","they","you","me",
        "him","her","us","them","my","his","our","your","their","this","that",
        "these","those","not","no","so","as","from","will","would","can","could",
        "just","than","then","also","very","too","more","most","what","which",
        "who","whom","how","when","where","why","all","each","every","both",
        "few","many","some","any","about","up","out","into","over","after",
        "before","between","there","here","only","own","same","other",
    }

# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})
TOP_N = 25
PAL = ["#66c2a5", "#fc8d62"]

def _save(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

def _ngrams(texts, n=2, top_k=20, remove_sw=True):
    counter = Counter()
    for t in texts:
        words = str(t).split()
        if remove_sw:
            words = [w for w in words if w not in STOPWORDS]
        for i in range(len(words) - n + 1):
            counter[" ".join(words[i:i+n])] += 1
    return pd.DataFrame(counter.most_common(top_k), columns=["ngram", "count"])

def _word_freq(series, top_n=TOP_N, remove_sw=True):
    counter = Counter()
    for t in series:
        words = str(t).split()
        if remove_sw:
            words = [w for w in words if w not in STOPWORDS]
        counter.update(words)
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])

# ---------------------------------------------------------------------------
def perform_eda(input_path, figures_dir, tables_dir=None):
    print("=" * 60)
    print("  ADVANCED EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run 02_preprocess_data.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    os.makedirs(figures_dir, exist_ok=True)
    if tables_dir:
        os.makedirs(tables_dir, exist_ok=True)

    # Ensure columns
    if "word_count" not in df.columns:
        df["word_count"] = df["cleaned_review"].apply(lambda x: len(str(x).split()))
    if "char_count" not in df.columns:
        df["char_count"] = df["cleaned_review"].apply(len)

    pos = df[df["sentiment"] == "positive"]
    neg = df[df["sentiment"] == "negative"]
    pos_text = pos["cleaned_review"]
    neg_text = neg["cleaned_review"]

    step = 0

    # ===== 1. Sentiment Distribution =====
    step += 1; print(f"\n[{step}] Sentiment distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["sentiment"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=PAL, edgecolor="white", width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+200,
                f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count"); ax.set_ylim(0, counts.max()*1.15)
    _save(os.path.join(figures_dir, "sentiment_distribution.png"))

    # ===== 2. Word-count histogram =====
    step += 1; print(f"[{step}] Word-count distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="word_count", hue="sentiment", bins=60,
                 kde=True, palette=PAL, element="step", ax=ax)
    ax.axvline(df["word_count"].mean(), color="red", ls="--", lw=1.2, label=f'Mean={df["word_count"].mean():.0f}')
    ax.axvline(df["word_count"].median(), color="blue", ls="--", lw=1.2, label=f'Median={df["word_count"].median():.0f}')
    ax.legend(); ax.set_xlim(0, 1000)
    ax.set_title("Word-Count Distribution by Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Words"); ax.set_ylabel("Frequency")
    _save(os.path.join(figures_dir, "review_length_distribution.png"))

    # ===== 3. Violin plot =====
    step += 1; print(f"[{step}] Violin plot")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="sentiment", y="word_count", hue="sentiment",
                   palette=PAL, inner="quartile", legend=False, ax=ax)
    ax.set_title("Word Count Distribution — Violin Plot", fontsize=14, fontweight="bold")
    ax.set_ylabel("Word Count"); ax.set_ylim(0, 800)
    _save(os.path.join(figures_dir, "word_count_violin.png"))

    # ===== 4. Boxplot =====
    step += 1; print(f"[{step}] Boxplot")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="sentiment", y="word_count", hue="sentiment",
                palette=PAL, legend=False, fliersize=1.5, ax=ax)
    ax.set_title("Word Count by Sentiment — Boxplot", fontsize=14, fontweight="bold")
    ax.set_ylabel("Word Count")
    _save(os.path.join(figures_dir, "word_count_boxplot.png"))

    # ===== 5. Character-count distribution =====
    step += 1; print(f"[{step}] Character-count distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="char_count", hue="sentiment", bins=60,
                 kde=True, palette=PAL, element="step", ax=ax)
    ax.set_xlim(0, df["char_count"].quantile(0.99))
    ax.set_title("Character-Count Distribution by Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Characters"); ax.set_ylabel("Frequency")
    _save(os.path.join(figures_dir, "char_count_distribution.png"))

    # ===== 6. Average word length =====
    step += 1; print(f"[{step}] Average word length")
    df["avg_word_len"] = df["cleaned_review"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).strip() else 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="avg_word_len", hue="sentiment", bins=50,
                 kde=True, palette=PAL, element="step", ax=ax)
    ax.set_title("Average Word Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Word Length (chars)"); ax.set_ylabel("Frequency")
    _save(os.path.join(figures_dir, "avg_word_length.png"))

    # ===== 7. Top-N unigrams =====
    step += 1; print(f"[{step}] Top-{TOP_N} unigrams (stopwords removed)")
    top_pos = _word_freq(pos_text)
    top_neg = _word_freq(neg_text)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].barh(top_pos["word"][::-1], top_pos["count"][::-1], color=PAL[0])
    axes[0].set_title(f"Top {TOP_N} Words — Positive", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Frequency")
    axes[1].barh(top_neg["word"][::-1], top_neg["count"][::-1], color=PAL[1])
    axes[1].set_title(f"Top {TOP_N} Words — Negative", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frequency")
    _save(os.path.join(figures_dir, "top_words_by_sentiment.png"))

    # ===== 8. Top bigrams =====
    step += 1; print(f"[{step}] Top-20 bigrams")
    bi_pos = _ngrams(pos_text, n=2, top_k=20)
    bi_neg = _ngrams(neg_text, n=2, top_k=20)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].barh(bi_pos["ngram"][::-1], bi_pos["count"][::-1], color=PAL[0])
    axes[0].set_title("Top 20 Bigrams — Positive", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Frequency")
    axes[1].barh(bi_neg["ngram"][::-1], bi_neg["count"][::-1], color=PAL[1])
    axes[1].set_title("Top 20 Bigrams — Negative", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frequency")
    _save(os.path.join(figures_dir, "top_bigrams_by_sentiment.png"))

    # ===== 9. Top trigrams =====
    step += 1; print(f"[{step}] Top-15 trigrams")
    tri_pos = _ngrams(pos_text, n=3, top_k=15)
    tri_neg = _ngrams(neg_text, n=3, top_k=15)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(tri_pos["ngram"][::-1], tri_pos["count"][::-1], color=PAL[0])
    axes[0].set_title("Top 15 Trigrams — Positive", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Frequency")
    axes[1].barh(tri_neg["ngram"][::-1], tri_neg["count"][::-1], color=PAL[1])
    axes[1].set_title("Top 15 Trigrams — Negative", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frequency")
    _save(os.path.join(figures_dir, "top_trigrams_by_sentiment.png"))

    # ===== 10. TF-IDF most discriminative words =====
    step += 1; print(f"[{step}] TF-IDF discriminative words")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,1))
    X = tfidf.fit_transform(df["cleaned_review"].astype(str))
    feature_names = np.array(tfidf.get_feature_names_out())
    pos_idx = (df["sentiment"] == "positive").values
    neg_idx = (df["sentiment"] == "negative").values
    mean_pos = np.array(X[pos_idx].mean(axis=0)).flatten()
    mean_neg = np.array(X[neg_idx].mean(axis=0)).flatten()
    diff = mean_pos - mean_neg
    top_pos_idx = diff.argsort()[-20:][::-1]
    top_neg_idx = diff.argsort()[:20]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].barh(feature_names[top_pos_idx][::-1], diff[top_pos_idx][::-1], color=PAL[0])
    axes[0].set_title("Most Discriminative — Positive\n(TF-IDF mean difference)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("TF-IDF Score Difference (pos − neg)")
    axes[1].barh(feature_names[top_neg_idx][::-1], np.abs(diff[top_neg_idx])[::-1], color=PAL[1])
    axes[1].set_title("Most Discriminative — Negative\n(TF-IDF mean difference)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("|TF-IDF Score Difference| (neg − pos)")
    _save(os.path.join(figures_dir, "tfidf_discriminative_words.png"))

    # ===== 11. Log-odds ratio =====
    step += 1; print(f"[{step}] Log-odds ratio")
    all_words_pos = Counter()
    all_words_neg = Counter()
    for t in pos_text: all_words_pos.update([w for w in str(t).split() if w not in STOPWORDS])
    for t in neg_text: all_words_neg.update([w for w in str(t).split() if w not in STOPWORDS])
    total_pos = sum(all_words_pos.values())
    total_neg = sum(all_words_neg.values())
    common_words = set(all_words_pos.keys()) & set(all_words_neg.keys())
    common_words = [w for w in common_words if all_words_pos[w]+all_words_neg[w] >= 100]
    log_odds = {}
    for w in common_words:
        p_pos = (all_words_pos[w]+1) / (total_pos+len(common_words))
        p_neg = (all_words_neg[w]+1) / (total_neg+len(common_words))
        log_odds[w] = np.log2(p_pos / p_neg)
    lo_df = pd.DataFrame(list(log_odds.items()), columns=["word","log_odds"]).sort_values("log_odds")
    top_neg_lo = lo_df.head(15)
    top_pos_lo = lo_df.tail(15)
    combined = pd.concat([top_neg_lo, top_pos_lo])
    colors_lo = [PAL[1] if v < 0 else PAL[0] for v in combined["log_odds"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(combined["word"], combined["log_odds"], color=colors_lo)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Log-Odds Ratio: Most Sentiment-Specific Words\n(positive → right, negative → left)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Log₂ Odds Ratio")
    _save(os.path.join(figures_dir, "log_odds_ratio.png"))

    # ===== 12. Word clouds =====
    if _HAS_WORDCLOUD:
        step += 1; print(f"[{step}] Word clouds")
        for label, series in [("positive", pos_text), ("negative", neg_text)]:
            wc = WordCloud(width=1000, height=500, max_words=200, background_color="white",
                           colormap="viridis", stopwords=STOPWORDS, random_state=RANDOM_SEED
                           ).generate(" ".join(series.astype(str)))
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            ax.set_title(f"Word Cloud — {label.capitalize()} Reviews", fontsize=15, fontweight="bold")
            _save(os.path.join(figures_dir, f"wordcloud_{label}.png"))
    else:
        step += 1; print(f"[{step}] Skipped word clouds")

    # ===== 13. Vocabulary richness (TTR) =====
    step += 1; print(f"[{step}] Vocabulary richness (Type-Token Ratio)")
    df["unique_words"] = df["cleaned_review"].apply(lambda x: len(set(str(x).split())))
    df["ttr"] = df["unique_words"] / df["word_count"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="ttr", hue="sentiment", bins=50,
                 kde=True, palette=PAL, element="step", ax=ax)
    ax.set_title("Type-Token Ratio (Vocabulary Richness) by Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("TTR (unique words / total words)"); ax.set_ylabel("Frequency")
    _save(os.path.join(figures_dir, "type_token_ratio.png"))

    # ===== 14. Zipf's Law =====
    step += 1; print(f"[{step}] Zipf's law")
    all_counter = Counter()
    for t in df["cleaned_review"]: all_counter.update(str(t).split())
    freqs = sorted(all_counter.values(), reverse=True)
    ranks = np.arange(1, len(freqs)+1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ranks, freqs, ".", markersize=2, alpha=0.5, color="#5e4fa2")
    ax.loglog(ranks, freqs[0]/ranks, "--", color="red", alpha=0.7, label="Ideal Zipf (1/rank)")
    ax.set_title("Zipf's Law — Word Frequency vs. Rank", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rank (log)"); ax.set_ylabel("Frequency (log)")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(os.path.join(figures_dir, "zipfs_law.png"))

    # ===== 15. Punctuation analysis (from light-cleaned text) =====
    step += 1; print(f"[{step}] Punctuation analysis")
    if "cleaned_review_light" in df.columns:
        src = df["cleaned_review_light"]
    else:
        src = df["review"] if "review" in df.columns else df["cleaned_review"]
    df["excl_count"] = src.apply(lambda x: str(x).count("!"))
    df["quest_count"] = src.apply(lambda x: str(x).count("?"))
    df["caps_ratio"] = src.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)),1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x="sentiment", y="excl_count", hue="sentiment",
                palette=PAL, legend=False, fliersize=1, ax=axes[0])
    axes[0].set_title("Exclamation Marks (!)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count per review"); axes[0].set_ylim(0, df["excl_count"].quantile(0.95))
    sns.boxplot(data=df, x="sentiment", y="quest_count", hue="sentiment",
                palette=PAL, legend=False, fliersize=1, ax=axes[1])
    axes[1].set_title("Question Marks (?)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count per review"); axes[1].set_ylim(0, df["quest_count"].quantile(0.95))
    sns.boxplot(data=df, x="sentiment", y="caps_ratio", hue="sentiment",
                palette=PAL, legend=False, fliersize=1, ax=axes[2])
    axes[2].set_title("Uppercase Ratio", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Ratio")
    _save(os.path.join(figures_dir, "punctuation_analysis.png"))

    # ===== 16. Feature correlation heatmap =====
    step += 1; print(f"[{step}] Feature correlation heatmap")
    num_cols = ["word_count","char_count","avg_word_len","unique_words","ttr",
                "excl_count","quest_count","caps_ratio","sentiment_label"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    _save(os.path.join(figures_dir, "correlation_heatmap.png"))

    # ===== 17. Length bucket class distribution =====
    step += 1; print(f"[{step}] Length bucket analysis")
    bins = [0, 50, 100, 200, 300, 500, float("inf")]
    labels_b = ["0-50", "51-100", "101-200", "201-300", "301-500", "500+"]
    df["length_bucket"] = pd.cut(df["word_count"], bins=bins, labels=labels_b, right=True)
    ct = pd.crosstab(df["length_bucket"], df["sentiment"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", stacked=True, color=PAL, edgecolor="white", ax=ax)
    ax.set_title("Sentiment Proportion by Review Length Bucket", fontsize=14, fontweight="bold")
    ax.set_xlabel("Word Count Bucket"); ax.set_ylabel("Percentage (%)")
    ax.legend(title="Sentiment"); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=9)
    _save(os.path.join(figures_dir, "length_bucket_sentiment.png"))

    # ===== 18. CDF of review lengths =====
    step += 1; print(f"[{step}] Cumulative distribution (ECDF)")
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in zip(["positive","negative"], PAL):
        subset = df[df["sentiment"]==label]["word_count"].sort_values()
        cdf = np.arange(1, len(subset)+1) / len(subset)
        ax.plot(subset, cdf, color=color, label=label, lw=1.5)
    ax.set_title("Empirical CDF of Review Word Counts", fontsize=14, fontweight="bold")
    ax.set_xlabel("Word Count"); ax.set_ylabel("Cumulative Probability")
    ax.legend(); ax.set_xlim(0, 800); ax.grid(True, alpha=0.3)
    _save(os.path.join(figures_dir, "ecdf_word_count.png"))

    # ===== SUMMARY TABLES =====
    print(f"\n[TABLES] Writing summary tables...")
    if tables_dir:
        # --- Main summary ---
        counts_s = df["sentiment"].value_counts().sort_index()
        rows = [{"metric":f"count__{i}","value":int(v)} for i,v in counts_s.items()]
        for col in ["word_count","char_count","avg_word_len","ttr","excl_count","quest_count","caps_ratio"]:
            desc = df[col].describe()
            for stat in ["mean","std","min","25%","50%","75%","max"]:
                rows.append({"metric":f"{stat}_{col}","value":round(float(desc[stat]),4)})
        for label in ["positive","negative"]:
            sub = df[df["sentiment"]==label]
            for col in ["word_count","char_count"]:
                desc = sub[col].describe()
                rows.append({"metric":f"mean_{col}__{label}","value":round(float(desc["mean"]),2)})
                rows.append({"metric":f"std_{col}__{label}","value":round(float(desc["std"]),2)})
                rows.append({"metric":f"median_{col}__{label}","value":round(float(desc["50%"]),2)})
        vocab_all = set(); vocab_pos = set(); vocab_neg = set()
        for t in df["cleaned_review"]: vocab_all.update(str(t).split())
        for t in pos_text: vocab_pos.update(str(t).split())
        for t in neg_text: vocab_neg.update(str(t).split())
        rows.append({"metric":"vocabulary_size_total","value":len(vocab_all)})
        rows.append({"metric":"vocabulary_size_positive","value":len(vocab_pos)})
        rows.append({"metric":"vocabulary_size_negative","value":len(vocab_neg)})
        rows.append({"metric":"vocabulary_overlap","value":len(vocab_pos&vocab_neg)})
        jaccard = len(vocab_pos&vocab_neg)/len(vocab_pos|vocab_neg)
        rows.append({"metric":"vocabulary_jaccard_similarity","value":round(jaccard,4)})
        for bucket in labels_b:
            n = int((df["length_bucket"]==bucket).sum())
            rows.append({"metric":f"bucket_{bucket}_count","value":n})
            rows.append({"metric":f"bucket_{bucket}_pct","value":round(100*n/len(df),2)})
        pd.DataFrame(rows).to_csv(os.path.join(tables_dir,"03_eda_summary.csv"), index=False)
        top_pos.to_csv(os.path.join(tables_dir,"03_top_words_positive.csv"), index=False)
        top_neg.to_csv(os.path.join(tables_dir,"03_top_words_negative.csv"), index=False)
        bi_pos.to_csv(os.path.join(tables_dir,"03_top_bigrams_positive.csv"), index=False)
        bi_neg.to_csv(os.path.join(tables_dir,"03_top_bigrams_negative.csv"), index=False)
        tri_pos.to_csv(os.path.join(tables_dir,"03_top_trigrams_positive.csv"), index=False)
        tri_neg.to_csv(os.path.join(tables_dir,"03_top_trigrams_negative.csv"), index=False)
        lo_df.to_csv(os.path.join(tables_dir,"03_log_odds_all.csv"), index=False)
        tfidf_df = pd.DataFrame({"word": feature_names, "tfidf_pos": mean_pos, "tfidf_neg": mean_neg, "diff": diff})
        tfidf_df.sort_values("diff", ascending=False).to_csv(
            os.path.join(tables_dir,"03_tfidf_discriminative.csv"), index=False)
        print(f"  -> Saved all tables to {tables_dir}/")

    print(f"\n{'='*60}")
    print(f"  DONE — {step} figures + summary tables generated")
    print(f"{'='*60}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    perform_eda(
        os.path.join("data","processed","cleaned_imdb_reviews.csv"),
        os.path.join("outputs","figures"),
        os.path.join("outputs","tables"),
    )
