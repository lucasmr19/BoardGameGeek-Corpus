import os
import sys
import io
from collections import Counter
import matplotlib.pyplot as plt
import json
import pandas as pd

from ..config import CORPORA_STATISTICS_DIR

def load_json(path):
    """Carga un archivo JSON y devuelve un dict, o None si no existe."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_csv(path):
    """Carga un CSV como DataFrame (o DataFrame vacío si no existe)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def generate_corpus_statistics(corpus, base_path=CORPORA_STATISTICS_DIR):
    """
    Generate and save general statistics for a given corpus.

    This function computes general descriptive statistics, prints them to the console,
    and saves both a text report and generated figures in:
        data/processed/corpora/statistics/

    Parameters
    ----------
    corpus : Corpus
        The corpus object containing reviews and metadata.
    base_path : str, optional
        Path where reports and figures will be stored. Default is
        "data/processed/corpora/statistics".

    Output
    ------
    - A text report file named `corpus_statistics_<timestamp>.txt`
    - Figures (PNG) saved in a `figures/` subfolder.
    
    Example
    -------
    >>> generate_corpus_statistics(corpus)
    📊 General statistics
    Report saved to: data/processed/corpora/statistics/corpus_statistics_20251103_143500.txt
    Figures saved to: data/processed/corpora/statistics/figures/
    """
    # Create directories
    report_path = os.path.join(base_path, f"corpus_statistics_report.txt")
    figures_path = os.path.join(base_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # Capture printed output
    buffer = io.StringIO()
    sys.stdout = buffer

    print("\n📊 General statistics")
    print("- Total reviews:", corpus.num_reviews())
    print("- Rated reviews:", corpus.num_reviews_rated())
    print("- Reviews with text:", corpus.num_reviews_commented())
    print("- Reviews with rating & text:", corpus.num_reviews_rated_and_commented())
    print("- Unique users:", corpus.num_unique_users())
    print("- Non-unique users:", corpus.num_no_unique_users())
    print("- 10 non-unique users:", corpus.no_unique_users()[:10])

    # Ratings distribution
    print("\n⭐ Ratings distribution (top 10)")
    rating_dist = corpus.rating_distribution()
    for rating, count in sorted(rating_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"  {rating:>4}: {count}")

    # Sample raw text
    print("\n📝 Sample raw text:")
    print(corpus.raw()[:5], "...\n")

    # Word contexts
    print("🔍 Contexts for 'game' (window=3, first 5):")
    for ctx in corpus.contexts("game", window=3)[:5]:
        print(" ", ctx)

    # Common contexts
    print("\n⚖️ Top 5 common contexts for ['good', 'bad'] (window=2):")
    print(corpus.common_contexts(["good", "bad"], window=2)[:5])

    # Word frequency
    print("\n📌 Most frequent words (top 15):")
    for word, freq in corpus.most_common(15):
        print(f"  {word}: {freq}")

    # Lexical dispersion graph
    fig = corpus.lexical_dispersion_plot(["good", "game", "player"])
    fig.savefig(os.path.join(figures_path, "lexical_dispersion.png"))
    plt.close(fig)

    # Hapax legomena
    print("\n🟢 Hapax legomena (first 20):")
    print(corpus.hapaxes()[:20])

    # Word length distribution
    print("\n📏 Word length distribution (top 10):")
    length_dist = corpus.word_length_distribution()
    for length, count in sorted(length_dist.items())[:10]:
        print(f"  Length {length}: {count} occurrences")

    # Word length graph
    fig = corpus.plot_word_length_distribution()
    fig.savefig(os.path.join(figures_path, "word_length_distribution.png"))
    plt.close(fig)

    # N-grams and collocations
    print("\n🔗 Top 10 bigrams:")
    for bg, freq in Counter(corpus.bigrams()).most_common(10):
        print(f"  {bg}: {freq}")

    print("\n🔗 Top 5 trigrams:")
    for tg, freq in Counter(corpus.trigrams()).most_common(5):
        print(f"  {tg}: {freq}")

    print("\n📎 Top 10 collocations:")
    for coll, freq in corpus.collocations(10):
        print(f"  {coll}: {freq}")

    # Category comparison
    print("\n📂 Token counts by category:")
    corpus.print_category_stats()

    print("\n📂 Review counts by category:")
    corpus.print_review_counts()

    # Word frequency graph
    fig, words, freqs = corpus.plot_frequency_distribution(30, title="Corpus Word Frequency")
    fig.savefig(os.path.join(figures_path, "word_frequency_distribution.png"))
    plt.close(fig)

    # Restore console output
    sys.stdout = sys.__stdout__

    # Write report
    os.makedirs(base_path, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    print("\n✅ Corpus statistics generated successfully.")
    print(f"📄 Report saved to: {report_path}")
    print(f"🖼️ Figures saved to: {figures_path}")
