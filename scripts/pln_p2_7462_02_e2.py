#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vector representation of BoardGameGeek reviews using TF-IDF and opinion features.
Generates and saves:
- TF-IDF based n-grams (unigrams + bigrams)
- Opinion / sentiment features
- Combined sparse representation

Usage example:
    python scripts/pln_p2_7462_02_e2.py \
        --corpus path/to/bgg_corpus.json \
        --output_dir path/to/save/vectors \
        --max_features 8000 \
        --ngram_range 1 2
"""

import os
import argparse
from collections import Counter
from scipy.sparse import save_npz
import joblib

from src.bgg_corpus.models import Corpus
from src.bgg_corpus.features.vectorization import ReviewVectorizer
from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import CORPORA_DIR, VECTORS_DIR

def main():
    # ----------------------------
    # 0. Parse command-line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="Generate TF-IDF + opinion feature vectors for BGG reviews")
    parser.add_argument("--corpus", type=str, default=os.path.join(CORPORA_DIR, "bgg_corpus.json"),
                        help="Path to the processed corpus JSON")
    parser.add_argument("--output_dir", type=str, default=VECTORS_DIR,
                        help="Directory to save vectorized matrices and vectorizer")
    parser.add_argument("--max_features", type=int, default=8000,
                        help="Maximum number of TF-IDF features")
    parser.add_argument("--ngram_range", type=int, nargs=2, default=[1, 2],
                        help="n-gram min/max value range for TF-IDF, e.g., 1 2 for unigrams + bigrams")
    args = parser.parse_args()

    # ----------------------------
    # 1. Load processed corpus
    # ----------------------------
    LOGGER.info(f"Loading corpus from {args.corpus}")
    corpus = Corpus.from_json(args.corpus)

    # ----------------------------
    # 2. Prepare input features
    # ----------------------------
    tokens_per_doc = []
    langs = []
    opinion_features = []
    skipped_docs = 0

    for i, doc in enumerate(corpus.documents):
        tokens_no_stop = doc.processed.get("tokens_no_stopwords")
        if not tokens_no_stop:
            skipped_docs += 1
            continue

        tokens_per_doc.append(tokens_no_stop)
        langs.append(doc.language)
        opinion_features.append(doc.linguistic_features)

    # ----------------------------
    # 3. Summarize corpus statistics
    # ----------------------------
    total_docs = len(corpus.documents)
    processed_docs = len(tokens_per_doc)
    LOGGER.info(f"Total documents: {total_docs}")
    LOGGER.info(f"Documents processed: {processed_docs}, skipped: {skipped_docs}")

    if tokens_per_doc:
        lang_counts = Counter(langs)
        doc_lengths = [len(t) for t in tokens_per_doc]
        LOGGER.info(f"Language distribution: {dict(lang_counts)}")
        LOGGER.info(
            "Tokens per document: min=%d, max=%d, mean=%.1f" %
            (min(doc_lengths), max(doc_lengths), sum(doc_lengths)/len(doc_lengths))
        )

    # ----------------------------
    # 4. Initialize vectorizer
    # ----------------------------
    vec = ReviewVectorizer(max_features=args.max_features,
                           ngram_range=(args.ngram_range[0], args.ngram_range[1]),
                           stopwords=None)

    # ----------------------------
    # 5. Fit-transform and save
    # ----------------------------
    if tokens_per_doc:
        LOGGER.info("Vectorizing reviews...")
        X = vec.fit_transform(tokens_per_doc, langs, opinion_features)

        os.makedirs(args.output_dir, exist_ok=True)
        save_npz(os.path.join(args.output_dir, "bgg_combined_matrix.npz"), X)
        joblib.dump(vec, os.path.join(args.output_dir, "bgg_vectorizer.pkl"))

        LOGGER.info("TF-IDF + linguistic/opinion vectorization complete")
    else:
        LOGGER.error("No documents with tokens found, vectorization skipped")


if __name__ == "__main__":
    main()