#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linguistic feature extraction from BoardGameGeek reviews.
- Extract sentiment, negation, intensifiers, domain terms, hedges, etc. defined in src/features/SentimentLexicon
- Save results in the corpus.

NOTE: This script is redundant because these features are already extracted in the main preprocessing review
pipeline implemented in:
BoardGameGeek-Corpus/src/bgg_corpus/preprocessing/review_processor.py.
"""

import os
import argparse
from tqdm import tqdm

from src.bgg_corpus.models import Corpus
from src.bgg_corpus.features import LinguisticFeaturesExtractor
from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import CORPORA_DIR, FEATURES_DIR

def main():
    parser = argparse.ArgumentParser(description="Extract linguistic features for BGG reviews")
    parser.add_argument("--corpus", type=str, default=os.path.join(CORPORA_DIR, "bgg_corpus.json"),
                        help="Path to the corpus JSON")
    parser.add_argument("--output_dir", type=str, default=FEATURES_DIR,
                        help="Directory to save extracted features")
    args = parser.parse_args()

    # Load corpus
    LOGGER.info(f"Loading corpus from {args.corpus}")
    corpus = Corpus.from_json(args.corpus)

    # Same code as in review_processor.py
    extractor = LinguisticFeaturesExtractor()
    skipped = 0

    for doc in tqdm(corpus.documents, desc="Processing reviews"):
        tokens_no_stop = doc.processed.get("tokens_no_stopwords", [])
        lemmas = doc.processed.get("lemmas", [])
        sentences = doc.processed.get("sentences", [])
        dependencies = doc.processed.get("dependencies", [])
        pos_tags = doc.processed.get("pos_tags", [])
        raw_text = doc.raw_text or ""

        features = extractor.extract_features(
            lemmas=lemmas,
            tokens_no_stopwords=tokens_no_stop,
            dependencies=dependencies,
            sentences=sentences,
            pos_tags=pos_tags,
            raw_text=raw_text
        )

        doc.linguistic_features = features

    LOGGER.info(f"Processed {len(corpus.documents)-skipped} reviews, skipped {skipped}")

    # Save updated corpus with linguistic features
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "bgg_corpus.json")
    corpus.to_json(output_path)
    LOGGER.info(f"Linguistic features saved to {output_path}")


if __name__ == "__main__":
    main()
