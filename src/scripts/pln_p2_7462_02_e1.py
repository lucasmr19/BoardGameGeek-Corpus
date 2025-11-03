#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linguistic feature extraction from BoardGameGeek reviews.

This script extracts linguistic features such as sentiment, negation, intensifiers, 
domain terms, and hedges from preprocessed BoardGameGeek reviews. 
It relies on lexicons and utilities defined in:
`src/bgg_corpus/features/SentimentLexicon.py`

The extracted features are saved back into the corpus JSON.

NOTE:
This script is somewhat redundant because these features are already extracted 
in the main preprocessing review pipeline implemented in:
`src/bgg_corpus/preprocessing/review_processor.py`

Example:
--------
    $ python src/scripts/extract_linguistic_features.py \
        --corpus data/processed/corpora/bgg_corpus.json \
        --output_dir data/processed/corpora/

This will:
    1. Load the specified corpus.
    2. Extract linguistic features for each review.
    3. Save the resulting corpus with added features to the same folder as `<CORPUS_NAME>_features.json`.
"""

import os
import argparse
from tqdm import tqdm

from src.bgg_corpus.models import Corpus
from src.bgg_corpus.features import LinguisticFeaturesExtractor
from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import CORPORA_DIR, CORPUS_NAME

def main():
    parser = argparse.ArgumentParser(description="Extract linguistic features for BGG reviews")
    parser.add_argument(
        "--corpus",
        type=str,
        default=os.path.join(CORPORA_DIR, f"{CORPUS_NAME}.json"),
        help="Path to the corpus JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CORPORA_DIR,
        help="Directory to save extracted features"
    )
    args = parser.parse_args()

    # Load corpus
    LOGGER.info(f"Loading corpus from {args.corpus}")
    corpus = Corpus.from_json(args.corpus)

    extractor = LinguisticFeaturesExtractor()
    skipped = 0

    for doc in tqdm(corpus.documents, desc="Processing reviews"):
        try:
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

        except Exception as e:
            skipped += 1
            LOGGER.warning(f"Skipping document due to error: {e}")

    LOGGER.info(f"Processed {len(corpus.documents) - skipped} reviews, skipped {skipped}")

    # Save updated corpus with linguistic features
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{CORPUS_NAME}_features.json")
    corpus.to_json(output_path)
    LOGGER.info(f"Linguistic features saved to {output_path}")


if __name__ == "__main__":
    main()