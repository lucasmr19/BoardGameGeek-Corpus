#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Creation Script
Description:
    This script loads a preprocessed and labeled corpus of board game reviews
    and generates training, validation, and test datasets for sentiment
    classification tasks.

Usage example:
    python pln_p2_7462_02_e3.py \
        --corpus_path data/bgg_corpus.json \
        --output_dir data/processed/datasets \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42 \
        --format json \
        --verbose
"""

import os
import random
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

from src.bgg_corpus.models import Corpus
from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import CORPORA_DIR, PROCESSED_DIR


# ---------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test datasets from a labeled corpus.")
    parser.add_argument("--corpus_path", type=str,
                        default=os.path.join(CORPORA_DIR, "bgg_corpus.json"),
                        help="Path to the labeled corpus JSON file.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROCESSED_DIR, "datasets"),
                        help="Directory where the datasets will be saved.")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Proportion of data used for training.")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Proportion of data used for validation.")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Proportion of data used for testing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                        help="Output file format (json or csv).")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging.")
    parser.add_argument("--save-config", action="store_true", default=True,
                        help="Save the configuration used for dataset creation.")
    return parser.parse_args()


# ---------------------------------------------------------------
# Extract documents into a flat list of review data
# ---------------------------------------------------------------
def extract_documents(corpus: Corpus) -> list[dict]:
    """Flatten all CorpusDocuments into a list of review records."""
    records = []
    for doc in corpus.documents:
        text = doc.clean_text
        records.append({
            "game_id": doc.game_id,
            "username": getattr(doc.review, "username", None),
            "rating": getattr(doc.review, "rating", None),
            "timestamp": getattr(doc.review, "timestamp", None),
            "language": doc.language,
            "text": text,
            "category": doc.category
        })
    return records


# ---------------------------------------------------------------
# Main script logic
# ---------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    # 1. Load corpus
    if args.verbose:
        LOGGER.info(f"Loading corpus from {args.corpus_path}")
    corpus = Corpus.from_json(args.corpus_path)
    if args.verbose:
        LOGGER.info(f"Loaded {len(corpus.games)} games and {len(corpus.documents)} documents")

    # 2. Extract reviews
    reviews = extract_documents(corpus)
    df = pd.DataFrame(reviews)
    if args.verbose:
        LOGGER.info(f"Extracted {len(df)} labeled reviews")
        LOGGER.info(f"Class distribution in full dataset: {Counter(df['category'])}")

    # 3. Split dataset (stratified)
    train_val_df, test_df = train_test_split(
        df,
        test_size=args.test_ratio,
        stratify=df['category'],
        random_state=args.seed
    )
    val_size = args.val_ratio / (args.train_ratio + args.val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['category'],
        random_state=args.seed
    )
    if args.verbose:
        LOGGER.info(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

    # 4. Save resulting datasets
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, f"train.{args.format}")
    val_path = os.path.join(args.output_dir, f"val.{args.format}")
    test_path = os.path.join(args.output_dir, f"test.{args.format}")

    if args.format == "json":
        train_df.to_json(train_path, orient="records", force_ascii=False, indent=2)
        val_df.to_json(val_path, orient="records", force_ascii=False, indent=2)
        test_df.to_json(test_path, orient="records", force_ascii=False, indent=2)
    else:
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

    if args.verbose:
        LOGGER.info(f"Datasets successfully saved in {args.output_dir}")
        LOGGER.info(f"Train/Val/Test distributions:")
        LOGGER.info(f"Train: {Counter(train_df['category'])}")
        LOGGER.info(f"Val:   {Counter(val_df['category'])}")
        LOGGER.info(f"Test:  {Counter(test_df['category'])}")

    # 5. Save summary log to file
    if args.save_config:
        summary_path = os.path.join(args.output_dir, "dataset_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset summary\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Total documents: {len(df)}\n")
            f.write(f"Train/Val/Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}\n\n")
            f.write("Class distributions:\n")
            f.write(f"Full dataset: {dict(Counter(df['category']))}\n")
            f.write(f"Train: {dict(Counter(train_df['category']))}\n")
            f.write(f"Validation: {dict(Counter(val_df['category']))}\n")
            f.write(f"Test: {dict(Counter(test_df['category']))}\n\n")
            # Save config used
            f.write("Configuration used:\n")
            f.write(json.dumps(vars(args), indent=2))

        if args.verbose:
            LOGGER.info(f"Dataset summary saved to {summary_path}")


if __name__ == "__main__":
    main()