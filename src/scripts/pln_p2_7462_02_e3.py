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
        --vector_dir data/processed/vectors \
        --output_dir data/processed/datasets \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42 \
        --format npz \
        --verbose
"""

import os
import random
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split
from collections import Counter

from src.bgg_corpus.models import Corpus
from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import VECTORS_DIR, SPLITS_DIR


# ---------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/val/test datasets from vectorized data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vector_dir", type=str,
                        default=VECTORS_DIR,
                        help="Directory with vectorized data (bgg_combined_matrix.npz and vectorizer_data.pkl).")
    parser.add_argument("--output_dir", type=str,
                        default=SPLITS_DIR,
                        help="Directory where the datasets will be saved.")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Proportion of data used for training.")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Proportion of data used for validation.")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Proportion of data used for testing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--format", choices=["json", "csv", "npz"], default="npz",
                        help="Output format: 'npz' for sparse matrices (recommended), "
                             "'json' for metadata only, 'csv' for tabular data.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging.")
    parser.add_argument("--save-config", action="store_true", default=True,
                        help="Save the configuration used for dataset creation.")
    return parser.parse_args()


# ---------------------------------------------------------------
# Save functions for different formats
# ---------------------------------------------------------------
def save_split_npz(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    """Save splits as sparse matrices (NPZ)."""
    save_npz(os.path.join(output_dir, "X_train.npz"), X_train)
    save_npz(os.path.join(output_dir, "X_val.npz"), X_val)
    save_npz(os.path.join(output_dir, "X_test.npz"), X_test)

    joblib.dump(y_train, os.path.join(output_dir, "y_train.npz"))
    joblib.dump(y_val, os.path.join(output_dir, "y_val.npz"))
    joblib.dump(y_test, os.path.join(output_dir, "y_test.npz"))

    LOGGER.info(f"Saved NPZ format (X and y) to {output_dir}")


def save_split_json(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    """Save splits as JSON."""
    # Convert sparse to dense for JSON
    X_train_dense = X_train.toarray().tolist()
    X_val_dense = X_val.toarray().tolist()
    X_test_dense = X_test.toarray().tolist()

    splits = {
        "train": {"X": X_train_dense, "y": y_train.tolist()},
        "val": {"X": X_val_dense, "y": y_val.tolist()},
        "test": {"X": X_test_dense, "y": y_test.tolist()},
    }

    for name, data in splits.items():
        output_path = os.path.join(output_dir, f"{name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"Saved JSON format (X and y) to {output_dir}")


def save_split_csv(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    """Save splits as CSV."""
    # Convert sparse matrices to DataFrames
    X_train_df = pd.DataFrame(X_train.toarray())
    X_val_df = pd.DataFrame(X_val.toarray())
    X_test_df = pd.DataFrame(X_test.toarray())

    X_train_df['label'] = y_train
    X_val_df['label'] = y_val
    X_test_df['label'] = y_test

    X_train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding='utf-8')
    X_val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding='utf-8')
    X_test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding='utf-8')

    LOGGER.info(f"Saved CSV format (X and y) to {output_dir}")


# ---------------------------------------------------------------
# Main script logic
# ---------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load vectorized data
    if args.verbose:
        LOGGER.info(f"Loading vectorized data from {args.vector_dir}")
    
    vectorizer_data = joblib.load(os.path.join(args.vector_dir, 'vectorizer_data.pkl'))
    X = load_npz(os.path.join(args.vector_dir, 'bgg_combined_matrix.npz'))
    
    y = np.array(vectorizer_data['categories'])

    if args.verbose:
        LOGGER.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        LOGGER.info(f"Full dataset class distribution: {Counter(y)}")

    # 2. Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        LOGGER.warning(f"Ratios sum to {total_ratio:.3f}, not 1.0. Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    # 3. Split into train/val/test (stratified)
    if args.verbose:
        LOGGER.info("Splitting dataset (stratified by category)...")
    
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=args.test_ratio,
        stratify=y,
        random_state=args.seed
    )

    # Second split: separate train and validation
    val_size = args.val_ratio / (args.train_ratio + args.val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        stratify=y_train_val,
        random_state=args.seed
    )


    # 4. Log split statistics
    LOGGER.info("\n" + "="*60)
    LOGGER.info("DATASET SPLIT SUMMARY")
    LOGGER.info("="*60)
    LOGGER.info(f"Train: {X_train.shape[0]:>6} samples ({X_train.shape[0]/X.shape[0]*100:>5.1f}%)")
    LOGGER.info(f"Val:   {X_val.shape[0]:>6} samples ({X_val.shape[0]/X.shape[0]*100:>5.1f}%)")
    LOGGER.info(f"Test:  {X_test.shape[0]:>6} samples ({X_test.shape[0]/X.shape[0]*100:>5.1f}%)")
    LOGGER.info("-"*60)
    LOGGER.info(f"Train distribution: {dict(Counter(y_train))}")
    LOGGER.info(f"Val   distribution: {dict(Counter(y_val))}")
    LOGGER.info(f"Test  distribution: {dict(Counter(y_test))}")
    LOGGER.info("="*60 + "\n")

    # 5. Prepare configuration metadata
    config = {
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "format": args.format,
        "total_samples": X.shape[0],
        "n_features": X.shape[1],
        "train_size": X_train.shape[0],
        "val_size": X_val.shape[0],
        "test_size": X_test.shape[0],
        "class_distribution": {
            "train": dict(Counter(y_train)),
            "val": dict(Counter(y_val)),
            "test": dict(Counter(y_test)),
        },
        "save_config": args.save_config,
    }

    # 6. Save in the requested format
    if args.format == "npz":
        save_split_npz(args.output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    elif args.format == "json":
        save_split_json(args.output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    elif args.format == "csv":
        save_split_csv(args.output_dir, X_train, X_val, X_test, y_train, y_val, y_test)

    
    if config.get('save_config'):
        with open(os.path.join(args.output_dir, "split_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    LOGGER.info(f"✓ Dataset splits successfully created in '{args.output_dir}'")


if __name__ == "__main__":
    main()