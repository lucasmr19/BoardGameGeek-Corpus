#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datasets Creation Script

This script loads either:
    - Sparse TF-IDF/BoW matrices (.npz) with tokens_no_stopwords located in VECTORS_DIR, or
    - Dense static embeddings (.npy) located in EMBEDDINGS_DIR

It then creates stratified train/val/test splits and stores them in the
requested output format (npz, json, csv).

The script automatically detects whether the input is sparse or dense,
but the user may also force the input format using --input_type {npz,npy}.

Usage Example:
    python create_splits.py \
        --embedding bow \
        --input_type auto \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import os
import json
import random
import argparse
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split

from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import (VECTORS_DIR_BOW, VECTORS_DIR_EMB_W2V, VECTORS_DIR_EMB_GLOVE,
                                   VECTORS_DIR_EMB_FASTTEXT, SPLITS_DIR_BOW,
                                   DATASETS_DIR, SPLITS_DIR_EMB_W2V, SPLITS_DIR_EMB_GLOVE, 
                                   SPLITS_DIR_EMB_FASTTEXT)

# -------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/val/test datasets (supports sparse .npz and dense .npy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--embedding",
        choices=["bow", "w2v", "glove", "fasttext"],
        required=True,
        help="Representation type."
    )

    parser.add_argument(
        "--vector_dir",
        type=str,
        default=None,
        help="Optional custom directory. If omitted, project defaults are used."
    )

    parser.add_argument("--output_dir", type=str, default=DATASETS_DIR)
    parser.add_argument("--input_type", choices=["auto", "npz", "npy"], default="auto")

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["json", "csv", "npz", "npy"], default="npz")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--save-config", action="store_true", default=True)

    return parser.parse_args()


# -------------------------------------------------------------------
# Detect npz / npy input
# -------------------------------------------------------------------
def detect_input_type(vector_dir, forced_type="auto"):
    if forced_type in ["npz", "npy"]:
        return forced_type

    if os.path.exists(os.path.join(vector_dir, "bgg_combined_matrix.npz")):
        return "npz"
    if os.path.exists(os.path.join(vector_dir, "doc_embeddings.npy")):
        return "npy"

    raise FileNotFoundError(
        f"Could not detect sparse or dense format in {vector_dir}.\n"
        "Expected bgg_combined_matrix.npz or doc_embeddings.npy."
    )


# -------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------
def load_sparse_vectors(vector_dir):
    X = load_npz(os.path.join(vector_dir, "bgg_combined_matrix.npz"))
    meta = joblib.load(os.path.join(vector_dir, "vectorizer_data.pkl"))
    y = np.array(meta["categories"])
    LOGGER.info(f"Loaded sparse matrix: {X.shape}")
    return X, y


def load_dense_vectors(vector_dir):
    X = np.load(os.path.join(vector_dir, "doc_embeddings.npy"))
    y = np.load(os.path.join(vector_dir, "labels.npy"))
    LOGGER.info(f"Loaded dense embeddings: {X.shape}")
    return X, y


# -------------------------------------------------------------------
# Saving
# -------------------------------------------------------------------
def save_split_npz(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    save_npz(os.path.join(output_dir, "X_train.npz"), X_train)
    save_npz(os.path.join(output_dir, "X_val.npz"), X_val)
    save_npz(os.path.join(output_dir, "X_test.npz"), X_test)
    save_npz(os.path.join(output_dir, "y_train.npz"), y_train)
    save_npz(os.path.join(output_dir, "y_val.npz"), y_val)
    save_npz(os.path.join(output_dir, "y_test.npz"), y_test)
    LOGGER.info("Saved splits in NPZ format.")

def save_split_npy(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    """Save splits as .npy (dense embeddings)."""
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    LOGGER.info("Saved splits in NPY format.")

def save_split_json(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    data = {
        "train": {"X": X_train.tolist(), "y": y_train.tolist()},
        "val":   {"X": X_val.tolist(), "y": y_val.tolist()},
        "test":  {"X": X_test.tolist(), "y": y_test.tolist()},
    }
    for split, d in data.items():
        with open(os.path.join(output_dir, f"{split}.json"), "w", encoding="utf8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved splits in JSON format.")


def save_split_csv(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    pd.DataFrame(X_train).assign(label=y_train).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(X_val).assign(label=y_val).to_csv(os.path.join(output_dir, "val.csv"), index=False)
    pd.DataFrame(X_test).assign(label=y_test).to_csv(os.path.join(output_dir, "test.csv"), index=False)
    LOGGER.info("Saved splits in CSV format.")

# -------------------------------------------------------------------
# Main saving function that decides format automatically
# -------------------------------------------------------------------
def save_splits(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, input_type, save_format):
    """
    Save train/val/test splits in the correct format:
    - Dense (.npy) for input_type='npy'
    - Sparse (.npz) for input_type='npz'
    Or override with save_format ('npy', 'npz', 'json', 'csv')
    """
    os.makedirs(output_dir, exist_ok=True)

    if save_format == "npy" or (input_type == "npy" and save_format == "npz"):
        # Always save dense embeddings as .npy
        save_split_npy(output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    elif save_format == "npz" or input_type == "npz":
        # Sparse matrices
        save_split_npz(output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    elif save_format == "json":
        save_split_json(output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    elif save_format == "csv":
        save_split_csv(output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    else:
        raise ValueError(f"Unknown save format: {save_format}")



# -------------------------------------------------------------------
# Directory maps (AUTO-MAP)
# -------------------------------------------------------------------
VECTOR_DIR_MAP = {
    "bow": VECTORS_DIR_BOW,
    "w2v": VECTORS_DIR_EMB_W2V,
    "glove": VECTORS_DIR_EMB_GLOVE,
    "fasttext": VECTORS_DIR_EMB_FASTTEXT,
}

OUTPUT_DIR_MAP = {
    "bow": SPLITS_DIR_BOW,
    "w2v": SPLITS_DIR_EMB_W2V,
    "glove": SPLITS_DIR_EMB_GLOVE,
    "fasttext": SPLITS_DIR_EMB_FASTTEXT,
}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. AUTO-SELECT vector_dir
    vector_dir = args.vector_dir or VECTOR_DIR_MAP[args.embedding]
    LOGGER.info(f"Using vector_dir = {vector_dir}")

    # 2. Detect type
    input_type = detect_input_type(vector_dir, args.input_type)
    LOGGER.info(f"Detected input type: {input_type}")

    # 3. Load data
    if input_type == "npz":
        X, y = load_sparse_vectors(vector_dir)
    elif input_type == "npy":
        X, y = load_dense_vectors(vector_dir)

    # 4. Splits
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_ratio, stratify=y, random_state=args.seed
    )
    val_size = args.val_ratio / (args.train_ratio + args.val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        stratify=y_train_val,
        random_state=args.seed,
    )

    # 5. AUTO-SELECT output_dir
    output_dir = OUTPUT_DIR_MAP[args.embedding] if args.output_dir == DATASETS_DIR else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    LOGGER.info(f"Saving splits to {output_dir}")

    # 6. Save splits
    save_splits(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, input_type, args.format)

    
    # Log split statistics
    # '''
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
    # '''
    
    # Save config
    if args.save_config:
        cfg = {
            "input_type": input_type,
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
        with open(os.path.join(output_dir, "split_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    LOGGER.info("✓ Splits created successfully.")

if __name__ == "__main__":
    main()