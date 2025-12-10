#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate train/val/test splits from BoardGameGeek corpus.
Saves each split as a .pkl file with raw text (clean_text) and labels.

Usage example:
    python scripts/create_splits_bert.py \
        --corpus path/to/bgg_corpus.json \
        --output_dir path/to/save/splits \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import os
import json
from collections import Counter
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.bgg_corpus.models import Corpus
from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import CORPORA_DIR, CORPUS_NAME, SPLITS_DIR_BERT

# -----------------------------
# Saving functions
# -----------------------------
def save_splits(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, save_format="pkl"):
    os.makedirs(output_dir, exist_ok=True)

    if save_format == "pkl":
        joblib.dump(X_train, os.path.join(output_dir, "train_texts.pkl"))
        joblib.dump(X_val, os.path.join(output_dir, "val_texts.pkl"))
        joblib.dump(X_test, os.path.join(output_dir, "test_texts.pkl"))
        joblib.dump(y_train, os.path.join(output_dir, "y_train.pkl"))
        joblib.dump(y_val, os.path.join(output_dir, "y_val.pkl"))
        joblib.dump(y_test, os.path.join(output_dir, "y_test.pkl"))
        LOGGER.info("Saved splits in PKL format.")
    elif save_format == "json":
        for split, X, y in zip(["train","val","test"], [X_train,X_val,X_test], [y_train,y_val,y_test]):
            with open(os.path.join(output_dir,f"{split}.json"),"w",encoding="utf8") as f:
                json.dump({"X": X, "y": y}, f, ensure_ascii=False, indent=2)
        LOGGER.info("Saved splits in JSON format.")
    elif save_format == "csv":
        for split, X, y in zip(["train","val","test"], [X_train,X_val,X_test], [y_train,y_val,y_test]):
            pd.DataFrame({"text": X, "label": y}).to_csv(os.path.join(output_dir,f"{split}.csv"), index=False)
        LOGGER.info("Saved splits in CSV format.")
    elif save_format == "npy":
        np.save(os.path.join(output_dir,"X_train.npy"), np.array(X_train))
        np.save(os.path.join(output_dir,"X_val.npy"), np.array(X_val))
        np.save(os.path.join(output_dir,"X_test.npy"), np.array(X_test))
        np.save(os.path.join(output_dir,"y_train.npy"), np.array(y_train))
        np.save(os.path.join(output_dir,"y_val.npy"), np.array(y_val))
        np.save(os.path.join(output_dir,"y_test.npy"), np.array(y_test))
        LOGGER.info("Saved splits in NPY format.")
    else:
        raise ValueError(f"Unknown save format: {save_format}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits from BGG corpus with flexible format")
    parser.add_argument("--corpus", type=str, default=os.path.join(CORPORA_DIR, f"{CORPUS_NAME}.json"))
    parser.add_argument("--output_dir", type=str, default=SPLITS_DIR_BERT, help="Directory to save the splits")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--format", choices=["pkl","json","csv","npy"], default="pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-config", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    LOGGER.info(f"Loading corpus from {args.corpus}")
    corpus = Corpus.from_json(args.corpus)

    texts, labels = [], []
    skipped_docs = 0
    for doc in corpus.documents:
        clean_text = doc.clean_text
        if not clean_text:
            skipped_docs += 1
            continue
        texts.append(clean_text)
        labels.append(doc.category)

    LOGGER.info(f"Total documents: {len(corpus.documents)}, processed: {len(texts)}, skipped: {skipped_docs}")

    # Normalize ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    # Train/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=args.test_ratio, stratify=labels, random_state=args.seed
    )

    val_size = args.val_ratio / (args.train_ratio + args.val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=args.seed
    )

    save_splits(args.output_dir, X_train, X_val, X_test, y_train, y_val, y_test, args.format)

    # Log split statistics
    # '''
    LOGGER.info("\n" + "="*60)
    LOGGER.info("DATASET SPLIT SUMMARY")
    LOGGER.info("="*60)
    LOGGER.info(f"Train: {len(X_train):>6} samples ({len(X_train)/len(texts)*100:>5.1f}%)")
    LOGGER.info(f"Val:   {len(X_val):>6} samples ({len(X_val)/len(texts)*100:>5.1f}%)")
    LOGGER.info(f"Test:  {len(X_test):>6} samples ({len(X_test)/len(texts)*100:>5.1f}%)")
    LOGGER.info("-"*60)
    LOGGER.info(f"Train distribution: {dict(Counter(y_train))}")
    LOGGER.info(f"Val   distribution: {dict(Counter(y_val))}")
    LOGGER.info(f"Test  distribution: {dict(Counter(y_test))}")
    LOGGER.info("="*60 + "\n")
    # '''
    
    # Save config
    if args.save_config:
        cfg = {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "format": args.format,
            "total_samples": len(texts),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "class_distribution": {
                "train": dict(Counter(y_train)),
                "val": dict(Counter(y_val)),
                "test": dict(Counter(y_test)),
            },
            "save_config": args.save_config,
        }
        with open(os.path.join(args.output_dir, "split_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    LOGGER.info("✓ Splits created successfully.")

if __name__ == "__main__":
    main()