#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 4: Classification Model Construction

Trains supervised classification models to predict review polarity using 
the vector representations from Exercise 2.

Models implemented:
- Multinomial Naive Bayes
- SGD Classifier (SVC with SGD training if `loss`=hinge)
- Random Forest
- XGBoost (if available)

Feature subsets:
- N-grams only (TF-IDF)
- Linguistic/sentiment features only
- Combined features (both)

Usage:
    python scripts/pln_p2_7462_02_e4.py \
        --vector_dir path/to/vectors \
        --output_dir path/to/models
"""

import argparse

from src.bgg_corpus.config import VECTORS_DIR, MODELS_DIR, SPLITS_DIR

from src.modeling import ModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 4: Train classification models using vector representations from Ex2"
    )
    parser.add_argument(
        "--vector_dir",
        type=str,
        default=VECTORS_DIR,
        help="Directory containing vectors from Exercise 2"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=MODELS_DIR,
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=SPLITS_DIR,
        help="Directory containing pre-split datasets from Exercise 3"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "npz"],
        default="npz",
        help="Format of split datasets from Ex3"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer and run
    trainer = ModelTrainer(
        vector_dir=args.vector_dir,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        split_format=args.format,
        seed=args.seed,
    )
    
    trainer.run_training()


if __name__ == "__main__":
    main()