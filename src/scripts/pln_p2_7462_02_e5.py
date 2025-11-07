#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 5: Classification Algorithm Evaluation and Technical Report

Implements rigorous evaluation of trained models and generates a technical report.

Evaluation includes:
- Testing on train/test/val splits from Exercise 3
- Hyperparameter tuning via GridSearchCV or RandomizedSearchCV
- Classification metrics: accuracy, precision, recall, F1-score
- Confusion matrices for best models
- Comprehensive technical report generation

Usage:
    python scripts/pln_p2_7462_02_e5.py \
        --vector_dir path/to/vectors \
        --splits_dir path/to/datasets \
        --models_dir path/to/models \
        --output_dir path/to/results \
        --tune
"""

import os
import argparse

from src.bgg_corpus.config import VECTORS_DIR, SPLITS_DIR, MODELS_DIR

# Import shared utilities
from ..modeling import (
    ModelEvaluator
)


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 5: Evaluate classification models with hyperparameter tuning"
    )
    parser.add_argument(
        "--vector_dir",
        type=str,
        default=VECTORS_DIR,
        help="Directory with vectors from Ex2"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=SPLITS_DIR,
        help="Directory with train/test/valid splits from Ex3"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "npz"],
        default="npz",
        help="Format of split datasets from Ex3"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=MODELS_DIR,
        help="Directory with trained models from Ex4"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(MODELS_DIR, "evaluation"),
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning (default: use pre-trained models)"
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        vector_dir=args.vector_dir,
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        split_format=args.format,
        seed=args.seed
    )
    
    evaluator.run_evaluation(tune_hyperparams=args.tune)


if __name__ == "__main__":
    main()