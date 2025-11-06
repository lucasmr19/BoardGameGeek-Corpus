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

import os
import argparse
import joblib
import numpy as np
from collections import Counter
from scipy.sparse import load_npz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.bgg_corpus.config import VECTORS_DIR, MODELS_DIR, SPLITS_DIR
from src.bgg_corpus.resources import LOGGER


class ShiftToPositive(BaseEstimator, TransformerMixin):
    """Shift all features so that the minimum value becomes zero."""
    def fit(self, X, y=None):
        if hasattr(X, "min"):  # works for sparse or dense
            self.min_ = X.min()
        else:
            self.min_ = np.min(X)
        return self

    def transform(self, X):
        if self.min_ < 0:
            if hasattr(X, "toarray"):  # handle sparse matrices
                X = X.toarray()
            return X - self.min_
        return X


class ModelTrainer:
    """Trains classification models using pre-computed vector representations."""

    def __init__(self, vector_dir: str, output_dir: str, split_format: str,
                 splits_dir: str = SPLITS_DIR, seed: int = 42):
        self.vector_dir = vector_dir
        self.output_dir = output_dir
        self.split_format = split_format
        self.splits_dir = splits_dir
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)

        # Load vectorizer and data from Exercise 2
        LOGGER.info(f"Loading vectorizer from: {vector_dir}")
        self.vectorizer = joblib.load(os.path.join(vector_dir, "bgg_vectorizer.pkl"))
        
        vectorizer_data = joblib.load(os.path.join(vector_dir, 'vectorizer_data.pkl'))
        self.tokens_per_doc = vectorizer_data['tokens_per_doc']
        self.langs = vectorizer_data['langs']
        self.opinion_features = vectorizer_data['opinion_features']
        self.categories = vectorizer_data['categories']
        
        LOGGER.info(f"Loaded {len(self.categories)} documents")
        LOGGER.info(f"Class distribution: {Counter(self.categories)}")
    
    def load_splits(self, splits_dir: str):
        """Load pre-split datasets (train/val/test) saved in npz, json, or csv formats."""
        LOGGER.info(f"Loading pre-split datasets from: {splits_dir}")
        fmt = self.split_format.lower()

        def load_npz_split(name):
            """Load a split when stored as NPZ."""
            X = load_npz(os.path.join(splits_dir, f"X_{name}.npz"))
            y = joblib.load(os.path.join(splits_dir, f"y_{name}.npz"))
            return X, np.array(y)

        def load_json_split(name):
            """Load a split when stored as JSON."""
            import json
            with open(os.path.join(splits_dir, f"{name}.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            X = np.array(data["X"])
            y = np.array(data["y"])
            return X, y

        def load_csv_split(name):
            """Load a split when stored as CSV."""
            import pandas as pd
            df = pd.read_csv(os.path.join(splits_dir, f"{name}.csv"))
            X = df.drop(columns=["label"]).to_numpy()
            y = df["label"].to_numpy()
            return X, y

        loaders = {
            "npz": load_npz_split,
            "json": load_json_split,
            "csv": load_csv_split,
        }

        if fmt not in loaders:
            raise ValueError(f"Unsupported format: {fmt}. Expected one of: npz, json, csv")

        load_fn = loaders[fmt]
        X_train, y_train = load_fn("train")
        X_val, y_val = load_fn("val")
        X_test, y_test = load_fn("test")

        LOGGER.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        LOGGER.info(f"Train class distribution: {Counter(y_train)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def generate_feature_matrix(self):
        """Generate the full feature matrix using the vectorizer from Ex2."""
        LOGGER.info("Generating feature matrix...")
        X = self.vectorizer.transform(
            self.tokens_per_doc,
            self.langs,
            self.opinion_features
        )
        y = np.array(self.categories)
        
        LOGGER.info(f"Feature matrix shape: {X.shape}")
        return X, y

    def determine_feature_slices(self, X):
        """Determine indices for TF-IDF and opinion features."""
        n_tfidf = len(self.vectorizer.tfidf.vocabulary_)
        n_total = X.shape[1]

        self.n_tfidf_features = n_tfidf
        self.n_opinion_features = n_total - n_tfidf

        LOGGER.info("Feature distribution:")
        LOGGER.info(f"  - TF-IDF (n-grams): {self.n_tfidf_features}")
        LOGGER.info(f"  - Opinion features: {self.n_opinion_features}")
        LOGGER.info(f"  - Total features: {n_total}")

    def get_feature_subset(self, X, subset_type: str):
        """Extract specific feature subset from the combined matrix."""
        if subset_type == "ngrams":
            return X[:, :self.n_tfidf_features]
        elif subset_type == "opinion":
            return X[:, self.n_tfidf_features:]
        elif subset_type == "combined":
            return X
        else:
            raise ValueError(f"Unknown subset type: {subset_type}")

    def create_pipeline(self, model_name: str, base_model):
        """
        Create a pipeline with appropriate preprocessing for each model.
        
        Scaling strategies:
        - MultinomialNB: MaxAbsScaler (requires non-negative values)
        - SGDClassifier: StandardScaler (benefits from standardization)
        - RandomForest: No scaling (tree-based, scale-invariant)
        - XGBoost: No scaling (tree-based, scale-invariant)
        """
        if model_name == "MultinomialNB":
            LOGGER.info("  → Pipeline: ShiftToPositive + MaxAbsScaler + MultinomialNB")
            return Pipeline([
                ('shift', ShiftToPositive()),
                ('scaler', MaxAbsScaler()),
                ('classifier', base_model)
            ])
        
        elif model_name == "SGDClassifier":
            LOGGER.info("  → Pipeline: StandardScaler + SGDClassifier")
            return Pipeline([
                ('scaler', StandardScaler(with_mean=False)),  # with_mean=False for sparse matrices
                ('classifier', base_model)
            ])
        
        elif model_name in ["RandomForest", "XGBoost"]:
            LOGGER.info(f"  → Pipeline: {model_name} (no scaling needed)")
            return Pipeline([
                ('classifier', base_model)
            ])
        
        else:
            LOGGER.info(f"  → Pipeline: {model_name} (default, no scaling)")
            return Pipeline([
                ('classifier', base_model)
            ])

    def train_model(self, model_name: str, base_model, X, y, feature_subset: str):
        print(f"\n{'='*70}")
        LOGGER.info(f"Training: {model_name} | Features: {feature_subset}")
        print(f"{'='*70}")

        X_subset = self.get_feature_subset(X, feature_subset)
        LOGGER.info(f"Feature subset shape: {X_subset.shape}")

        label_encoder = None

        # Label encoding for XGBoost
        if model_name == "XGBoost":
            LOGGER.info("Encoding class labels for XGBoost...")
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y

        # Create pipeline with appropriate scaling
        pipeline = self.create_pipeline(model_name, base_model)

        # Train pipeline
        LOGGER.info("Training pipeline...")
        pipeline.fit(X_subset, y_encoded)
        LOGGER.info("✓ Pipeline trained successfully")

        # Save pipeline
        model_filename = f"{model_name}_{feature_subset}.pkl"
        joblib.dump(pipeline, os.path.join(self.output_dir, model_filename))
        LOGGER.info(f"✓ Pipeline saved: {model_filename}")

        # Save label encoder separately (if any)
        if label_encoder is not None:
            encoder_filename = f"{model_name}_{feature_subset}_label_encoder.pkl"
            joblib.dump(label_encoder, os.path.join(self.output_dir, encoder_filename))
            LOGGER.info(f"✓ LabelEncoder saved: {encoder_filename}")

        return pipeline

    def run_training(self):
        """Main pipeline: train all models with all feature subsets."""
        # Load splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_splits(self.splits_dir)
        X = X_train
        y = y_train
        self.determine_feature_slices(X)
        
        # Define models to train
        models = {
            'MultinomialNB': MultinomialNB(alpha=1.0),
            'SGDClassifier': SGDClassifier(
                loss='hinge',           # SVM-like behavior
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                tol=1e-3,
                random_state=self.seed,
                n_jobs=-1
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.seed,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.seed,
                eval_metric='logloss'
            )
        }
        
        # Feature subsets to evaluate
        feature_subsets = ['ngrams', 'opinion', 'combined']
        
        # Train all combinations
        trained_count = 0
        total_models = len(models) * len(feature_subsets)
        
        LOGGER.info(f"\nStarting training: {len(models)} models × {len(feature_subsets)} feature subsets = {total_models} total configurations")
        print("="*70)
        
        for model_name, model in models.items():
            for subset in feature_subsets:
                try:
                    self.train_model(model_name, model, X, y, subset)
                    trained_count += 1
                    LOGGER.info(f"Progress: {trained_count}/{total_models} models trained")
                except Exception as e:
                    LOGGER.error(f"✗ Error training {model_name} with {subset} features: {e}")
        
        # Save training summary
        self.save_training_summary(trained_count, total_models)
        
        print("\n" + "="*70)
        LOGGER.info(f"TRAINING COMPLETED: {trained_count}/{total_models} models trained successfully")
        LOGGER.info(f"Models saved to: {self.output_dir}")
        print("="*70)

    def save_training_summary(self, trained_count: int, total_models: int):
        """Save a summary of the training process."""
        summary_path = os.path.join(self.output_dir, "training_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EXERCISE 4: MODEL TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total documents: {len(self.categories)}\n")
            f.write(f"Class distribution: {dict(Counter(self.categories))}\n")
            f.write(f"Total features: {self.n_tfidf_features + self.n_opinion_features}\n")
            f.write(f"  - TF-IDF features: {self.n_tfidf_features}\n")
            f.write(f"  - Opinion features: {self.n_opinion_features}\n\n")
            
            f.write(f"Models trained: {trained_count}/{total_models}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write("Model configurations:\n")
            f.write("  - MultinomialNB: Pipeline with ShiftToPositive + MaxAbsScaler\n")
            f.write("  - SGDClassifier: Pipeline with StandardScaler\n")
            f.write("  - RandomForest: No scaling (tree-based)\n")
            f.write("  - XGBoost: No scaling (tree-based)\n")
            f.write("\n")
            
            f.write("Trained model files:\n")
            for filename in sorted(os.listdir(self.output_dir)):
                if filename.endswith('.pkl') and filename != 'training_summary.txt':
                    f.write(f"  - {filename}\n")
        
        LOGGER.info(f"✓ Training summary saved: {summary_path}")


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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=SPLITS_DIR,
        help="Directory containing pre-split datasets from Exercise 3 (optional)."
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "npz"],
        default="npz",
        help="Output format saved for splits in Ex3"
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