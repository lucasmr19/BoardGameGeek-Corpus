#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 4: Classification Model Construction

Trains supervised classification models to predict review polarity using 
the vector representations from Exercise 2.

Models implemented:
- Multinomial Naive Bayes
- Support Vector Machines (LinearSVC)
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

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from src.bgg_corpus.config import VECTORS_DIR, MODELS_DIR
from src.bgg_corpus.resources import LOGGER

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    LOGGER.warning("XGBoost not available. Install with: pip install xgboost")


class ModelTrainer:
    """Trains classification models using pre-computed vector representations."""

    def __init__(self, vector_dir: str, output_dir: str, seed: int = 42):
        self.vector_dir = vector_dir
        self.output_dir = output_dir
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
        # TF-IDF features come first
        n_tfidf = self.vectorizer.tfidf.transform(
            self.vectorizer._prefix_tokens_with_language([['sample']], ['en'])
        ).shape[1]
        n_total = X.shape[1]
        
        self.n_tfidf_features = n_tfidf
        self.n_opinion_features = n_total - n_tfidf
        
        LOGGER.info(f"Feature distribution:")
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

    def train_model(self, model_name: str, model, X, y, feature_subset: str):
        """Train a single model with a specific feature subset."""
        LOGGER.info(f"\n{'='*70}")
        LOGGER.info(f"Training: {model_name} | Features: {feature_subset}")
        LOGGER.info(f"{'='*70}")
        
        # Extract feature subset
        X_subset = self.get_feature_subset(X, feature_subset)
        LOGGER.info(f"Feature subset shape: {X_subset.shape}")
        
        # Apply MinMaxScaler for MultinomialNB (requires non-negative features)
        scaler = None
        if isinstance(model, MultinomialNB):
            LOGGER.info("Applying MinMax scaling for MultinomialNB...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_subset = scaler.fit_transform(
                X_subset.toarray() if hasattr(X_subset, "toarray") else X_subset
            )
        
        # Train model
        LOGGER.info("Training model...")
        model.fit(X_subset, y)
        LOGGER.info("✓ Model trained successfully")
        
        # Save model
        model_filename = f"{model_name}_{feature_subset}.pkl"
        model_path = os.path.join(self.output_dir, model_filename)
        joblib.dump(model, model_path)
        LOGGER.info(f"✓ Model saved: {model_path}")
        
        # Save scaler if used
        if scaler is not None:
            scaler_filename = f"{model_name}_{feature_subset}_scaler.pkl"
            scaler_path = os.path.join(self.output_dir, scaler_filename)
            joblib.dump(scaler, scaler_path)
            LOGGER.info(f"✓ Scaler saved: {scaler_path}")
        
        return model

    def run_training(self):
        """Main pipeline: train all models with all feature subsets."""
        # Generate feature matrix
        X, y = self.generate_feature_matrix()
        self.determine_feature_slices(X)
        
        # Define models to train
        models = {
            'MultinomialNB': MultinomialNB(alpha=1.0),
            'LinearSVM': LinearSVC(C=1.0, max_iter=2000, random_state=self.seed),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.seed,
                n_jobs=-1
            )
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.seed,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        # Feature subsets to evaluate
        feature_subsets = ['ngrams', 'opinion', 'combined']
        
        # Train all combinations
        trained_count = 0
        total_models = len(models) * len(feature_subsets)
        
        LOGGER.info(f"\nStarting training: {len(models)} models × {len(feature_subsets)} feature subsets = {total_models} total configurations")
        LOGGER.info("="*70)
        
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
        
        LOGGER.info("\n" + "="*70)
        LOGGER.info(f"TRAINING COMPLETED: {trained_count}/{total_models} models trained successfully")
        LOGGER.info(f"Models saved to: {self.output_dir}")
        LOGGER.info("="*70)

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
    
    args = parser.parse_args()
    
    # Initialize trainer and run
    trainer = ModelTrainer(
        vector_dir=args.vector_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    trainer.run_training()


if __name__ == "__main__":
    main()