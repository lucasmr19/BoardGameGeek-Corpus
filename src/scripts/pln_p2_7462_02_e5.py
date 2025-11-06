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
        --output_dir path/to/results
"""

import os
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from scipy.sparse import load_npz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.bgg_corpus.config import VECTORS_DIR, SPLITS_DIR, MODELS_DIR
from src.bgg_corpus.resources import LOGGER

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    LOGGER.warning("XGBoost not available.")


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


class ModelEvaluator:
    """Evaluates trained models on test datasets with hyperparameter tuning."""

    def __init__(self, vector_dir: str, splits_dir: str, models_dir: str, 
                 output_dir: str, split_format: str, seed: int = 42):
        self.vector_dir = vector_dir
        self.splits_dir = splits_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.split_format = split_format
        self.seed = seed
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)

        # Load vectorizer and data
        LOGGER.info(f"Loading vectorizer from: {vector_dir}")
        self.vectorizer = joblib.load(os.path.join(vector_dir, "bgg_vectorizer.pkl"))

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

    def determine_feature_slices(self, X_splits: dict):
        """
        Determine indices for TF-IDF and opinion features
        across all splits and ensure consistency.
        """
        n_tfidf = len(self.vectorizer.tfidf.vocabulary_)

        n_totals = {name: X.shape[1] for name, X in X_splits.items()}
        unique_dims = set(n_totals.values())

        if len(unique_dims) > 1:
            LOGGER.warning(f"Inconsistent feature dimensions across splits: {n_totals}")
        n_total = max(unique_dims)

        self.n_tfidf_features = n_tfidf
        self.n_opinion_features = n_total - n_tfidf

        LOGGER.info("Feature distribution (verified across splits):")
        LOGGER.info(f"  - TF-IDF (n-grams): {self.n_tfidf_features}")
        LOGGER.info(f"  - Opinion features: {self.n_opinion_features}")
        LOGGER.info(f"  - Total features: {n_total}")

    def get_feature_subset(self, X, subset_type: str):
        """Extract feature subset."""
        if subset_type == "ngrams":
            return X[:, :self.n_tfidf_features]
        elif subset_type == "opinion":
            return X[:, self.n_tfidf_features:]
        elif subset_type == "combined":
            return X
        else:
            raise ValueError(f"Unknown subset: {subset_type}")

    def create_pipeline(self, model_name: str, base_model):
        """
        Create a pipeline matching Exercise 4 configuration.
        
        Scaling strategies:
        - MultinomialNB: ShiftToPositive + MaxAbsScaler
        - SGDClassifier: StandardScaler (with_mean=False for sparse)
        - RandomForest: No scaling
        - XGBoost: No scaling
        """
        if model_name == "MultinomialNB":
            return Pipeline([
                ('shift', ShiftToPositive()),
                ('scaler', MaxAbsScaler()),
                ('classifier', base_model)
            ])
        
        elif model_name == "SGDClassifier":
            return Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('classifier', base_model)
            ])
        
        elif model_name in ["RandomForest", "XGBoost"]:
            return Pipeline([
                ('classifier', base_model)
            ])
        
        else:
            return Pipeline([
                ('classifier', base_model)
            ])

    def tune_hyperparameters(self, model_name: str, X_train, y_train, 
                            feature_subset: str, search_type: str = 'grid'):
        """Perform hyperparameter tuning with pipelines."""
        LOGGER.info(f"\nTuning hyperparameters for {model_name} ({feature_subset})...")
        
        # Define parameter grids (with pipeline prefix 'classifier')
        param_grids = {
            'MultinomialNB': {
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'SGDClassifier': {
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__max_iter': [1000, 2000],
                'classifier__loss': ['hinge', 'log_loss']
            },
            'RandomForest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            },
            'XGBoost': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
        # Initialize base model
        if model_name == 'MultinomialNB':
            base_model = MultinomialNB()
        elif model_name == 'SGDClassifier':
            base_model = SGDClassifier(random_state=self.seed, n_jobs=-1)
        elif model_name == 'RandomForest':
            base_model = RandomForestClassifier(random_state=self.seed, n_jobs=-1)
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            base_model = xgb.XGBClassifier(
                random_state=self.seed,
                eval_metric='logloss'
            )
        else:
            LOGGER.warning(f"Model {model_name} not available for tuning")
            return None
        
        # Create pipeline
        pipeline = self.create_pipeline(model_name, base_model)
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                pipeline,
                param_grids.get(model_name, {}),
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                pipeline,
                param_grids.get(model_name, {}),
                n_iter=10,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=self.seed,
                verbose=1
            )
        else:
            raise ValueError(f"search_type must be 'random' or 'grid'")
        
        search.fit(X_train, y_train)
        
        LOGGER.info(f"Best params: {search.best_params_}")
        LOGGER.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_

    def evaluate_model(self, model_name: str, feature_subset: str,
                      X_train, y_train, X_test, y_test, X_val, y_val,
                      tune_hyperparams: bool = True):
        """Evaluate a single model configuration."""
        print(f"\n{'='*70}")
        LOGGER.info(f"Evaluating: {model_name} | Features: {feature_subset}")
        print(f"{'='*70}")
        
        # Extract feature subsets
        X_train_subset = self.get_feature_subset(X_train, feature_subset)
        X_test_subset = self.get_feature_subset(X_test, feature_subset)
        X_val_subset = self.get_feature_subset(X_val, feature_subset)
        
        # Load or tune model
        if tune_hyperparams:
            pipeline = self.tune_hyperparameters(
                model_name, X_train_subset, y_train, feature_subset
            )
            if pipeline is None:
                return
        else:
            # Load pre-trained pipeline from Exercise 4
            model_path = os.path.join(self.models_dir, 
                                     f"{model_name}_{feature_subset}.pkl")
            if os.path.exists(model_path):
                pipeline = joblib.load(model_path)
                LOGGER.info(f"✓ Loaded pre-trained pipeline from: {model_path}")
            else:
                LOGGER.warning(f"✗ Pre-trained pipeline not found: {model_path}")
                return
        
        # Handle XGBoost label encoding
        label_encoder = None
        y_train_encoded = y_train
        y_val_encoded = y_val
        y_test_encoded = y_test
        
        if model_name == "XGBoost":
            label_path = os.path.join(self.models_dir, 
                                     f"{model_name}_{feature_subset}_label_encoder.pkl")
            if os.path.exists(label_path):
                label_encoder = joblib.load(label_path)
                LOGGER.info(f"✓ Loaded label encoder for XGBoost")
                y_train_encoded = label_encoder.transform(y_train)
                y_val_encoded = label_encoder.transform(y_val)
                y_test_encoded = label_encoder.transform(y_test)
        
        # Predictions (pipeline handles scaling internally)
        y_val_pred = pipeline.predict(X_val_subset)
        y_test_pred = pipeline.predict(X_test_subset)
        
        # Decode XGBoost predictions
        if label_encoder is not None:
            y_val_pred = label_encoder.inverse_transform(y_val_pred)
            y_test_pred = label_encoder.inverse_transform(y_test_pred)
            # Use original labels for metrics
            y_val_encoded = y_val
            y_test_encoded = y_test
        
        # Calculate metrics
        val_acc = accuracy_score(y_val_encoded, y_val_pred)
        val_p, val_r, val_f1, _ = precision_recall_fscore_support(
            y_val_encoded, y_val_pred, average='weighted', zero_division=0
        )
        
        test_acc = accuracy_score(y_test_encoded, y_test_pred)
        test_p, test_r, test_f1, _ = precision_recall_fscore_support(
            y_test_encoded, y_test_pred, average='weighted', zero_division=0
        )
        
        # Store results
        result = {
            'model': model_name,
            'features': feature_subset,
            'val_accuracy': val_acc,
            'val_precision': val_p,
            'val_recall': val_r,
            'val_f1': val_f1,
            'test_accuracy': test_acc,
            'test_precision': test_p,
            'test_recall': test_r,
            'test_f1': test_f1,
            'tuned': tune_hyperparams
        }
        self.results.append(result)
        
        # Print detailed results
        LOGGER.info(f"\nValidation Metrics:")
        LOGGER.info(f"  Accuracy:  {val_acc:.4f}")
        LOGGER.info(f"  Precision: {val_p:.4f}")
        LOGGER.info(f"  Recall:    {val_r:.4f}")
        LOGGER.info(f"  F1-Score:  {val_f1:.4f}")
        
        LOGGER.info(f"\nTest Metrics:")
        LOGGER.info(f"  Accuracy:  {test_acc:.4f}")
        LOGGER.info(f"  Precision: {test_p:.4f}")
        LOGGER.info(f"  Recall:    {test_r:.4f}")
        LOGGER.info(f"  F1-Score:  {test_f1:.4f}")
        
        # Classification report
        labels = sorted(list(set(y_test_encoded)))
        target_names = [str(label).capitalize() for label in labels]
        LOGGER.info("\nClassification Report:")
        LOGGER.info("\n" + classification_report(
            y_test_encoded, y_test_pred, target_names=target_names, zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_test_pred, labels=labels)
        LOGGER.info(f"\nConfusion Matrix:\n{cm}")
        
        # Save confusion matrix plot
        self.plot_confusion_matrix(
            cm, target_names, model_name, feature_subset
        )
        
        return result

    def plot_confusion_matrix(self, cm, labels, model_name, feature_subset):
        """Generate and save confusion matrix visualization."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} - {feature_subset}\nConfusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f"cm_{model_name}_{feature_subset}.png"
        filepath = os.path.join(self.output_dir, 'confusion_matrices', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"✓ Confusion matrix saved: {filepath}")

    def run_evaluation(self, tune_hyperparams: bool = False):
        """Main evaluation pipeline."""
        # Load splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_splits(self.splits_dir)
        
        self.dataset_sizes = {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test)
        }
   
        self.determine_feature_slices({
            "train": X_train,
            "val": X_val,
            "test": X_test
        })
        
        # Define models and feature subsets (matching Exercise 4)
        model_names = ['MultinomialNB', 'SGDClassifier', 'RandomForest']
        if XGBOOST_AVAILABLE:
            model_names.append('XGBoost')
        
        feature_subsets = ['ngrams', 'opinion', 'combined']
        
        total = len(model_names) * len(feature_subsets)
        evaluated = 0
        
        LOGGER.info(f"\nStarting evaluation: {len(model_names)} models × "
                   f"{len(feature_subsets)} subsets = {total} configurations")
        print("="*70)
        
        # Evaluate all combinations
        for model_name in model_names:
            for subset in feature_subsets:
                try:
                    self.evaluate_model(
                        model_name, subset,
                        X_train, y_train,
                        X_test, y_test,
                        X_val, y_val,
                        tune_hyperparams=tune_hyperparams
                    )
                    evaluated += 1
                    LOGGER.info(f"Progress: {evaluated}/{total}")
                except Exception as e:
                    LOGGER.error(f"✗ Error evaluating {model_name}/{subset}: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
        
        # Generate reports
        self.generate_technical_report()
        
        print(f"\n{'='*70}")
        LOGGER.info(f"EVALUATION COMPLETED: {evaluated}/{total} configurations")
        LOGGER.info(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")

    def generate_technical_report(self):
        """Generate comprehensive technical report."""
        if not self.results:
            LOGGER.warning("No results to report")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort by test F1 score
        sorted_results = sorted(self.results, key=lambda x: x['test_f1'], 
                               reverse=True)
        
        # Text report
        report_path = os.path.join(self.output_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EXERCISE 5: MODEL EVALUATION TECHNICAL REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {timestamp}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Train: {self.dataset_sizes['train']} | "
                    f"Val: {self.dataset_sizes['val']} | "
                    f"Test: {self.dataset_sizes['test']}\n\n")
            
            f.write("MODEL CONFIGURATIONS (from Exercise 4)\n")
            f.write("-"*70 + "\n")
            f.write("  - MultinomialNB: Pipeline with ShiftToPositive + MaxAbsScaler\n")
            f.write("  - SGDClassifier: Pipeline with StandardScaler\n")
            f.write("  - RandomForest: No scaling (tree-based)\n")
            f.write("  - XGBoost: No scaling (tree-based)\n\n")
            
            f.write("EVALUATION RESULTS (sorted by Test F1-Score)\n")
            f.write("-"*70 + "\n\n")
            
            for i, r in enumerate(sorted_results, 1):
                f.write(f"{i}. {r['model']} ({r['features']} features)\n")
                f.write(f"   Validation - Acc: {r['val_accuracy']:.4f}, "
                       f"P: {r['val_precision']:.4f}, R: {r['val_recall']:.4f}, "
                       f"F1: {r['val_f1']:.4f}\n")
                f.write(f"   Test       - Acc: {r['test_accuracy']:.4f}, "
                       f"P: {r['test_precision']:.4f}, R: {r['test_recall']:.4f}, "
                       f"F1: {r['test_f1']:.4f}\n")
                f.write(f"   Hyperparameters tuned: {r['tuned']}\n\n")
            
            f.write("\nBEST MODEL\n")
            f.write("-"*70 + "\n")
            best = sorted_results[0]
            f.write(f"Model: {best['model']}\n")
            f.write(f"Features: {best['features']}\n")
            f.write(f"Test F1-Score: {best['test_f1']:.4f}\n")
            f.write(f"Test Accuracy: {best['test_accuracy']:.4f}\n")
        
        LOGGER.info(f"✓ Technical report saved: {report_path}")
        
        # CSV export
        csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_results[0].keys())
            writer.writeheader()
            writer.writerows(sorted_results)
        
        LOGGER.info(f"✓ Results CSV saved: {csv_path}")


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
        help="Output format saved for splits in Ex3"
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