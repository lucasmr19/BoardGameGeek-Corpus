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
        --dataset_dir path/to/datasets \
        --models_dir path/to/models \
        --output_dir path/to/results
"""

import os
import argparse
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.bgg_corpus.config import VECTORS_DIR, DATASETS_DIR, MODELS_DIR
from src.bgg_corpus.resources import LOGGER

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    LOGGER.warning("XGBoost not available.")


class ModelEvaluator:
    """Evaluates trained models on test datasets with hyperparameter tuning."""

    def __init__(self, vector_dir: str, dataset_dir: str, models_dir: str, 
                 output_dir: str, seed: int = 42):
        self.vector_dir = vector_dir
        self.dataset_dir = dataset_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.seed = seed
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)

        # Load vectorizer and data
        LOGGER.info(f"Loading vectorizer from: {vector_dir}")
        self.vectorizer = joblib.load(os.path.join(vector_dir, "bgg_vectorizer.pkl"))
        
        vectorizer_data = joblib.load(os.path.join(vector_dir, 'vectorizer_data.pkl'))
        self.tokens_per_doc = vectorizer_data['tokens_per_doc']
        self.langs = vectorizer_data['langs']
        self.opinion_features = vectorizer_data['opinion_features']
        self.categories = vectorizer_data['categories']
        self.doc_ids = vectorizer_data['doc_ids']

    def load_split(self, split_name: str):
        """Load a dataset split and extract corresponding vectors."""
        split_path = os.path.join(self.dataset_dir, f"{split_name}.json")
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)

        # Extract doc_ids in this split
        split_doc_ids = [d['doc_id'] for d in split_data]

        # Find indices in the global list
        id_to_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        split_indices = [id_to_index[doc_id] for doc_id in split_doc_ids]

        # Subset the features
        tokens_split = [self.tokens_per_doc[i] for i in split_indices]
        langs_split = [self.langs[i] for i in split_indices]
        opinion_split = [self.opinion_features[i] for i in split_indices]
        y_split = [self.categories[i] for i in split_indices]

        # Transform
        X_split = self.vectorizer.transform(tokens_split, langs_split, opinion_split)

        LOGGER.info(f"{split_name.upper()}: {X_split.shape[0]} samples, "
                   f"distribution: {Counter(y_split)}")
        
        return X_split, np.array(y_split)

    def determine_feature_slices(self, X_sample):
        """Determine feature boundaries."""
        n_tfidf = self.vectorizer.tfidf.transform(
            self.vectorizer._prefix_tokens_with_language([['sample']], ['en'])
        ).shape[1]
        n_total = X_sample.shape[1]
        
        self.n_tfidf_features = n_tfidf
        self.n_opinion_features = n_total - n_tfidf
        
        LOGGER.info(f"Features: TF-IDF={self.n_tfidf_features}, "
                   f"Opinion={self.n_opinion_features}, Total={n_total}")

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

    def tune_hyperparameters(self, model_name: str, X_train, y_train, 
                            feature_subset: str, search_type: str = 'grid'):
        """Perform hyperparameter tuning."""
        LOGGER.info(f"\nTuning hyperparameters for {model_name} ({feature_subset})...")
        
        # Define parameter grids
        param_grids = {
            'MultinomialNB': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'LinearSVM': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [2000]
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
        # Initialize base model
        if model_name == 'MultinomialNB':
            base_model = MultinomialNB()
        elif model_name == 'LinearSVM':
            base_model = LinearSVC(random_state=self.seed)
        elif model_name == 'RandomForest':
            base_model = RandomForestClassifier(random_state=self.seed, n_jobs=-1)
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            base_model = xgb.XGBClassifier(
                random_state=self.seed,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            LOGGER.warning(f"Model {model_name} not available for tuning")
            return None
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                base_model,
                param_grids.get(model_name, {}),
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                base_model,
                param_grids.get(model_name, {}),
                n_iter=10,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=self.seed,
                verbose=1
            )
        else:
            raise Exception(f"search_type must be 'random' or 'grid'")
        
        search.fit(X_train, y_train)
        
        LOGGER.info(f"Best params: {search.best_params_}")
        LOGGER.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_

    def evaluate_model(self, model_name: str, feature_subset: str,
                      X_train, y_train, X_test, y_test, X_val, y_val,
                      tune_hyperparams: bool = True):
        """Evaluate a single model configuration."""
        LOGGER.info(f"\n{'='*70}")
        LOGGER.info(f"Evaluating: {model_name} | Features: {feature_subset}")
        LOGGER.info(f"{'='*70}")
        
        # Extract feature subsets
        X_train_subset = self.get_feature_subset(X_train, feature_subset)
        X_test_subset = self.get_feature_subset(X_test, feature_subset)
        X_val_subset = self.get_feature_subset(X_val, feature_subset)
        
        # Apply scaling if needed
        scaler = None
        if model_name == 'MultinomialNB':
            LOGGER.info("Applying MinMax scaling...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_subset = scaler.fit_transform(
                X_train_subset.toarray() if hasattr(X_train_subset, "toarray") 
                else X_train_subset
            )
            X_test_subset = scaler.transform(
                X_test_subset.toarray() if hasattr(X_test_subset, "toarray") 
                else X_test_subset
            )
            X_val_subset = scaler.transform(
                X_val_subset.toarray() if hasattr(X_val_subset, "toarray") 
                else X_val_subset
            )
        
        # Tune hyperparameters if requested
        if tune_hyperparams:
            model = self.tune_hyperparameters(
                model_name, X_train_subset, y_train, feature_subset
            )
            if model is None:
                return
        else:
            # Load pre-trained model from Exercise 4
            model_path = os.path.join(self.models_dir, 
                                     f"{model_name}_{feature_subset}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                LOGGER.info(f"Loaded pre-trained model from: {model_path}")
            else:
                LOGGER.warning(f"Pre-trained model not found: {model_path}")
                return
        
        # Predictions
        y_val_pred = model.predict(X_val_subset)
        y_test_pred = model.predict(X_test_subset)
        
        # Calculate metrics
        val_acc = accuracy_score(y_val, y_val_pred)
        val_p, val_r, val_f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='weighted', zero_division=0
        )
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_p, test_r, test_f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='weighted', zero_division=0
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
        labels = sorted(list(set(y_test)))
        target_names = [label.capitalize() for label in labels]
        LOGGER.info("\nClassification Report:")
        LOGGER.info("\n" + classification_report(
            y_test, y_test_pred, target_names=target_names, zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
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

    def run_evaluation(self, tune_hyperparams: bool = True):
        """Main evaluation pipeline."""
        # Load splits
        X_train, y_train = self.load_split('train')
        X_test, y_test = self.load_split('test')
        X_val, y_val = self.load_split('val')
        
        self.determine_feature_slices(X_train)
        
        # Define models and feature subsets
        model_names = ['MultinomialNB', 'LinearSVM', 'RandomForest']
        if XGBOOST_AVAILABLE:
            model_names.append('XGBoost')
        
        feature_subsets = ['ngrams', 'opinion', 'combined']
        
        total = len(model_names) * len(feature_subsets)
        evaluated = 0
        
        LOGGER.info(f"\nStarting evaluation: {len(model_names)} models × "
                   f"{len(feature_subsets)} subsets = {total} configurations")
        LOGGER.info("="*70)
        
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
                    LOGGER.error(f"Error evaluating {model_name}/{subset}: {e}")
        
        # Generate reports
        self.generate_technical_report()
        
        LOGGER.info(f"\n{'='*70}")
        LOGGER.info(f"EVALUATION COMPLETED: {evaluated}/{total} configurations")
        LOGGER.info(f"Results saved to: {self.output_dir}")
        LOGGER.info(f"{'='*70}")

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
            f.write(f"Total documents: {len(self.categories)}\n")
            f.write(f"Class distribution: {dict(Counter(self.categories))}\n\n")
            
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
        "--dataset_dir",
        type=str,
        default=DATASETS_DIR,
        help="Directory with train/test splits from Ex3"
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
        help="Perform hyperparameter tuning"
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        vector_dir=args.vector_dir,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    evaluator.run_evaluation(tune_hyperparams=args.tune)


if __name__ == "__main__":
    main()