#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import os
from itertools import product
from joblib import Parallel, delayed, load
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.bgg_corpus.resources import LOGGER

# Import shared utilities
from .model_utils import (
    DataLoader,
    FeatureManager,
    PipelineFactory,
    LabelEncoderManager,
    get_model_instance
)


class HyperparameterTuner:
    """Handles hyperparameter tuning for different models."""
    
    # Hyps to test (Add more as needed...)
    PARAM_GRIDS = {
        'MultinomialNB': {
            # α controla el suavizado de Laplace — afecta directamente la dispersión de probas
            'classifier__alpha': [0.1, 0.5, 1.0]
        },
        
        'SGDClassifier': {
            # α (regularización), loss (SVM lineal o regresión logística)
            'classifier__alpha': [1e-5, 1e-4, 1e-3],
            'classifier__loss': ['hinge', 'log_loss'],  # SVM vs Logistic Regression
            'classifier__penalty': ['l2', 'elasticnet'],
            'classifier__max_iter': [10000],
            'classifier__tol': [1e-3]  # evita convergencia muy laxa
        },
        
        'RandomForest': {
            # Menos combinaciones pero con impacto real
            #'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt', 'log2']  # controla la aleatoriedad en splits
        },
        
        'XGBoost': {
            # Los más influyentes en boosting
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__subsample': [0.8, 1.0],  # control del bagging interno
            'classifier__colsample_bytree': [0.8, 1.0]  # número de features por árbol
        }
    }

    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def make_composite_scorer(self, alpha_acc=0.25, alpha_prec=0.25, alpha_rec=0.25, alpha_f1=0.25):
        """
        Custom weighted scoring function combining accuracy, precision, recall and F1.
        By default, returns the average.
        """
        def composite_score(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            return alpha_acc * acc + alpha_prec * p + alpha_rec * r + alpha_f1 * f1
        return make_scorer(composite_score, greater_is_better=True)
    
    def tune(self, model_name: str, X_train, y_train, 
             search_type: str = 'grid', cv: int = 3) -> tuple:
        """
        Perform hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        
        Returns:
            Tuple of (best_pipeline, best_params, best_score)
        """
        LOGGER.info(f"Tuning hyperparameters for {model_name}...")
        
        # Get base model and create pipeline
        base_model = get_model_instance(model_name, self.seed)
        pipeline = PipelineFactory.create(model_name, base_model)
        
        # Create a custom scorer
        custom_scorer = self.make_composite_scorer(
            alpha_acc=0.25, alpha_prec=0.25, alpha_rec=0.25, alpha_f1=0.25
        )
        
        # Get parameter grid
        param_grid = self.PARAM_GRIDS.get(model_name, {})
        
        if not param_grid:
            LOGGER.warning(f"No parameter grid defined for {model_name}")
            return pipeline, {}, None
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring=custom_scorer,
                n_jobs=-1,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=10,
                cv=cv,
                scoring=custom_scorer,
                n_jobs=-1,
                random_state=self.seed,
                verbose=1
            )
        else:
            raise ValueError(f"search_type must be 'grid' or 'random'")
        
        # Perform search
        search.fit(X_train, y_train)
        
        LOGGER.info(f"Best params: {search.best_params_}")
        LOGGER.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_


class ModelEvaluator:
    """
    Evaluates trained models on test datasets.
    Inherits core functionality patterns from ModelTrainer.
    """
    
    MODEL_NAMES = ['MultinomialNB', 'SGDClassifier', 'RandomForest', 'XGBoost']
    FEATURE_SUBSETS = ['ngrams', 'opinion', 'combined']

    def __init__(self, vector_dir: str, splits_dir: str, models_dir: str, 
                 output_dir: str, split_format: str, seed: int = 42):
        self.vector_dir = vector_dir
        self.splits_dir = splits_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.split_format = split_format
        self.seed = seed
        self.results: List[Dict] = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)
        
        # Initialize managers
        self.data_loader = DataLoader(splits_dir, split_format)
        self.label_encoder_manager = LabelEncoderManager(models_dir)
        self.tuner = HyperparameterTuner(seed)
        
        # Load vectorizer
        LOGGER.info(f"Loading vectorizer from: {vector_dir}")
        self.vectorizer = load(os.path.join(vector_dir, "bgg_vectorizer.pkl"))
        
        # Initialize feature manager
        self.feature_manager = FeatureManager(self.vectorizer)
        
        # Dataset sizes (populated after loading)
        self.dataset_sizes: Dict[str, int] = {}
    
    def _load_pretrained_model(self, model_name: str, feature_subset: str):
        """Load a pre-trained model pipeline from Exercise 4."""
        model_path = os.path.join(self.models_dir, 
                                 f"{model_name}_{feature_subset}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pre-trained model not found: {model_path}")
        
        pipeline = load(model_path)
        LOGGER.info(f"✓ Loaded pre-trained pipeline from: {model_path}")
        
        return pipeline
    
    def evaluate_model(self, model_name: str, feature_subset: str,
                      X_train, y_train, X_val, y_val, X_test, y_test,
                      tune_hyperparams: bool = False, search_type: str = 'grid', cv: int = 3):
        """
        Evaluate a single model configuration.
        
        Args:
            model_name: Name of the model
            feature_subset: Type of features ('ngrams', 'opinion', 'combined')
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            tune_hyperparams: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*70}")
        LOGGER.info(f"Evaluating: {model_name} | Features: {feature_subset}")
        print(f"{'='*70}")
        
        # Extract feature subsets
        X_train_subset = self.feature_manager.get_subset(X_train, feature_subset)
        X_val_subset = self.feature_manager.get_subset(X_val, feature_subset)
        X_test_subset = self.feature_manager.get_subset(X_test, feature_subset)
        
        # Handle XGBoost label encoding
        needs_encoding = model_name == "XGBoost"
        
        if needs_encoding:
            try:
                y_train_encoded = self.label_encoder_manager.transform(y_train)
                y_val_for_pred = y_val
                y_test_for_pred = y_test
            except FileNotFoundError:
                LOGGER.warning("Label encoder not found. Skipping XGBoost evaluation.")
                return None
        else:
            y_train_encoded = y_train
            y_val_for_pred = y_val
            y_test_for_pred = y_test
        
        # Load or tune model
        best_params = {}
        cv_score = None
        
        if tune_hyperparams:
            pipeline, best_params, cv_score = self.tuner.tune(
                model_name, X_train_subset, y_train_encoded, search_type, cv
            )
        else:
            try:
                pipeline = self._load_pretrained_model(model_name, feature_subset)
            except FileNotFoundError as e:
                LOGGER.error(f"✗ {e}")
                return None
        
        # Make predictions
        y_val_pred = pipeline.predict(X_val_subset)
        y_test_pred = pipeline.predict(X_test_subset)
        
        # Decode XGBoost predictions
        if needs_encoding:
            y_val_pred = self.label_encoder_manager.inverse_transform(y_val_pred)
            y_test_pred = self.label_encoder_manager.inverse_transform(y_test_pred)
        
        # Calculate metrics
        val_metrics = self._calculate_metrics(y_val_for_pred, y_val_pred)
        test_metrics = self._calculate_metrics(y_test_for_pred, y_test_pred)
        
        # Store results
        result = {
            'model': model_name,
            'features': feature_subset,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_avg_acc_prec_rec_f1': val_metrics['avg_acc_prec_rec_f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_avg_acc_prec_rec_f1': test_metrics['avg_acc_prec_rec_f1'],
            'tuned': tune_hyperparams,
            'best_params': best_params if tune_hyperparams else {},
            'cv_score': cv_score
        }
        self.results.append(result)
        
        # Log results
        #self._log_metrics("Validation", val_metrics)
        #self._log_metrics("Test", test_metrics)
        
        if tune_hyperparams and best_params:
            print(f"\nBest hyperparameters: {best_params}")
            print(f"CV Avg_acc_prec_rec_f1: {cv_score:.4f}")
        
        # Classification report
        labels = sorted(list(set(y_test_for_pred)))
        target_names = [str(label).capitalize() for label in labels]
        LOGGER.info("\nClassification Report for Test split:")
        LOGGER.info("\n" + classification_report(
            y_test_for_pred, y_test_pred, target_names=target_names, zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_for_pred, y_test_pred, labels=labels)
        LOGGER.info(f"\nConfusion Matrix:\n{cm}")
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(cm, target_names, model_name, feature_subset)
        
        return result
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        composite_scorer = self.tuner.make_composite_scorer()
        custom_score = composite_scorer._score_func(y_true, y_pred)
        
        return {
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
            'avg_acc_prec_rec_f1': custom_score,
        }
    
    def _log_metrics(self, split_name: str, metrics: Dict[str, float]):
        """Log metrics in a formatted way."""
        LOGGER.info(f"\n{split_name} Metrics:")
        LOGGER.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        LOGGER.info(f"  Precision: {metrics['precision']:.4f}")
        LOGGER.info(f"  Recall:    {metrics['recall']:.4f}")
        LOGGER.info(f"  F1-Score:  {metrics['f1']:.4f}")
    
    def _plot_confusion_matrix(self, cm, labels, model_name, feature_subset):
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
    
    def get_available_models(self) -> List[str]:
        """Get list of models available for evaluation."""
        available = []
        
        for model_name in self.MODEL_NAMES:
            try:
                get_model_instance(model_name, self.seed)
                available.append(model_name)
            except ImportError:
                LOGGER.warning(f"Model {model_name} not available")
        
        return available
    
    def _evaluate_combination(self, model_name, subset,
                          X_train, y_train, X_val, y_val, X_test, y_test,
                          tune_hyperparams):
        try:
            result = self.evaluate_model(
                model_name, subset,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                tune_hyperparams=tune_hyperparams
            )
            if result is not None:
                LOGGER.info(f"✓ Completed {model_name}/{subset}")
                return result
        except Exception as e:
            LOGGER.error(f"✗ Error evaluating {model_name}/{subset}: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
        return None
    
    def run_evaluation(self, tune_hyperparams: bool = False):
        """Main evaluation pipeline."""
        # Load splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load_splits()
        
        self.dataset_sizes = {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test)
        }
        
        # Determine feature slices
        self.feature_manager.determine_feature_slices({
            "train": X_train,
            "val": X_val,
            "test": X_test
        })
        
        # Get available models
        available_models = self.get_available_models()
        
        if not available_models:
            LOGGER.error("No models available for evaluation!")
            return
        
        # Evaluation statistics
        total = len(available_models) * len(self.FEATURE_SUBSETS)
        evaluated = 0
        
        LOGGER.info(f"\nStarting evaluation: {len(available_models)} models × "
                   f"{len(self.FEATURE_SUBSETS)} subsets = {total} configurations")
        print("="*70)
        
        # Evaluate all combinations
        """
        for model_name in available_models:
            for subset in self.FEATURE_SUBSETS:
                try:
                    result = self.evaluate_model(
                        model_name, subset,
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        tune_hyperparams=tune_hyperparams
                    )
                    if result is not None:
                        evaluated += 1
                    LOGGER.info(f"Progress: {evaluated}/{total}")
                except Exception as e:
                    LOGGER.error(f"✗ Error evaluating {model_name}/{subset}: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
        """
        tasks = list(product(available_models, self.FEATURE_SUBSETS))
        total = len(tasks)
        LOGGER.info(f"Launching parallel evaluation of {total} configurations...")
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(self._evaluate_combination)(
                model_name, subset,
                X_train, y_train, X_val, y_val, X_test, y_test,
                tune_hyperparams
            )
            for model_name, subset in tasks
        )
        results = [r for r in results if r is not None]
        evaluated = len(results)
        self.results.extend(results)
        LOGGER.info(f"✓ Completed {evaluated}/{total} evaluations successfully")
        
        # Generate reports
        self._generate_technical_report()
        
        print(f"\n{'='*70}")
        LOGGER.info(f"EVALUATION COMPLETED: {evaluated}/{total} configurations")
        LOGGER.info(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
    
    def _generate_technical_report(self):
        """Generate comprehensive technical report."""
        if not self.results:
            LOGGER.warning("No results to report")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort by test cv score
        sorted_results = sorted(self.results, key=lambda x: x['cv_score'], reverse=True)
        
        # Text report
        report_path = os.path.join(self.output_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MODEL EVALUATION TECHNICAL REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {timestamp}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Train: {self.dataset_sizes['train']} | "
                    f"Val: {self.dataset_sizes['val']} | "
                    f"Test: {self.dataset_sizes['test']}\n\n")
            
            f.write("MODEL CONFIGURATIONS\n")
            f.write("-"*70 + "\n")
            f.write("  - MultinomialNB: Pipeline with ShiftToPositive + MaxAbsScaler\n")
            f.write("  - SGDClassifier: Pipeline with StandardScaler\n")
            f.write("  - RandomForest: No scaling (tree-based)\n")
            f.write("  - XGBoost: No scaling (tree-based)\n\n")
            
            f.write("HYPERPARAMETER SEARCH CONFIGURATIONS\n")
            f.write("-"*70 + "\n")
            f.write(f"Search strategy: GridSearchCV (deterministic exhaustive search)\n")
            f.write(f"Cross-validation folds: 3\n")
            f.write(f"Random seed: {self.tuner.seed}\n")
            f.write(f"Scoring metric: custom composite (0.25xAcc + 0.25xPrec + 0.25xRec + 0.25xF1)\n\n")

            f.write("Parameter grids per model:\n")
            for model_name, grid in self.tuner.PARAM_GRIDS.items():
                f.write(f"  ▸ {model_name}:\n")
                if not grid:
                    f.write("     (no parameters tuned)\n")
                else:
                    for param, values in grid.items():
                        f.write(f"     {param}: {values}\n")
                f.write("\n")            
            
            f.write("EVALUATION RESULTS (sorted by Test F1-Score)\n")
            f.write("-"*70 + "\n\n")
            
            for i, r in enumerate(sorted_results, 1):
                f.write(f"{i}. {r['model']} ({r['features']} features)\n")
                f.write(f"   Validation - Acc: {r['val_accuracy']:.4f}, "
                       f"P: {r['val_precision']:.4f}, R: {r['val_recall']:.4f}, "
                       f"F1: {r['val_f1']:.4f}, Avg: {r['val_avg_acc_prec_rec_f1']:.4f}\n")
                f.write(f"   Test       - Acc: {r['test_accuracy']:.4f}, "
                       f"P: {r['test_precision']:.4f}, R: {r['test_recall']:.4f}, "
                       f"F1: {r['test_f1']:.4f}, Avg: {r['test_avg_acc_prec_rec_f1']:.4f}\n")
                
                if r['tuned'] and r['best_params']:
                    f.write(f"   Tuned: Yes\n")
                    f.write(f"   Best hyperparameters: {r['best_params']}\n")
                    if r['cv_score'] is not None:
                        f.write(f"   CV Score: {r['cv_score']:.4f}\n")
                else:
                    f.write(f"   Tuned: No (using pre-trained model)\n")
                f.write("\n")
            
            f.write("\nBEST MODEL\n")
            f.write("-"*70 + "\n")
            best = sorted_results[0]
            f.write(f"Model: {best['model']}\n")
            f.write(f"Features: {best['features']}\n")
            f.write(f"Test Accuracy: {best['test_accuracy']:.4f}\n")
            f.write(f"Test F1-Score: {best['test_f1']:.4f}\n")
            f.write(f"Test Composite (Avg Acc+Prec+Rec+F1): {best['test_avg_acc_prec_rec_f1']:.4f}\n")
            if best['tuned'] and best['best_params']:
                f.write(f"Tuned: Yes\n")
                f.write(f"Optimal hyperparameters: {best['best_params']}\n")
                if best['cv_score'] is not None:
                    f.write(f"Cross-validated composite score: {best['cv_score']:.4f}\n")
            else:
                f.write(f"Tuned: No (using pre-trained model)\n")

        LOGGER.info(f"✓ Technical report saved: {report_path}")
        
        # CSV export
        self._export_results_csv(sorted_results)
    
    def _export_results_csv(self, sorted_results):
        """Export results to CSV format."""
        csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        import csv
        
        fieldnames = sorted_results[0].keys()
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_results)
        
        LOGGER.info(f"✓ Results CSV saved: {csv_path}")