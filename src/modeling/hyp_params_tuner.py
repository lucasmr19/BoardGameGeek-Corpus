#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)

from src.bgg_corpus.resources import LOGGER

# Import shared utilities
from .model_utils import (
    PipelineFactory,
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
            'classifier__n_estimators': [200, 300],
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