#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter tuning module for various models.
Includes:
BoW for ML classical approximations as NaiveBayes, SGD, RandomForest

"""
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from itertools import product

from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from src.bgg_corpus.resources import LOGGER

# Import shared utilities
from .model_utils import (
    PipelineFactory,
    get_model_instance
)

from .model_trainer_embeddings import (
    train_epoch, evaluate,
    FeedforwardNN, RecurrentNN)

from .model_trainer_bert import (BERTReviewDataset, train_epoch_bert, evaluate_bert)

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
class HyperparameterTunerEmbeddings:
    """Class for hyperparameter tuning in NN models using embeddings."""
    def __init__(self, model_type, input_dim, num_classes, device):
        self.model_type = model_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.results = []
    
    def get_param_grid(self, model_type):
        """Define hyperparameter grids for each model type"""
        if model_type == "fnn":
            return {
                'hidden_dims': [
                    [512, 256, 128],
                    [1024, 512, 256],
                    [1024, 512],
                    [512, 256]
                ],
                'dropout': [0.4, 0.5],
                'activation': ['relu', 'gelu', 'leakyrelu'],
                'normalization': ['batchnorm', 'layernorm', 'none'],
                'lr': [0.001, 0.0005, 0.0001],
                'batch_size': [64, 128],
                'weight_decay': [1e-5, 1e-4]
            }
        else:  # RNN, LSTM, GRU
            return {
                'hidden_dim': [128, 256, 512],
                'num_layers': [1, 2, 3],
                'bidirectional': [True, False],
                'dropout': [0.4, 0.5],
                'lr': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64, 128]
            }
    
    def create_model(self, params):
        """Create model with given hyperparameters"""
        if self.model_type == "fnn":
            model = FeedforwardNN(
                input_dim=self.input_dim,
                hidden_dims=params['hidden_dims'],
                num_classes=self.num_classes,
                dropout=params['dropout']
            )
        else:
            model = RecurrentNN(
                input_dim=self.input_dim,
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                num_classes=self.num_classes,
                rnn_type=self.model_type,
                bidirectional=params['bidirectional'],
                dropout=params['dropout']
            )
        return model.to(self.device)
    
    def train_with_config(self, params, train_loader, val_loader, epochs=30, patience=10):
        """Train model with specific configuration"""
        model = self.create_model(params)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params.get('weight_decay', 0))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience,
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_val_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, self.device)
            val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, self.device)
            val_acc = accuracy_score(val_labels, val_preds)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_loss, best_val_acc, model
    
    def tune(self, train_dataset, val_dataset, epochs=30, patience=10, max_trials=None):
        """Perform hyperparameter tuning"""
        param_grid = self.get_param_grid(self.model_type)
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        if max_trials and len(combinations) > max_trials:
            # Random sample if too many combinations
            np.random.shuffle(combinations)
            combinations = combinations[:max_trials]
        
        LOGGER.info(f"Testing {len(combinations)} hyperparameter configurations...")
        
        best_config = None
        best_score = 0
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            LOGGER.info(f"\nTrial {i+1}/{len(combinations)}: {params}")
            
            # Create dataloaders with current batch size
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
            
            try:
                val_loss, val_acc, model = self.train_with_config(
                    params, train_loader, val_loader, epochs, patience
                )
                
                result = {
                    'params': params,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                self.results.append(result)
                
                LOGGER.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if val_acc > best_score:
                    best_score = val_acc
                    best_config = (params, model)
                    LOGGER.info(f"New best configuration! Val Acc: {val_acc:.4f}")
            
            except Exception as e:
                LOGGER.error(f"Error with configuration {params}: {str(e)}")
                continue
        
        return best_config, self.results

# -------------------------------
# Hyperparameter Tuner for BERT
# -------------------------------
class HyperparameterTunerBERT:
    """Class for hyperparameter tuning in BERT fine-tuning"""
    def __init__(self, model_name, num_classes, device):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.results = []
    
    def get_param_grid(self):
        """Define hyperparameter grid for BERT"""
        return {
            'lr': [2e-5, 3e-5, 5e-5],
            'batch_size': [8, 16, 32],
            'max_length': [128, 256, 512],
            'warmup_steps': [0, 100, 500],
            'weight_decay': [0.0, 0.01]
        }
    
    def create_model(self):
        """Create fresh BERT model"""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        return model.to(self.device)
    
    def train_with_config(self, params, train_texts, train_labels, 
                         val_texts, val_labels, tokenizer, epochs=5, patience=3):
        """Train model with specific configuration"""
        # Create datasets
        train_dataset = BERTReviewDataset(
            train_texts, train_labels, tokenizer, params['max_length']
        )
        val_dataset = BERTReviewDataset(
            val_texts, val_labels, tokenizer, params['max_length']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=params['batch_size']
        )
        
        # Create model
        model = self.create_model()
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=params['warmup_steps'],
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch_bert(
                model, train_loader, criterion, optimizer, scheduler, self.device
            )
            val_loss, val_preds, val_labels_arr = evaluate_bert(
                model, val_loader, criterion, self.device
            )
            val_acc = accuracy_score(val_labels_arr, val_preds)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_loss, best_val_acc, model
    
    def tune(self, train_texts, train_labels, val_texts, val_labels, 
             tokenizer, epochs=5, patience=3, max_trials=None):
        """Perform hyperparameter tuning"""
        param_grid = self.get_param_grid()
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        if max_trials and len(combinations) > max_trials:
            np.random.shuffle(combinations)
            combinations = combinations[:max_trials]
        
        LOGGER.info(f"Testing {len(combinations)} hyperparameter configurations...")
        
        best_config = None
        best_score = 0
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            LOGGER.info(f"\nTrial {i+1}/{len(combinations)}: {params}")
            
            try:
                val_loss, val_acc, model = self.train_with_config(
                    params, train_texts, train_labels, 
                    val_texts, val_labels, tokenizer, epochs, patience
                )
                
                result = {
                    'params': params,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                self.results.append(result)
                
                LOGGER.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if val_acc > best_score:
                    best_score = val_acc
                    best_config = (params, model)
                    LOGGER.info(f"New best configuration! Val Acc: {val_acc:.4f}")
            
            except Exception as e:
                LOGGER.error(f"Error with configuration {params}: {str(e)}")
                continue
        
        return best_config, self.results