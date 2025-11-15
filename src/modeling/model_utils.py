#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for model training and evaluation.

This module contains reusable components:
- Custom transformers (ShiftToPositive)
- Data loading utilities
- Pipeline factories
- Label encoding management
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from collections import Counter
from scipy.sparse import load_npz, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from src.bgg_corpus.features import ReviewVectorizer
from src.bgg_corpus.resources import LOGGER


class ShiftToPositive(BaseEstimator, TransformerMixin):
    """
    Shift all features so that the minimum value becomes zero.
    Compatible with sparse matrices and avoids read-only errors.
    """

    def fit(self, X, y=None):
        """Learn the minimum value from the training data."""
        if issparse(X):
            # Ensure it's writable before calling .min()
            X = X.copy()
            self.min_ = X.min()
        else:
            self.min_ = np.min(X)
        return self

    def transform(self, X):
        """Apply the shift transformation without densifying."""
        if self.min_ < 0:
            shift_value = -float(self.min_)
            if issparse(X):
                X = X.copy()  # ensure writable
                X.data += shift_value
            else:
                X = X + shift_value
        return X


class LabelEncoderManager:
    """
    Centralized management of label encoders.
    Ensures consistent encoding across all XGBoost models.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.encoder_path = os.path.join(output_dir, "shared_label_encoder.pkl")
        self._encoder: Optional[LabelEncoder] = None
    
    def fit(self, y):
        """Fit a new label encoder on the provided labels."""
        self._encoder = LabelEncoder()
        self._encoder.fit(y)
        return self
    
    def save(self):
        """Save the encoder to disk."""
        if self._encoder is None:
            raise ValueError("Encoder must be fitted before saving")
        joblib.dump(self._encoder, self.encoder_path)
        LOGGER.info(f"✓ Shared label encoder saved: {self.encoder_path}")
    
    def load(self) -> LabelEncoder:
        """Load the encoder from disk."""
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {self.encoder_path}")
        self._encoder = joblib.load(self.encoder_path)
        LOGGER.info(f"✓ Shared label encoder loaded: {self.encoder_path}")
        return self._encoder
    
    def transform(self, y):
        """Transform labels to encoded values."""
        if self._encoder is None:
            self._encoder = self.load()
        return self._encoder.transform(y)
    
    def inverse_transform(self, y):
        """Transform encoded values back to original labels."""
        if self._encoder is None:
            self._encoder = self.load()
        return self._encoder.inverse_transform(y)
    
    @property
    def encoder(self) -> LabelEncoder:
        """Get the encoder instance."""
        if self._encoder is None:
            self._encoder = self.load()
        return self._encoder


class DataLoader:
    """Handles loading of pre-split datasets in various formats."""
    
    SUPPORTED_FORMATS = ["npz", "json", "csv"]
    
    def __init__(self, splits_dir: str, format: str = "npz"):
        self.splits_dir = splits_dir
        self.format = format.lower()
        
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Expected one of: {', '.join(self.SUPPORTED_FORMATS)}"
            )
    
    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                    np.ndarray, np.ndarray, np.ndarray]:
        """
        Load train/val/test splits from disk.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        LOGGER.info(f"Loading pre-split datasets from: {self.splits_dir}")
        
        loader_map = {
            "npz": self._load_npz_split,
            "json": self._load_json_split,
            "csv": self._load_csv_split,
        }
        
        load_fn = loader_map[self.format]
        X_train, y_train = load_fn("train")
        X_val, y_val = load_fn("val")
        X_test, y_test = load_fn("test")
        
        LOGGER.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        LOGGER.info(f"Train class distribution: {Counter(y_train)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _load_npz_split(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a split stored as NPZ."""
        X = load_npz(os.path.join(self.splits_dir, f"X_{name}.npz"))
        y = joblib.load(os.path.join(self.splits_dir, f"y_{name}.npz"))
        return X, np.array(y)
    
    def _load_json_split(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a split stored as JSON."""
        with open(os.path.join(self.splits_dir, f"{name}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return np.array(data["X"]), np.array(data["y"])
    
    def _load_csv_split(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a split stored as CSV."""
        
        df = pd.read_csv(os.path.join(self.splits_dir, f"{name}.csv"))
        X = df.drop(columns=["label"]).to_numpy()
        y = df["label"].to_numpy()
        return X, y


class FeatureManager:
    """Manages feature subset extraction and slicing."""
    
    def __init__(self, vectorizer: ReviewVectorizer):
        self.vectorizer = vectorizer
        self.n_tfidf_features: Optional[int] = None
        self.n_opinion_features: Optional[int] = None
    
    def determine_feature_slices(self, X_samples: Dict[str, np.ndarray]):
        """
        Determine indices for TF-IDF and opinion features.
        Validates consistency across multiple data splits.
        
        Args:
            X_samples: Dictionary of sample matrices (e.g., {'train': X_train, 'test': X_test})
        """
        n_tfidf = len(self.vectorizer.tfidf.vocabulary_)
        
        # Check consistency across splits
        n_totals = {name: X.shape[1] for name, X in X_samples.items()}
        unique_dims = set(n_totals.values())
        
        if len(unique_dims) > 1:
            LOGGER.warning(f"Inconsistent feature dimensions across splits: {n_totals}")
        
        n_total = max(unique_dims)
        
        self.n_tfidf_features = n_tfidf
        self.n_opinion_features = n_total - n_tfidf
        
        LOGGER.info("Feature distribution:")
        LOGGER.info(f"  - TF-IDF (n-grams): {self.n_tfidf_features}")
        LOGGER.info(f"  - Opinion features: {self.n_opinion_features}")
        LOGGER.info(f"  - Total features: {n_total}")
    
    def get_subset(self, X, subset_type: str):
        """
        Extract specific feature subset from the combined matrix.
        
        Args:
            X: Full feature matrix
            subset_type: One of 'ngrams', 'opinion', or 'combined'
        
        Returns:
            Subset of X according to subset_type
        """
        if self.n_tfidf_features is None:
            raise ValueError("Must call determine_feature_slices() first")
        
        subset_map = {
            "ngrams": lambda: X[:, :self.n_tfidf_features],
            "opinion": lambda: X[:, self.n_tfidf_features:],
            "combined": lambda: X,
        }
        
        if subset_type not in subset_map:
            raise ValueError(
                f"Unknown subset type: {subset_type}. "
                f"Expected one of: {', '.join(subset_map.keys())}"
            )
        
        return subset_map[subset_type]()


class PipelineFactory:
    """
    Factory for creating model pipelines with appropriate preprocessing.
    
    Scaling strategies:
    - MultinomialNB: ShiftToPositive + MaxAbsScaler (requires non-negative values)
    - SGDClassifier: StandardScaler (benefits from standardization)
    - RandomForest: No scaling (tree-based, scale-invariant)
    - XGBoost: No scaling (tree-based, scale-invariant)
    """
    
    @staticmethod
    def create(model_name: str, base_model: BaseEstimator) -> Pipeline:
        """
        Create a pipeline with appropriate preprocessing for the given model.
        
        Args:
            model_name: Name of the model (e.g., 'MultinomialNB')
            base_model: Instantiated sklearn/xgboost model
        
        Returns:
            sklearn Pipeline with preprocessing and model
        """
        pipeline_configs = {
            "MultinomialNB": [
                ('shift', ShiftToPositive()),
                ('scaler', MaxAbsScaler()),
                ('classifier', base_model)
            ],
            "SGDClassifier": [
                ('scaler', StandardScaler(with_mean=False)),
                ('classifier', base_model)
            ],
            "RandomForest": [
                ('classifier', base_model)
            ],
            "XGBoost": [
                ('classifier', base_model)
            ],
        }
        
        config = pipeline_configs.get(model_name, [('classifier', base_model)])
        
        LOGGER.info(f"  → Pipeline: {' → '.join([step[0] for step in config])}")
        
        return Pipeline(config)


def get_model_instance(model_name: str, seed: int = 42):
    """
    Factory function to instantiate model objects.
    
    Args:
        model_name: Name of the model
        seed: Random seed for reproducibility
    
    Returns:
        Instantiated model object
    """
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb

    models = {
        'MultinomialNB': lambda: MultinomialNB(alpha=1.0),
        'SGDClassifier': lambda: SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=seed,
            n_jobs=-1
        ),
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=seed,
            n_jobs=-1
        ),
        'XGBoost': lambda: xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            eval_metric='logloss'
        ),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = models[model_name]()
    
    if model is None:
        raise ImportError(f"{model_name} is not available")
    
    return model