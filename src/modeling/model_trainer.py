"""
Module to train a given model 
"""

import os
import joblib
from collections import Counter
from typing import List

from src.bgg_corpus.resources import LOGGER

from .model_utils import (DataLoader,FeatureManager, PipelineFactory,
    LabelEncoderManager,
    get_model_instance
)

class ModelTrainer:
    """Trains classification models using pre-computed vector representations."""
    
    MODEL_NAMES = ['MultinomialNB', 'SGDClassifier', 'RandomForest', 'XGBoost']
    FEATURE_SUBSETS = ['ngrams', 'opinion', 'combined']

    def __init__(self, vector_dir: str, output_dir: str, splits_dir: str,
                 split_format: str, seed: int = 42):
        self.vector_dir = vector_dir
        self.output_dir = output_dir
        self.splits_dir = splits_dir
        self.split_format = split_format
        self.seed = seed
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize managers
        self.data_loader = DataLoader(splits_dir, split_format)
        self.label_encoder_manager = LabelEncoderManager(output_dir)
        
        # Load vectorizer
        LOGGER.info(f"Loading vectorizer from: {vector_dir}")
        self.vectorizer = joblib.load(os.path.join(vector_dir, "bgg_vectorizer.pkl"))
        
        # Load metadata
        vectorizer_data = joblib.load(os.path.join(vector_dir, 'vectorizer_data.pkl'))
        self.categories = vectorizer_data['categories']
        
        LOGGER.info(f"Loaded {len(self.categories)} documents")
        LOGGER.info(f"Class distribution: {Counter(self.categories)}")
        
        # Initialize feature manager
        self.feature_manager = FeatureManager(self.vectorizer)
    
    def train_model(self, model_name: str, X, y, feature_subset: str):
        """
        Train a single model configuration.
        
        Args:
            model_name: Name of the model to train
            X: Full feature matrix
            y: Labels
            feature_subset: Type of features to use ('ngrams', 'opinion', 'combined')
        """
        print(f"\n{'='*70}")
        LOGGER.info(f"Training: {model_name} | Features: {feature_subset}")
        print(f"{'='*70}")
        
        # Extract feature subset
        X_subset = self.feature_manager.get_subset(X, feature_subset)
        LOGGER.info(f"Feature subset shape: {X_subset.shape}")
        
        # Handle XGBoost label encoding
        if model_name == "XGBoost":
            if not hasattr(self.label_encoder_manager, '_encoder') or \
               self.label_encoder_manager._encoder is None:
                LOGGER.info("Encoding class labels for XGBoost...")
                self.label_encoder_manager.fit(y)
                self.label_encoder_manager.save()
            
            y_encoded = self.label_encoder_manager.transform(y)
        else:
            y_encoded = y
        
        # Create model instance and pipeline
        base_model = get_model_instance(model_name, self.seed)
        pipeline = PipelineFactory.create(model_name, base_model)
        
        # Train pipeline
        LOGGER.info("Training pipeline...")
        pipeline.fit(X_subset, y_encoded)
        LOGGER.info("✓ Pipeline trained successfully")
        
        # Save pipeline
        model_filename = f"{model_name}_{feature_subset}.pkl"
        model_path = os.path.join(self.output_dir, model_filename)
        joblib.dump(pipeline, model_path)
        LOGGER.info(f"✓ Pipeline saved: {model_filename}")
        
        return pipeline
    
    def get_available_models(self) -> List[str]:
        """Get list of models that can be trained on this system."""
        available = []
        
        for model_name in self.MODEL_NAMES:
            try:
                get_model_instance(model_name, self.seed)
                available.append(model_name)
            except ImportError as e:
                LOGGER.warning(f"Model {model_name} not available: {e}")
        
        return available
    
    def run_training(self):
        """Main training pipeline: train all models with all feature subsets."""
        # Load data splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load_splits()
        
        # Use training set for model fitting
        X = X_train
        y = y_train
        
        # Determine feature slices
        self.feature_manager.determine_feature_slices({
            "train": X_train,
            "val": X_val,
            "test": X_test
        })
        
        # Get available models
        available_models = self.get_available_models()
        
        if not available_models:
            LOGGER.error("No models available for training!")
            return
        
        # Training statistics
        total_models = len(available_models) * len(self.FEATURE_SUBSETS)
        trained_count = 0
        
        LOGGER.info(f"\nStarting training: {len(available_models)} models × "
                   f"{len(self.FEATURE_SUBSETS)} feature subsets = {total_models} total configurations")
        print("="*70)
        
        # Train all combinations
        for model_name in available_models:
            for subset in self.FEATURE_SUBSETS:
                try:
                    self.train_model(model_name, X, y, subset)
                    trained_count += 1
                    LOGGER.info(f"Progress: {trained_count}/{total_models} models trained")
                except Exception as e:
                    LOGGER.error(f"✗ Error training {model_name} with {subset} features: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
        
        # Save training summary
        self._save_training_summary(trained_count, total_models, available_models)
        
        print("\n" + "="*70)
        LOGGER.info(f"TRAINING COMPLETED: {trained_count}/{total_models} models trained successfully")
        LOGGER.info(f"Models saved to: {self.output_dir}")
        print("="*70)
    
    def _save_training_summary(self, trained_count: int, total_models: int, 
                               available_models: List[str]):
        """Save a summary of the training process."""
        summary_path = os.path.join(self.output_dir, "training_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EXERCISE 4: MODEL TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total documents: {len(self.categories)}\n")
            f.write(f"Class distribution: {dict(Counter(self.categories))}\n")
            f.write(f"Total features: {self.feature_manager.n_tfidf_features + self.feature_manager.n_opinion_features}\n")
            f.write(f"  - TF-IDF features: {self.feature_manager.n_tfidf_features}\n")
            f.write(f"  - Opinion features: {self.feature_manager.n_opinion_features}\n\n")
            
            f.write(f"Available models: {', '.join(available_models)}\n")
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