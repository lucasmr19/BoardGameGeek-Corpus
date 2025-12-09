"""
Module for Sentiment Classification with BERT Fine-tuning
---------------------------------------------------------

This script performs fine-tuning of BERT (or variants) for sentiment
classification on the BoardGameGeek Corpus.

Usage example:
    python src/scripts/pln_p3_7462_02_e2.py \
        --model_name bert-base-uncased \
        --batch_size 16 \
        --epochs 5 \
        --lr 2e-5 \
        --max_length 256 \
        --patience 3 \
        --tune_hyperparams
"""

import os
import argparse
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datetime import datetime

from src.bgg_corpus.config import (
    SPLITS_DIR_EMB_W2V, SPLITS_DIR_EMB_GLOVE, SPLITS_DIR_EMB_FASTTEXT,
    MODELS_DIR_EMB_W2V, MODELS_DIR_EMB_GLOVE, MODELS_DIR_EMB_FASTTEXT,
    VECTORS_DIR_EMB_W2V, VECTORS_DIR_EMB_GLOVE, VECTORS_DIR_EMB_FASTTEXT,
)
from src.bgg_corpus.resources import LOGGER
from src.modeling import (
    plot_confusion_matrix, 
    plot_training_history,
    BERTReviewDataset,
    train_epoch_bert,
    evaluate_bert,
    HyperparameterTunerBERT,
)

# -------------------------------
# Save Training Summary
# -------------------------------
def save_training_summary(output_path, config, splits_info, tuning_results, final_results):
    """Save comprehensive training summary to text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BERT FINE-TUNING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config['model_name']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("DATASET SPLITS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training samples: {splits_info['train_size']}\n")
        f.write(f"Validation samples: {splits_info['val_size']}\n")
        f.write(f"Test samples: {splits_info['test_size']}\n")
        f.write(f"Total samples: {splits_info['total_size']}\n")
        f.write(f"Number of classes: {splits_info['num_classes']}\n")
        f.write(f"Class labels: {splits_info['class_labels']}\n\n")
        
        if config['tune_hyperparams'] and tuning_results:
            f.write("-" * 80 + "\n")
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total configurations tested: {len(tuning_results)}\n\n")
            
            sorted_results = sorted(tuning_results, key=lambda x: x['val_acc'], reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"\nConfiguration #{i}:\n")
                f.write(f"  Parameters: {result['params']}\n")
                f.write(f"  Validation Loss: {result['val_loss']:.4f}\n")
                f.write(f"  Validation Accuracy: {result['val_acc']:.4f}\n")
            
            f.write(f"\n\nBest Configuration:\n")
            f.write(f"  Parameters: {sorted_results[0]['params']}\n")
            f.write(f"  Validation Accuracy: {sorted_results[0]['val_acc']:.4f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("FINAL MODEL EVALUATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Loss: {final_results['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {final_results['test_acc']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(final_results['classification_report'])
        f.write("\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("TRAINING HISTORY\n")
        f.write("-" * 80 + "\n")
        history = final_results['history']
        f.write(f"Total epochs: {len(history['train_loss'])}\n\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}\n")
        f.write("-" * 60 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<8} {history['train_loss'][i]:<12.4f} {history['train_acc'][i]:<12.4f} "
                   f"{history['val_loss'][i]:<12.4f} {history['val_acc'][i]:<12.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")

# -------------------------------
# Load raw text data
# -------------------------------
def load_text_data(splits_dir):
    """Load raw text data from splits directory"""
    # Assuming you have saved the raw texts during split creation
    # If not, you'll need to modify this to load from your corpus
    train_texts = joblib.load(os.path.join(splits_dir, "train_texts.pkl"))
    val_texts = joblib.load(os.path.join(splits_dir, "val_texts.pkl"))
    test_texts = joblib.load(os.path.join(splits_dir, "test_texts.pkl"))
    
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    y_val = np.load(os.path.join(splits_dir, "y_val.npy"))
    y_test = np.load(os.path.join(splits_dir, "y_test.npy"))
    
    return train_texts, val_texts, test_texts, y_train, y_val, y_test

# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P3 E2: Sentiment Classification with BERT Fine-tuning"
    )
    # Embedding and paths
    parser.add_argument("--embedding_type", type=str, required=True, 
                        choices=['w2v', 'glove', 'fasttext'],
                        help="Type of static embedding (w2v, glove, fasttext)")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Pretrained model name (e.g., bert-base-uncased, roberta-base)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Number of warmup steps for scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience")
    
    # Hyperparameter tuning
    parser.add_argument("--tune_hyperparams", action='store_true',
                       help="Perform hyperparameter tuning")
    parser.add_argument("--tuning_epochs", type=int, default=3,
                       help="Epochs for each configuration during tuning")
    parser.add_argument("--max_trials", type=int, default=None,
                       help="Maximum number of configurations to try")
    
    # Paths
    parser.add_argument("--splits_dir", type=str, default=None,
                       help="Directory containing splits (defaults to SPLITS_DIR_EMB_W2V)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for models")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # Set paths
    splits_dir = args.splits_dir or SPLITS_DIR_EMB_W2V
    output_dir = args.output_dir or os.path.join(MODELS_DIR_EMB_W2V, "bert_models")
    
    LOGGER.info(f"Loading data from: {splits_dir}")
    LOGGER.info(f"Saving models to: {output_dir}")
    
    # -------------------------------
    # 1. Load data
    # -------------------------------
    LOGGER.info("Loading text data...")
    train_texts, val_texts, test_texts, y_train, y_val, y_test = load_text_data(splits_dir)
    
    # Create label mappings
    unique_labels = sorted(np.unique(y_train))
    num_classes = len(unique_labels)
    label2idx = {label: i for i, label in enumerate(unique_labels)}
    idx2label = {i: label for label, i in label2idx.items()}
    
    # Convert labels to indices
    y_train_idx = np.array([label2idx[y] for y in y_train], dtype=np.int64)
    y_val_idx = np.array([label2idx[y] for y in y_val], dtype=np.int64)
    y_test_idx = np.array([label2idx[y] for y in y_test], dtype=np.int64)
    
    LOGGER.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    LOGGER.info(f"Number of classes: {num_classes}, Labels: {unique_labels}")
    
    splits_info = {
        'train_size': len(train_texts),
        'val_size': len(val_texts),
        'test_size': len(test_texts),
        'total_size': len(train_texts) + len(val_texts) + len(test_texts),
        'num_classes': num_classes,
        'class_labels': unique_labels
    }
    
    # -------------------------------
    # 2. Load tokenizer
    # -------------------------------
    LOGGER.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # -------------------------------
    # 3. Hyperparameter Tuning (Optional)
    # -------------------------------
    tuning_results = None
    best_params = None
    
    if args.tune_hyperparams:
        LOGGER.info("Starting hyperparameter tuning...")
        tuner = HyperparameterTunerBERT(args.model_name, num_classes, device)
        (best_params, best_model), tuning_results = tuner.tune(
            train_texts, y_train_idx, val_texts, y_val_idx,
            tokenizer, 
            epochs=args.tuning_epochs,
            patience=args.patience,
            max_trials=args.max_trials
        )
        LOGGER.info(f"Best hyperparameters found: {best_params}")
        
        # Update args with best parameters
        args.lr = best_params['lr']
        args.batch_size = best_params['batch_size']
        args.max_length = best_params['max_length']
        args.warmup_steps = best_params['warmup_steps']
        args.weight_decay = best_params['weight_decay']
    
    # -------------------------------
    # 4. Create datasets with final parameters
    # -------------------------------
    train_dataset = BERTReviewDataset(train_texts, y_train_idx, tokenizer, args.max_length)
    val_dataset = BERTReviewDataset(val_texts, y_val_idx, tokenizer, args.max_length)
    test_dataset = BERTReviewDataset(test_texts, y_test_idx, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # -------------------------------
    # 5. Create final model
    # -------------------------------
    LOGGER.info(f"Creating model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes
    )
    model = model.to(device)
    
    LOGGER.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # -------------------------------
    # 6. Training configuration
    # -------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------
    # 7. Training loop
    # -------------------------------
    LOGGER.info("Starting training...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    model_name = args.model_name.replace('/', '_')
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    best_model_path = os.path.join(model_output_dir, "best_model")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_bert(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        val_loss, val_preds, val_labels = evaluate_bert(
            model, val_loader, criterion, device
        )
        val_acc = accuracy_score(val_labels, val_preds)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        LOGGER.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            LOGGER.info(f"Best model saved: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                LOGGER.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # -------------------------------
    # 8. Test evaluation
    # -------------------------------
    LOGGER.info("Evaluating on test set...")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    model = model.to(device)
    
    test_loss, test_preds, test_labels = evaluate_bert(
        model, test_loader, criterion, device
    )
    test_acc = accuracy_score(test_labels, test_preds)
    
    LOGGER.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    target_names = [idx2label[i] for i in range(num_classes)]
    report = classification_report(test_labels, test_preds, target_names=target_names)
    LOGGER.info(f"\nClassification Report:\n{report}")
    
    # -------------------------------
    # 9. Save results
    # -------------------------------
    eval_dir = os.path.join(model_output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    plot_confusion_matrix(
        test_labels, test_preds, target_names,
        os.path.join(eval_dir, "confusion_matrix.png")
    )
    
    plot_training_history(
        history,
        os.path.join(eval_dir, "training_history.png")
    )
    
    final_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'classification_report': report,
        'history': history
    }
    
    summary_path = os.path.join(model_output_dir, "training_summary.txt")
    save_training_summary(
        summary_path,
        config=vars(args),
        splits_info=splits_info,
        tuning_results=tuning_results,
        final_results=final_results
    )
    
    results = {
        'model_name': args.model_name,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'label_mapping': label2idx,
        'history': history,
        'config': vars(args),
        'tuning_results': tuning_results
    }
    joblib.dump(results, os.path.join(model_output_dir, "results.pkl"))
    
    LOGGER.info(f"All results saved to {model_output_dir}")

if __name__ == "__main__":
    main()