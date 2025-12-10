"""
Module for Classification with Static Embeddings and Neural Networks
-----------------------------------------------

This script allows training and evaluating feedforward and recurrent neural
networks using static word embeddings (Word2Vec, GloVe, FastText) on the
BoardGeekGames Corpus.
It supports hyperparameter tuning, early stopping, and saves detailed
training summaries and evaluation artifacts.

Usage example:
    python src/scripts/pln_p3_7462_02_e1.py \
        --embedding_type w2v \
        --model_type fnn \
        --hidden_dims 256 128 \
        --dropout 0.3 \
        --batch_size 32 \
        --epochs 50 \
        --lr 0.001 \
        --patience 10 \
        --tune_hyperparams \
"""

import os
import argparse
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

from src.bgg_corpus.config import (
    SPLITS_DIR_EMB_W2V, SPLITS_DIR_EMB_GLOVE, SPLITS_DIR_EMB_FASTTEXT,
    MODELS_DIR_EMB_W2V, MODELS_DIR_EMB_GLOVE, MODELS_DIR_EMB_FASTTEXT,
    VECTORS_DIR_EMB_W2V, VECTORS_DIR_EMB_GLOVE, VECTORS_DIR_EMB_FASTTEXT,
)
from src.bgg_corpus.resources import LOGGER
from src.modeling import (
    ReviewDataset, FeedforwardNN, RecurrentNN, 
    train_epoch, evaluate, plot_confusion_matrix, 
    plot_training_history, HyperparameterTunerEmbeddings
)

# -------------------------------
# Utility Functions
# -------------------------------
def get_paths_for_embedding_type(embedding_type):
    """Get appropriate paths based on embedding type"""
    paths = {
        'w2v': (SPLITS_DIR_EMB_W2V, MODELS_DIR_EMB_W2V, VECTORS_DIR_EMB_W2V),
        'glove': (SPLITS_DIR_EMB_GLOVE, MODELS_DIR_EMB_GLOVE, VECTORS_DIR_EMB_GLOVE),
        'fasttext': (SPLITS_DIR_EMB_FASTTEXT, MODELS_DIR_EMB_FASTTEXT, VECTORS_DIR_EMB_FASTTEXT)
    }
    
    if embedding_type not in paths:
        raise ValueError(f"Invalid embedding type: {embedding_type}. Choose from: {list(paths.keys())}")
    
    return paths[embedding_type]

def save_training_summary(output_path, config, splits_info, tuning_results, final_results):
    """Save comprehensive training summary to text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Timestamp
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Embedding Type: {config['embedding_type'].upper()}\n")
        f.write(f"Model Type: {config['model_type'].upper()}\n\n")
        
        # Configuration
        f.write("-" * 80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Splits Information
        f.write("-" * 80 + "\n")
        f.write("DATASET SPLITS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training samples: {splits_info['train_size']}\n")
        f.write(f"Validation samples: {splits_info['val_size']}\n")
        f.write(f"Test samples: {splits_info['test_size']}\n")
        f.write(f"Total samples: {splits_info['total_size']}\n")
        f.write(f"Number of classes: {splits_info['num_classes']}\n")
        f.write(f"Class labels: {splits_info['class_labels']}\n")
        f.write(f"Input dimension: {splits_info['input_dim']}\n\n")
        
        # Hyperparameter Tuning Results
        if config['tune_hyperparams'] and tuning_results:
            f.write("-" * 80 + "\n")
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total configurations tested: {len(tuning_results)}\n\n")
            
            # Sort by validation accuracy
            sorted_results = sorted(tuning_results, key=lambda x: x['val_acc'], reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"\nConfiguration #{i}:\n")
                f.write(f"  Parameters: {result['params']}\n")
                f.write(f"  Validation Loss: {result['val_loss']:.4f}\n")
                f.write(f"  Validation Accuracy: {result['val_acc']:.4f}\n")
            
            f.write(f"\n\nBest Configuration:\n")
            f.write(f"  Parameters: {sorted_results[0]['params']}\n")
            f.write(f"  Validation Accuracy: {sorted_results[0]['val_acc']:.4f}\n\n")
        
        # Final Model Results
        f.write("-" * 80 + "\n")
        f.write("FINAL MODEL EVALUATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Loss: {final_results['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {final_results['test_acc']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(final_results['classification_report'])
        f.write("\n\n")
        
        # Training History
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
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")

# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="P3 E1: Classification with Static Embeddings and Neural Networks")
    
    # Embedding and paths
    parser.add_argument("--embedding_type", type=str, required=True, 
                        choices=['w2v', 'glove', 'fasttext'],
                        help="Type of static embedding (w2v, glove, fasttext)")
    
    # Model configuration
    parser.add_argument("--model_type", choices=["fnn", "rnn", "lstm", "gru"], default="fnn",
                        help="Type of model to train")
    
    # FNN parameters
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256, 128],
                        help="Hidden layer dimensions (FNN)")
    
    # RNN parameters
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension (RNN)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="NNumber of recurrent layers")
    parser.add_argument("--bidirectional", action='store_true',
                        help="Use bidirectional RNN")
    
    # Training parameters
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Hyperparameter tuning
    parser.add_argument("--tune_hyperparams", action='store_true',
                        help="Perform hyperparameter tuning")
    parser.add_argument("--tuning_epochs", type=int, default=30,
                        help="Epochs for each configuration during tuning")
    parser.add_argument("--max_trials", type=int, default=None,
                        help="Maximum number of configurations to try")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # Get appropriate paths
    splits_dir, models_dir, vectors_dir = get_paths_for_embedding_type(args.embedding_type)
    LOGGER.info(f"Using splits from: {splits_dir}")
    LOGGER.info(f"Saving models to: {models_dir}")
    LOGGER.info(f"Using vectors (dense matrices) from: {vectors_dir}")
    
    # -------------------------------
    # 1. Load splits
    # -------------------------------
    LOGGER.info("Loading splits...")
    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(splits_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(splits_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(splits_dir, "y_test.npy"))
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Debugging info
    #''' 
    print("Train embeddings mean:", np.mean(X_train, axis=0)[:10])
    print("Train embeddings std:", np.std(X_train, axis=0)[:10])
    print("Number of zero vectors:", np.sum(np.all(X_train==0, axis=1)))
    # '''
    
    # Load label mappings
    train_labels = np.load(os.path.join(vectors_dir, "labels.npy"))
    unique_labels = sorted(np.unique(train_labels))
    num_classes = len(unique_labels)
    label2idx = {label: i for i, label in enumerate(unique_labels)}
    idx2label = {i: label for label, i in label2idx.items()}
    input_dim = X_train.shape[1]
    
    # Split labels into indices
    y_train_idx = np.array([label2idx[y] for y in y_train], dtype=np.int64)
    y_val_idx   = np.array([label2idx[y] for y in y_val], dtype=np.int64)
    y_test_idx  = np.array([label2idx[y] for y in y_test], dtype=np.int64)
    
    LOGGER.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    LOGGER.info(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    
    # Splits info for summary
    splits_info = {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'total_size': len(X_train) + len(X_val) + len(X_test),
        'num_classes': num_classes,
        'class_labels': unique_labels,
        'input_dim': input_dim
    }
    
    # -------------------------------
    # 2. Create Datasets
    # -------------------------------
    train_dataset = ReviewDataset(X_train, y_train_idx)
    val_dataset = ReviewDataset(X_val, y_val_idx)
    test_dataset = ReviewDataset(X_test, y_test_idx)
    
    # -------------------------------
    # 3. Hyperparameter Tuning
    # -------------------------------
    tuning_results = None
    best_params = None
    
    if args.tune_hyperparams:
        LOGGER.info("Starting hyperparameter tuning...")
        tuner = HyperparameterTunerEmbeddings(args.model_type, input_dim, num_classes, device)
        (best_params, best_model), tuning_results = tuner.tune(
            train_dataset, val_dataset, 
            epochs=args.tuning_epochs,
            patience=args.patience,
            max_trials=args.max_trials
        )
        LOGGER.info(f"Best hyperparameters found: {best_params}")
        
        # Update args with best parameters
        if args.model_type == "fnn":
            args.hidden_dims = best_params['hidden_dims']
            args.dropout = best_params['dropout']
            args.lr = best_params['lr']
            args.batch_size = best_params['batch_size']
        else:
            args.hidden_dim = best_params['hidden_dim']
            args.num_layers = best_params['num_layers']
            args.bidirectional = best_params['bidirectional']
            args.dropout = best_params['dropout']
            args.lr = best_params['lr']
            args.batch_size = best_params['batch_size']
    
    # -------------------------------
    # 4. Create DataLoaders with final batch size
    # -------------------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # -------------------------------
    # 5. Create final model
    # -------------------------------
    if args.model_type == "fnn":
        model = FeedforwardNN(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            num_classes=num_classes,
            dropout=args.dropout
        )
        model_name = f"FNN_{'_'.join(map(str, args.hidden_dims))}"
    else:
        model = RecurrentNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            rnn_type=args.model_type,
            bidirectional=args.bidirectional,
            dropout=args.dropout
        )
        bi_str = "bi" if args.bidirectional else "uni"
        model_name = f"{args.model_type.upper()}_{bi_str}_{args.hidden_dim}h_{args.num_layers}l"
    
    model = model.to(device)
    LOGGER.info(f"Model: {model_name}")
    LOGGER.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # -------------------------------
    # 6. Training configuration
    # -------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    
    # -------------------------------
    # 7. Training loop
    # -------------------------------
    LOGGER.info("Starting training...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create output directory
    output_dir = os.path.join(models_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.pt")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        LOGGER.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
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
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_acc = accuracy_score(test_labels, test_preds)
    
    LOGGER.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Classification report
    target_names = [idx2label[i] for i in range(num_classes)]
    report = classification_report(test_labels, test_preds, target_names=target_names)
    LOGGER.info(f"\nClassification Report:\n{report}")
    
    # -------------------------------
    # 9. Save results
    # -------------------------------
    # Create evaluation subdirectory
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save confusion matrix
    plot_confusion_matrix(test_labels, test_preds, target_names,
                         os.path.join(eval_dir, "confusion_matrix.png"))
    
    # Save training history
    plot_training_history(history, os.path.join(eval_dir, "training_history.png"))
    
    # Prepare final results
    final_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'classification_report': report,
        'history': history
    }
    
    # Save comprehensive training summary
    summary_path = os.path.join(output_dir, "training_summary.txt")
    save_training_summary(
        summary_path,
        config=vars(args),
        splits_info=splits_info,
        tuning_results=tuning_results,
        final_results=final_results
    )
    LOGGER.info(f"Training summary saved to {summary_path}")
    
    # Save results pickle
    results = {
        'model_type': args.model_type,
        'model_name': model_name,
        'embedding_type': args.embedding_type,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'label_mapping': label2idx,
        'history': history,
        'config': vars(args),
        'tuning_results': tuning_results
    }
    joblib.dump(results, os.path.join(output_dir, "results.pkl"))
    
    LOGGER.info(f"All results saved to {output_dir}")
    LOGGER.info(f"Evaluation artifacts saved to {eval_dir}")

if __name__ == "__main__":
    main()