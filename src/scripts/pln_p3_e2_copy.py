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
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datetime import datetime


SPLITS_DIR_BERT = r"C:\Users\TrendingPC\Documents\Ciencia e Ingeniería de Datos\4to año\PLN\Prácticas\BoardGeekGames-Corpus\data\processed\datasets\bert"
MODELS_DIR_BERT = r"C:\Users\TrendingPC\Documents\Ciencia e Ingeniería de Datos\4to año\PLN\Prácticas\BoardGeekGames-Corpus\data\processed\models\bert"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

from itertools import product


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
            num_labels=self.num_classes,
            hidden_dropout_prob=0.2
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
    
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Genera y guarda matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"Confusion matrix saved to {output_path}")

def plot_training_history(history, output_path):
    """Generates and saves training plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"Training history saved to {output_path}")


import torch
from torch.utils.data import Dataset

class BERTReviewDataset(Dataset):
    """Dataset for BERT tokenized reviews"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = torch.LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

from tqdm import tqdm
# -------------------------------
# Training and Evaluation Functions
# -------------------------------
def train_epoch_bert(model, dataloader, criterion, optimizer, scheduler, device):
    """Train one epoch for BERT model"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), correct / total

def evaluate_bert(model, dataloader, criterion, device):
    """Evaluate BERT model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)

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
    X_train = joblib.load(os.path.join(splits_dir, "train_texts.pkl"))
    X_val = joblib.load(os.path.join(splits_dir, "val_texts.pkl"))
    X_test = joblib.load(os.path.join(splits_dir, "test_texts.pkl"))
    
    y_train = joblib.load(os.path.join(splits_dir, "y_train.pkl"))
    y_val = joblib.load(os.path.join(splits_dir, "y_val.pkl"))
    y_test = joblib.load(os.path.join(splits_dir, "y_test.pkl"))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P3 E2: Sentiment Classification with BERT Fine-tuning"
    )    
    # Model configuration
    # Read BERT configuration Uses same defaults as 
    # https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#bertconfig
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Pretrained model name (e.g., bert-base-uncased, roberta-base)")
    parser.add_argument("--max_length", type=int, default=64,
                       help="Maximum sequence length")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Intermediate size of the feed-forward layer")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="Dropout probability for hidden layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help="Dropout probability for attention layers")
    parser.add_argument("--max_position_embeddings", type=int, default=512, help="Maximum number of tokens per sequence")
    parser.add_argument("--type_vocab_size", type=int, default=2, help="Vocabulary size for token types")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Standard deviation of the truncated_normal_initializer for initializing all weight matrices")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="Epsilon used by the layer normalization layers")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Padding token ID")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Use gradient checkpointing to save memory")
    
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5,
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
    parser.add_argument("--max_trials", type=int, default=5,
                       help="Maximum number of configurations to try")
    
    # Paths
    parser.add_argument("--splits_dir", type=str, default=SPLITS_DIR_BERT,
                       help="Directory containing BERT splits")
    parser.add_argument("--output_dir", type=str, default=MODELS_DIR_BERT,
                       help="Output directory for BERT models")
    
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
    
    LOGGER.info(f"Loading data from: {args.splits_dir}")
    LOGGER.info(f"Saving models to: {args.output_dir}")
    
    # -------------------------------
    # 1. Load data
    # -------------------------------
    LOGGER.info("Loading text data...")
    train_texts, val_texts, test_texts, y_train, y_val, y_test = load_text_data(args.splits_dir)
    
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

        def safe_tune(*args_tune, **kwargs_tune):
            torch.cuda.empty_cache()
            return tuner.tune(*args_tune, **kwargs_tune)

        (best_params, best_model), tuning_results = safe_tune(
            train_texts, y_train_idx, val_texts, y_val_idx,
            tokenizer,
            epochs=args.tuning_epochs,
            patience=args.patience,
            max_trials=args.max_trials,
            default_batch_size=8,
            default_max_length=128
        )

        LOGGER.info(f"Best hyperparameters found: {best_params}")
        
        # Update args with best hyperparameters
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
    config = BertConfig(
        vocab_size=30522,  # usualmente no lo cambias
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=args.initializer_range,
        layer_norm_eps=args.layer_norm_eps,
        gradient_checkpointing=args.gradient_checkpointing,
        num_labels=num_classes # IMPORTANT: To adapt to our number of classes
    )
    
    LOGGER.info(f"Creating model: {args.model_name}")
    
    # Create pretrained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config
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
    
    # Usual loss for classification
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
    model_output_dir = os.path.join(args.output_dir, model_name)
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