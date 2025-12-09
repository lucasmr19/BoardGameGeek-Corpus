import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.bgg_corpus.resources import LOGGER

# -------------------------------
# Custom dataset PyTorch
# -------------------------------
class ReviewDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# -------------------------------
# Util functions for model components
# -------------------------------
def get_activation(name):
    """Return activation function by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Activation {name} not supported")

def get_normalization(name, dim):
    """Return normalization layer by name, or None."""
    name = name.lower()
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    elif name == "layernorm":
        return nn.LayerNorm(dim)
    elif name in ("none", None):
        return None
    else:
        raise ValueError(f"Normalization {name} not supported")

# -------------------------------
# Classification Models
# -------------------------------
class FeedforwardNN(nn.Module):
    """Feedforward Neural Network for classification"""
    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128],
        num_classes=3,
        dropout=0.3,
        activation="relu",
        normalization="batchnorm"
    ):
        super(FeedforwardNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            norm_layer = get_normalization(normalization, hidden_dim)
            if norm_layer:
                layers.append(norm_layer)
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class RecurrentNN(nn.Module):
    """Recurrent Neural Network (RNN/LSTM/GRU) for classification"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=3, 
                 rnn_type='lstm', bidirectional=True, dropout=0.3):
        super(RecurrentNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        
        # Select RNN type
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        else:  # basic rnn
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        
        # Classification layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        # Expand for RNN: (batch_size, seq_len=1, embedding_dim)
        x = x.unsqueeze(1)
        
        # RNN forward
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(x)
        else:
            output, hidden = self.rnn(x)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward of the last layer
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Classification
        return self.fc(hidden)

# -------------------------------
# Training and evaluation functions
# -------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for embeddings, labels in tqdm(dataloader, desc="Training", leave=False):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)

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