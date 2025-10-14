"""
Training utilities for LSTM models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

import sys
sys.path.append('..')
from config.config import *
from utils.lstm_utils import get_class_weights


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=LSTM_GRAD_CLIP):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for x_batch, labels, lengths_batch in train_loader:
        x_batch = x_batch.to(device)
        labels = labels.to(device)
        lengths_batch = lengths_batch.to(device)
        
        outputs = model(x_batch, lengths_batch)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)
    
    train_acc = train_correct / train_total
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for x_batch, labels, lengths_batch in val_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            lengths_batch = lengths_batch.to(device)
            
            outputs = model(x_batch, lengths_batch)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    return val_loss, val_acc


def get_predictions(model, data_loader, device):
    """Get model predictions and probabilities"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, labels, lengths_batch in data_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            lengths_batch = lengths_batch.to(device)
            
            outputs = model(x_batch, lengths_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def train_model(model, train_loader, val_loader, device, epochs=LSTM_EPOCHS,
                lr=LSTM_LEARNING_RATE, weight_decay=LSTM_WEIGHT_DECAY,
                class_weights=None, verbose=True):
    """
    Train LSTM model
    
    Parameters:
    -----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    device : torch.device
        Device to train on
    epochs : int
        Number of epochs
    lr : float
        Learning rate
    weight_decay : float
        Weight decay
    class_weights : Tensor, optional
        Class weights for loss
    verbose : bool
        Print training progress
    
    Returns:
    --------
    model : nn.Module
        Trained model
    history : dict
        Training history
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return model, history


def cross_validate(dataset, model_class, model_params, device, n_folds=LSTM_N_FOLDS,
                   epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=True):
    """
    Perform k-fold cross-validation
    
    Parameters:
    -----------
    dataset : SegmentTraceDataset
        Full dataset
    model_class : class
        Model class to instantiate
    model_params : dict
        Model parameters
    device : torch.device
        Device to train on
    n_folds : int
        Number of folds
    epochs : int
        Number of epochs per fold
    batch_size : int
        Batch size
    verbose : bool
        Print progress
    
    Returns:
    --------
    results : dict
        Cross-validation results
    """
    y = dataset.labels.numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=LSTM_RANDOM_SEED)
    
    all_preds = []
    all_probs = []
    all_labels = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold}/{n_folds}")
            print('='*60)
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)
        
        # Initialize model
        model = model_class(**model_params).to(device)
        
        # Get class weights
        y_train = y[train_idx]
        class_weights = get_class_weights(y_train, device)
        
        # Train
        model, _ = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, class_weights=class_weights, verbose=verbose
        )
        
        # Get predictions
        preds, probs, labels = get_predictions(model, val_loader, device)
        
        fold_acc = accuracy_score(labels, preds)
        fold_accuracies.append(fold_acc)
        
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels)
        
        if verbose:
            print(f"Fold {fold} Accuracy: {fold_acc:.4f}")
    
    results = {
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels),
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies)
    }
    
    return results