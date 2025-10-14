"""
Utility functions for LSTM training and evaluation
"""

import torch
import numpy as np
import pickle
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight


def load_features_from_pickle(pickle_file, feature_key='features', label_key='label'):
    """
    Load features and labels from pickle file
    
    Parameters:
    -----------
    pickle_file : str
        Path to pickle file
    feature_key : str
        Key for features in pickle data
    label_key : str
        Key for labels in pickle data
    
    Returns:
    --------
    X : list of Tensors
        Feature tensors
    y : list
        Labels
    metadata : list
        Metadata for each sample
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    
    X = [torch.tensor(d[feature_key], dtype=torch.float32) for d in data]
    y = [d[label_key] for d in data]
    metadata = [d.get('metadata', '') for d in data]
    
    return X, y, metadata


def normalize_features(X, method='per_trace'):
    """
    Normalize feature tensors
    
    Parameters:
    -----------
    X : list of Tensors
        Feature tensors
    method : str
        Normalization method: 'per_trace', 'global', or 'none'
    
    Returns:
    --------
    X_norm : list of Tensors
        Normalized features
    """
    if method == 'per_trace':
        # Per-trace z-score normalization
        X_norm = [(x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-8) for x in X]
    elif method == 'global':
        # Global z-score normalization
        all_features = torch.cat(X, dim=0)
        mean = all_features.mean(0, keepdim=True)
        std = all_features.std(0, keepdim=True) + 1e-8
        X_norm = [(x - mean) / std for x in X]
    else:
        X_norm = X
    
    return X_norm


def prepare_data(X, y, normalize=True):
    """
    Prepare data for training
    
    Parameters:
    -----------
    X : list of Tensors
        Feature tensors
    y : list
        Labels
    normalize : bool
        Whether to normalize features
    
    Returns:
    --------
    X_pad : Tensor
        Padded feature tensor
    y_tensor : Tensor
        Label tensor
    lengths : Tensor
        Sequence lengths
    """
    if normalize:
        X = normalize_features(X, method='per_trace')
    
    lengths = torch.tensor([len(x) for x in X], dtype=torch.long)
    X_pad = pad_sequence(X, batch_first=True)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return X_pad, y_tensor, lengths


def filter_by_classes(X, y, metadata, target_classes):
    """
    Filter data by specific classes
    
    Parameters:
    -----------
    X : list
        Features
    y : list
        Labels
    metadata : list
        Metadata
    target_classes : list
        Classes to keep
    
    Returns:
    --------
    X_filtered : list
        Filtered features
    y_filtered : list
        Filtered labels
    metadata_filtered : list
        Filtered metadata
    """
    indices = [i for i, label in enumerate(y) if label in target_classes]
    X_filtered = [X[i] for i in indices]
    y_filtered = [y[i] for i in indices]
    metadata_filtered = [metadata[i] for i in indices]
    
    return X_filtered, y_filtered, metadata_filtered


def relabel_binary(y, positive_classes, negative_classes):
    """
    Convert labels to binary (0/1)
    
    Parameters:
    -----------
    y : list
        Original labels
    positive_classes : list
        Classes to label as 0
    negative_classes : list
        Classes to label as 1
    
    Returns:
    --------
    y_binary : list
        Binary labels
    """
    y_binary = []
    for label in y:
        if label in positive_classes:
            y_binary.append(0)
        elif label in negative_classes:
            y_binary.append(1)
        else:
            raise ValueError(f"Label {label} not in positive or negative classes")
    
    return y_binary


def get_class_weights(y, device):
    """
    Compute balanced class weights
    
    Parameters:
    -----------
    y : array-like
        Labels
    device : torch.device
        Device for tensor
    
    Returns:
    --------
    weights : Tensor
        Class weights
    """
    y_np = np.array(y) if not isinstance(y, np.ndarray) else y
    classes = np.unique(y_np)
    weights = compute_class_weight('balanced', classes=classes, y=y_np)
    return torch.tensor(weights, dtype=torch.float32).to(device)