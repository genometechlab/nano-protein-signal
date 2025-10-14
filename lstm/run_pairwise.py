"""
Run pairwise amino acid classification
"""

import torch
import numpy as np
import json
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
from config.config import *
from lstm.models import TraceLSTMClassifier
from lstm.dataset import SegmentTraceDataset
from lstm.train import train_model, get_predictions
from utils.lstm_utils import (
    load_features_from_pickle, prepare_data, get_class_weights
)


def run_pairwise_classification(pickle_file, output_dir=None, input_size=LSTM_INPUT_SIZE,
                                epochs=PAIRWISE_EPOCHS, batch_size=LSTM_BATCH_SIZE,
                                test_size=PAIRWISE_TEST_SIZE, save_results=True):
    """
    Run pairwise classification for all amino acid pairs
    
    Parameters:
    -----------
    pickle_file : str
        Path to features pickle file
    output_dir : str
        Output directory
    input_size : int
        Number of features
    epochs : int
        Training epochs per pair
    batch_size : int
        Batch size
    test_size : float
        Test set fraction
    save_results : bool
        Save results
    
    Returns:
    --------
    results : dict
        Pairwise classification results
    """
    
    if output_dir is None:
        output_dir = Path(LSTM_OUTPUT_DIR) / "pairwise"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    X, y, metadata = load_features_from_pickle(pickle_file)
    
    unique_labels = sorted(set(y))
    print(f"Found {len(unique_labels)} unique classes")
    
    results = {}
    total_pairs = len(list(combinations(unique_labels, 2)))
    
    print(f"\nRunning pairwise classification for {total_pairs} pairs...")
    
    for pair_idx, (aa1, aa2) in enumerate(combinations(unique_labels, 2), 1):
        print(f"\n[{pair_idx}/{total_pairs}] Classifying {IDX_TO_AA[aa1]} vs {IDX_TO_AA[aa2]}")
        
        # Filter data for this pair
        indices = [i for i, label in enumerate(y) if label in (aa1, aa2)]
        
        if len(indices) < 10:
            print(f"  Skipping: insufficient data ({len(indices)} samples)")
            continue
        
        X_pair = [X[i] for i in indices]
        y_pair = [0 if y[i] == aa1 else 1 for i in indices]
        
        # Prepare data
        X_pad, y_tensor, lengths = prepare_data(X_pair, y_pair, normalize=True)
        
        # Train/test split
        X_tr, X_te, y_tr, y_te, len_tr, len_te = train_test_split(
            X_pad, y_tensor, lengths,
            test_size=test_size,
            stratify=y_tensor,
            random_state=LSTM_RANDOM_SEED
        )
        
        train_dataset = SegmentTraceDataset(X_tr, y_tr, len_tr)
        test_dataset = SegmentTraceDataset(X_te, y_te, len_te)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Model
        model = TraceLSTMClassifier(input_size=input_size, num_classes=2).to(device)
        
        # Class weights
        class_weights = get_class_weights(y_tr.numpy(), device)
        
        # Train
        model, _ = train_model(
            model, train_loader, test_loader, device,
            epochs=epochs, class_weights=class_weights, verbose=False
        )
        
        # Evaluate
        preds, probs, labels = get_predictions(model, test_loader, device)
        acc = accuracy_score(labels, preds)
        
        key = f"{aa1}_vs_{aa2}"
        results[key] = {
            "aa1": IDX_TO_AA[aa1],
            "aa2": IDX_TO_AA[aa2],
            "accuracy": float(acc),
            "n_samples": len(indices)
        }
        
        print(f"  Accuracy: {acc:.4f}")
    
    # Create accuracy matrix
    n_classes = len(unique_labels)
    acc_matrix = np.full((n_classes, n_classes), np.nan)
    
    for key, val in results.items():
        a1_str, a2_str = val['aa1'], val['aa2']
        i1 = AA_CLASS_MAP[a1_str]
        i2 = AA_CLASS_MAP[a2_str]
        acc_matrix[i1, i2] = acc_matrix[i2, i1] = val["accuracy"]
    
    # Plot heatmap
    mask = np.triu(np.ones_like(acc_matrix, dtype=bool))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        acc_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=[IDX_TO_AA[i] for i in unique_labels],
        yticklabels=[IDX_TO_AA[i] for i in unique_labels],
        mask=mask,
        cbar_kws={'label': 'Accuracy'}
    )
    plt.title("Pairwise Classification Accuracy")
    plt.xlabel("Amino Acid")
    plt.ylabel("Amino Acid")
    plt.tight_layout()
    
    if save_results:
        plt.savefig(output_dir / "pairwise_heatmap.png", dpi=300)
        print(f"\nSaved heatmap to {output_dir / 'pairwise_heatmap.png'}")
    
    plt.show()
    
    # Save results
    if save_results:
        with open(output_dir / "pairwise_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_dir}")
    
    print(f"\nAverage pairwise accuracy: {np.nanmean(acc_matrix):.4f}")
    
    return results, acc_matrix


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pairwise LSTM classification')
    parser.add_argument('input', type=str, help='Input features pickle file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--input-size', type=int, default=LSTM_INPUT_SIZE, help='Feature size')
    parser.add_argument('--epochs', type=int, default=PAIRWISE_EPOCHS, help='Epochs per pair')
    parser.add_argument('--batch-size', type=int, default=LSTM_BATCH_SIZE, help='Batch size')
    parser.add_argument('--test-size', type=float, default=PAIRWISE_TEST_SIZE, help='Test fraction')
    
    args = parser.parse_args()
    
    run_pairwise_classification(
        args.input,
        output_dir=args.output,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size
    )