"""
Run multi-group classification between amino acid groups
Supports 2-way (binary), 3-way, 4-way, 5-way, or any N-way classification
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import sys
sys.path.append('..')
from config.config import *
from lstm.models import TraceLSTMClassifier
from lstm.dataset import SegmentTraceDataset
from lstm.train import cross_validate
from utils.lstm_utils import (
    load_features_from_pickle, prepare_data, filter_by_classes
)


def relabel_multigroup(y, group_classes_list):
    """
    Convert labels to multi-group (0, 1, 2, ...)
    
    Parameters:
    -----------
    y : list
        Original labels
    group_classes_list : list of lists
        List of class groups, e.g., [[0,1,2], [3,4], [5,6,7,8]]
    
    Returns:
    --------
    y_multigroup : list
        Multi-group labels
    """
    y_multigroup = []
    for label in y:
        for group_idx, group_classes in enumerate(group_classes_list):
            if label in group_classes:
                y_multigroup.append(group_idx)
                break
        else:
            raise ValueError(f"Label {label} not found in any group")
    
    return y_multigroup


def run_multigroup_classification(pickle_file, group_names, output_dir=None,
                                  input_size=LSTM_INPUT_SIZE, epochs=LSTM_EPOCHS,
                                  n_folds=LSTM_N_FOLDS, batch_size=LSTM_BATCH_SIZE,
                                  save_results=True):
    """
    Run classification between multiple amino acid groups
    
    Parameters:
    -----------
    pickle_file : str
        Path to features pickle file
    group_names : list of str
        List of group names (e.g., ['positive', 'negative'] or ['very_small', 'small', 'medium'])
    output_dir : str
        Output directory
    input_size : int
        Number of features
    epochs : int
        Training epochs
    n_folds : int
        Cross-validation folds
    batch_size : int
        Batch size
    save_results : bool
        Save results
    
    Returns:
    --------
    results : dict
        Classification results
    """
    
    if not isinstance(group_names, list) or len(group_names) < 2:
        raise ValueError("group_names must be a list with at least 2 groups")
    
    num_groups = len(group_names)
    
    if output_dir is None:
        output_dir = Path(LSTM_OUTPUT_DIR) / "_vs_".join(group_names)
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get group definitions
    for group_name in group_names:
        if group_name not in AA_GROUPS:
            raise ValueError(f"Group '{group_name}' not found. Available groups: {list(AA_GROUPS.keys())}")
    
    group_classes_list = [AA_GROUPS[name] for name in group_names]
    
    print(f"\n{num_groups}-way classification:")
    for i, (name, classes) in enumerate(zip(group_names, group_classes_list)):
        print(f"  Group {i} ({name}): {[IDX_TO_AA[c] for c in classes]}")
    
    # Load data
    print("\nLoading data...")
    X, y, metadata = load_features_from_pickle(pickle_file)
    
    # Filter by target classes
    all_classes = []
    for group_classes in group_classes_list:
        all_classes.extend(group_classes)
    
    X_filtered, y_filtered, metadata_filtered = filter_by_classes(
        X, y, metadata, all_classes
    )
    
    # Relabel to multi-group
    y_multigroup = relabel_multigroup(y_filtered, group_classes_list)
    
    print(f"\nFiltered to {len(X_filtered)} samples")
    for i, name in enumerate(group_names):
        count = y_multigroup.count(i)
        print(f"  {name}: {count} samples")
    
    # Prepare data
    X_pad, y_tensor, lengths = prepare_data(X_filtered, y_multigroup, normalize=True)
    dataset = SegmentTraceDataset(X_pad, y_tensor, lengths)
    
    # Model parameters
    model_params = {
        'input_size': input_size,
        'num_classes': num_groups
    }
    
    # Cross-validation
    print(f"\nStarting {n_folds}-fold cross-validation...")
    cv_results = cross_validate(
        dataset, TraceLSTMClassifier, model_params, device,
        n_folds=n_folds, epochs=epochs, batch_size=batch_size
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"{' vs '.join([g.upper() for g in group_names])} RESULTS")
    print("="*60)
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in cv_results['fold_accuracies']]}")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        cv_results['labels'], cv_results['predictions'],
        target_names=group_names,
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(cv_results['labels'], cv_results['predictions'])
    
    fig_size = (max(6, num_groups * 2), max(5, num_groups * 1.8))
    plt.figure(figsize=fig_size)
    ConfusionMatrixDisplay(cm, display_labels=group_names).plot(
        cmap="Blues", ax=plt.gca()
    )
    plt.title(f"{' vs '.join(group_names)} ({n_folds}-Fold CV)")
    plt.tight_layout()
    
    if save_results:
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    
    plt.show()
    
    # ROC curves (for binary and multi-class)
    if num_groups == 2:
        # Binary ROC
        fpr, tpr, _ = roc_curve(cv_results['labels'], cv_results['probabilities'][:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {' vs '.join(group_names)}")
        plt.legend()
        plt.tight_layout()
        
        if save_results:
            plt.savefig(output_dir / "roc_curve.png", dpi=300)
        
        plt.show()
        
        auc_scores = {'binary_auc': float(roc_auc)}
    
    else:
        # Multi-class ROC (one-vs-rest)
        y_bin = label_binarize(cv_results['labels'], classes=list(range(num_groups)))
        
        plt.figure(figsize=(10, 8))
        auc_scores = {}
        
        for i, name in enumerate(group_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], cv_results['probabilities'][:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[name] = float(roc_auc)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves: {' vs '.join(group_names)}")
        plt.legend()
        plt.tight_layout()
        
        if save_results:
            plt.savefig(output_dir / "roc_curves_multiclass.png", dpi=300)
        
        plt.show()
    
    # Per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for name, acc in zip(group_names, class_accuracies):
        print(f"  {name}: {acc:.4f}")
    
    # Save results
    if save_results:
        results_dict = {
            'groups': group_names,
            'num_groups': num_groups,
            'group_classes': {name: [IDX_TO_AA[i] for i in classes] 
                             for name, classes in zip(group_names, group_classes_list)},
            'fold_accuracies': [float(acc) for acc in cv_results['fold_accuracies']],
            'mean_accuracy': float(cv_results['mean_accuracy']),
            'std_accuracy': float(cv_results['std_accuracy']),
            'per_class_accuracy': {name: float(acc) for name, acc in zip(group_names, class_accuracies)},
            'auc_scores': auc_scores,
            'num_samples': len(X_filtered),
            'samples_per_group': {name: y_multigroup.count(i) for i, name in enumerate(group_names)}
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    return cv_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-group LSTM classification')
    parser.add_argument('input', type=str, help='Input features pickle file')
    parser.add_argument('groups', type=str, nargs='+', 
                       help=f'Group names (2 or more): {list(AA_GROUPS.keys())}')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--input-size', type=int, default=LSTM_INPUT_SIZE, help='Feature size')
    parser.add_argument('--epochs', type=int, default=LSTM_EPOCHS, help='Number of epochs')
    parser.add_argument('--folds', type=int, default=LSTM_N_FOLDS, help='Number of CV folds')
    parser.add_argument('--batch-size', type=int, default=LSTM_BATCH_SIZE, help='Batch size')
    
    args = parser.parse_args()
    
    run_multigroup_classification(
        args.input,
        args.groups,
        output_dir=args.output,
        input_size=args.input_size,
        epochs=args.epochs,
        n_folds=args.folds,
        batch_size=args.batch_size
    )