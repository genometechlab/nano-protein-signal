"""
Run multiclass amino acid classification
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import sys
sys.path.append('..')
from config.config import *
from lstm.models import TraceLSTMClassifier
from lstm.dataset import SegmentTraceDataset
from lstm.train import cross_validate
from utils.lstm_utils import load_features_from_pickle, prepare_data


def run_multiclass_classification(pickle_file, output_dir=None, input_size=LSTM_INPUT_SIZE,
                                  num_classes=20, epochs=LSTM_EPOCHS, n_folds=LSTM_N_FOLDS,
                                  batch_size=LSTM_BATCH_SIZE, save_results=True):
    """
    Run multiclass amino acid classification
    
    Parameters:
    -----------
    pickle_file : str
        Path to features pickle file
    output_dir : str
        Output directory for results
    input_size : int
        Number of features per segment
    num_classes : int
        Number of amino acid classes
    epochs : int
        Number of training epochs
    n_folds : int
        Number of cross-validation folds
    batch_size : int
        Batch size
    save_results : bool
        Whether to save results
    
    Returns:
    --------
    results : dict
        Classification results
    """
    
    if output_dir is None:
        output_dir = Path(LSTM_OUTPUT_DIR) / "multiclass"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    X, y, metadata = load_features_from_pickle(pickle_file)
    print(f"Loaded {len(X)} samples with {num_classes} classes")
    
    # Prepare data
    X_pad, y_tensor, lengths = prepare_data(X, y, normalize=True)
    dataset = SegmentTraceDataset(X_pad, y_tensor, lengths)
    
    # Model parameters
    model_params = {
        'input_size': input_size,
        'num_classes': num_classes
    }
    
    # Cross-validation
    print(f"\nStarting {n_folds}-fold cross-validation...")
    cv_results = cross_validate(
        dataset, TraceLSTMClassifier, model_params, device,
        n_folds=n_folds, epochs=epochs, batch_size=batch_size
    )
    
    # Print results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in cv_results['fold_accuracies']]}")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(cv_results['labels'], cv_results['predictions'], zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(cv_results['labels'], cv_results['predictions'])
    
    plt.figure(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=[IDX_TO_AA[i] for i in range(num_classes)]).plot(
        cmap="Blues", ax=plt.gca()
    )
    plt.title(f"Multiclass Confusion Matrix ({n_folds}-Fold CV)")
    plt.tight_layout()
    
    if save_results:
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
        print(f"\nSaved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    
    plt.show()
    
    # Save results
    if save_results:
        results_dict = {
            'fold_accuracies': [float(acc) for acc in cv_results['fold_accuracies']],
            'mean_accuracy': float(cv_results['mean_accuracy']),
            'std_accuracy': float(cv_results['std_accuracy']),
            'num_classes': num_classes,
            'num_samples': len(X)
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_dir}")
    
    return cv_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multiclass LSTM classification')
    parser.add_argument('input', type=str, help='Input features pickle file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--input-size', type=int, default=LSTM_INPUT_SIZE, help='Feature size')
    parser.add_argument('--num-classes', type=int, default=20, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=LSTM_EPOCHS, help='Number of epochs')
    parser.add_argument('--folds', type=int, default=LSTM_N_FOLDS, help='Number of CV folds')
    parser.add_argument('--batch-size', type=int, default=LSTM_BATCH_SIZE, help='Batch size')
    
    args = parser.parse_args()
    
    run_multiclass_classification(
        args.input,
        output_dir=args.output,
        input_size=args.input_size,
        num_classes=args.num_classes,
        epochs=args.epochs,
        n_folds=args.folds,
        batch_size=args.batch_size
    )