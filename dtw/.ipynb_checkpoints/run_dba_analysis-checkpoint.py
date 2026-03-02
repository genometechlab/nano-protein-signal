"""
Main script for DTW DBA analysis
"""

import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix
import multiprocessing as mp

import sys
sys.path.append('..')
from config.config import *
from dtw.preprocessing import load_and_preprocess_data
from dtw.barycenter import build_centroids, save_centroids
from dtw.classification import grid_search_optimization


def run_dba_analysis(pickle_file, output_dir=None, fixed_seg_len=FIXED_SEG_LEN,
                     target_aas=DBA_TARGET_AAS, min_segments=DBA_MIN_SEGMENTS,
                     max_segments=DBA_MAX_SEGMENTS, min_length=DBA_MIN_TRACE_LENGTH,
                     max_length=DBA_MAX_TRACE_LENGTH, coarse_grid=COARSE_GRID,
                     fine_grid=True, save_results=True):
    """
    Run complete DBA analysis pipeline
    
    Parameters:
    -----------
    pickle_file : str
        Path to segmented pickle file
    output_dir : str
        Output directory for results
    fixed_seg_len : int
        Fixed segment length for interpolation
    target_aas : list, optional
        Specific amino acids to process
    min_segments : int, optional
        Minimum segments per trace
    max_segments : int, optional
        Maximum segments per trace
    min_length : int, optional
        Minimum trace length
    max_length : int, optional
        Maximum trace length
    coarse_grid : list
        Coarse grid for parameter search
    fine_grid : bool
        Whether to run fine grid search
    save_results : bool
        Whether to save results
    
    Returns:
    --------
    results : dict
        Analysis results including centroids and best parameters
    """
    
    start_time = time.time()
    
    if output_dir is None:
        output_dir = DTW_OUTPUT_DIR
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DTW DBA Analysis")
    print("="*60)
    print(f"Input file: {pickle_file}")
    print(f"Output directory: {output_dir}")
    print(f"Fixed segment length: {fixed_seg_len}")
    print(f"Target AAs: {target_aas if target_aas else 'All'}")
    print(f"Segment filters: min={min_segments}, max={max_segments}")
    print(f"Length filters: min={min_length}, max={max_length}")
    print("="*60)
    
    # Load and preprocess
    norm_by_aa, AA_LIST = load_and_preprocess_data(
        pickle_file, target_aas, min_segments, max_segments, min_length, max_length
    )
    
    # Build centroids
    centroids = build_centroids(norm_by_aa, AA_LIST, fixed_seg_len)
    
    # Save centroids
    if save_results:
        centroid_path = output_path / "dba_centroids.json"
        save_centroids(centroids, centroid_path)
    
    # Grid search
    best_params, best_acc = grid_search_optimization(
        norm_by_aa, centroids, AA_LIST, coarse_grid, fine_grid
    )
    
    if best_params is not None:
        best_alpha, best_beta, best_true, best_preds = best_params
        
        cm = confusion_matrix(best_true, best_preds, labels=AA_LIST)
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Best parameters: α={best_alpha:.3f}, β={best_beta:.3f}")
        print(f"Best accuracy: {best_acc:.4f} ({best_acc:.2%})")
        print("="*60)
        
        # Per-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-class accuracy:")
        for aa, acc in zip(AA_LIST, class_acc):
            print(f"  {aa}: {acc:.3f}")
        
        # Save results
        if save_results:
            results_dict = {
                "best_alpha": float(best_alpha),
                "best_beta": float(best_beta),
                "best_accuracy": float(best_acc),
                "confusion_matrix": cm.tolist(),
                "amino_acids": AA_LIST,
                "per_class_accuracy": {aa: float(acc) for aa, acc in zip(AA_LIST, class_acc)}
            }
            
            import json
            with open(output_path / "results.json", "w") as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"\nResults saved to: {output_dir}")
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.1f} seconds")
        
        return {
            "centroids": centroids,
            "best_alpha": best_alpha,
            "best_beta": best_beta,
            "best_accuracy": best_acc,
            "confusion_matrix": cm,
            "AA_LIST": AA_LIST
        }
    else:
        print("Grid search failed!")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DTW DBA analysis')
    parser.add_argument('input', type=str, help='Input segmented pickle file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--seg-len', type=int, default=FIXED_SEG_LEN, 
                       help='Fixed segment length for DBA')
    parser.add_argument('--aas', type=str, nargs='+', default=None,
                       help='Specific amino acids to process')
    parser.add_argument('--min-segments', type=int, default=None,
                       help='Minimum segments per trace')
    parser.add_argument('--max-segments', type=int, default=None,
                       help='Maximum segments per trace')
    parser.add_argument('--min-length', type=int, default=None,
                       help='Minimum trace length')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum trace length')
    parser.add_argument('--no-fine-grid', action='store_true',
                       help='Skip fine grid search')
    
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    
    run_dba_analysis(
        args.input,
        output_dir=args.output,
        fixed_seg_len=args.seg_len,
        target_aas=args.aas,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        min_length=args.min_length,
        max_length=args.max_length,
        fine_grid=not args.no_fine_grid
    )