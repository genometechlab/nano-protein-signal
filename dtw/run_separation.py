"""
Run pairwise separation analysis for DBA centroids
"""

import numpy as np
import json
from pathlib import Path
import time

import sys
sys.path.append('..')
from config.config import *
from dtw.preprocessing import load_and_preprocess_data
from dtw.barycenter import build_centroids, save_centroids
from dtw.separation_analysis import (
    compute_pairwise_separation,
    get_separation_statistics,
    get_best_worst_pairs
)


def run_separation_analysis(pickle_file, output_dir=None, fixed_seg_len=FIXED_SEG_LEN,
                           target_aas=None, min_segments=None, max_segments=None,
                           min_length=None, max_length=None, use_zscore=True,
                           save_results=True):
    """
    Run pairwise separation analysis
    
    Parameters:
    -----------
    pickle_file : str
        Path to segmented pickle file
    output_dir : str
        Output directory
    fixed_seg_len : int
        Fixed segment length for DBA
    target_aas : list, optional
        Specific amino acids
    min_segments : int, optional
        Minimum segments per trace
    max_segments : int, optional
        Maximum segments per trace
    min_length : int, optional
        Minimum trace length
    max_length : int, optional
        Maximum trace length
    use_zscore : bool
        Apply z-score normalization
    save_results : bool
        Save results
    
    Returns:
    --------
    results : dict
        Separation analysis results
    """
    
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(DTW_OUTPUT_DIR) / "separation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DTW Pairwise Separation Analysis")
    print("="*70)
    print(f"Input file: {pickle_file}")
    print(f"Output directory: {output_dir}")
    print(f"Z-score normalization: {use_zscore}")
    print("="*70)
    
    # Load and preprocess
    data_by_aa, AA_LIST = load_and_preprocess_data(
        pickle_file, target_aas, min_segments, max_segments,
        min_length, max_length
    )
    
    # Build centroids
    centroids = build_centroids(data_by_aa, AA_LIST, fixed_seg_len)
    
    if save_results:
        save_centroids(centroids, output_dir / "centroids.json")
    
    # Compute separation
    results = compute_pairwise_separation(data_by_aa, centroids, AA_LIST)
    
    # Save matrices
    if save_results:
        np.save(output_dir / "cost_separation.npy", results['cost_separation'])
        np.save(output_dir / "dev_separation.npy", results['dev_separation'])
        np.save(output_dir / "combined_separation.npy", results['combined_separation'])
        np.save(output_dir / "cost_raw.npy", results['cost_raw'])
        np.save(output_dir / "dev_raw.npy", results['dev_raw'])
        
        with open(output_dir / "intra_distances.json", "w") as f:
            json.dump({
                'cost_intra': results['cost_intra'],
                'dev_intra': results['dev_intra']
            }, f, indent=2)
    
    # Compute statistics
    print("\n" + "="*70)
    print("SEPARATION STATISTICS")
    print("="*70)
    
    for metric_name, matrix_key in [('Deviation', 'dev_separation'),
                                     ('DTW Cost', 'cost_separation'),
                                     ('Combined', 'combined_separation')]:
        stats = get_separation_statistics(results[matrix_key], AA_LIST)
        print(f"\n{metric_name}:")
        print(f"  Mean:        {stats['mean']:.3f}")
        print(f"  Median:      {stats['median']:.3f}")
        print(f"  Min:         {stats['min']:.3f}")
        print(f"  Max:         {stats['max']:.3f}")
        print(f"  Std Dev:     {stats['std']:.3f}")
        print(f"  % > 1.0:     {stats['percent_above_1']:.1f}%")
    
    # Best and worst pairs (using deviation as primary metric)
    print("\n" + "-"*70)
    best_pairs, worst_pairs = get_best_worst_pairs(
        results['dev_separation'], AA_LIST, n_pairs=5
    )
    
    print("Best separated pairs (Deviation):")
    for pair in best_pairs:
        print(f"  Centroid {pair['centroid']} vs Traces {pair['trace']}: {pair['separation']:.3f}")
    
    print("\nWorst separated pairs (Deviation):")
    for pair in worst_pairs:
        print(f"  Centroid {pair['centroid']} vs Traces {pair['trace']}: {pair['separation']:.3f}")
    
    # Save summary
    if save_results:
        summary = {
            'statistics': {
                'deviation': get_separation_statistics(results['dev_separation'], AA_LIST),
                'cost': get_separation_statistics(results['cost_separation'], AA_LIST),
                'combined': get_separation_statistics(results['combined_separation'], AA_LIST)
            },
            'best_pairs_deviation': best_pairs,
            'worst_pairs_deviation': worst_pairs,
            'amino_acids': AA_LIST
        }
        
        with open(output_dir / "separation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DTW pairwise separation analysis')
    parser.add_argument('input', type=str, help='Input segmented pickle file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--seg-len', type=int, default=FIXED_SEG_LEN,
                       help='Fixed segment length for DBA')
    parser.add_argument('--aas', type=str, nargs='+', default=None,
                       help='Specific amino acids')
    parser.add_argument('--min-segments', type=int, default=None)
    parser.add_argument('--max-segments', type=int, default=None)
    parser.add_argument('--min-length', type=int, default=None)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--no-zscore', action='store_true',
                       help='Skip z-score normalization')
    
    args = parser.parse_args()
    
    run_separation_analysis(
        args.input,
        output_dir=args.output,
        fixed_seg_len=args.seg_len,
        target_aas=args.aas,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        min_length=args.min_length,
        max_length=args.max_length,
        use_zscore=not args.no_zscore
    )