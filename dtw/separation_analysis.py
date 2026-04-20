"""
Pairwise separation analysis for DBA centroids
Computes how well each centroid separates its own class from other classes
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

import sys
sys.path.append('..')
from utils.dtw_utils import compute_cost_matrix, dtw_mean_and_dev


def compute_trace_to_centroid(trace, centroid):
    """
    Compute DTW cost and deviation between trace and centroid
    
    Parameters:
    -----------
    trace : list of arrays
        Segmented trace
    centroid : list of arrays
        Barycenter centroid
    
    Returns:
    --------
    cost : float
        Mean DTW cost per segment
    deviation : float
        Mean path deviation from diagonal
    """
    C = compute_cost_matrix([np.array(s) for s in centroid], trace)
    cost, dev = dtw_mean_and_dev(C)
    return cost, dev


def compute_pairwise_separation(data_by_aa, centroids, AA_LIST, metric='deviation'):
    """
    Compute pairwise separation ratios
    
    For each pair (centroid_i, traces_j):
    - intra_i = median distance from centroid_i to traces_i
    - inter_ij = median distance from centroid_i to traces_j  
    - separation[i,j] = inter_ij / intra_i
    
    Higher ratio = better separation (traces_j far from centroid_i)
    
    Parameters:
    -----------
    data_by_aa : dict
        Traces organized by amino acid
    centroids : dict
        Barycenter for each amino acid
    AA_LIST : list
        List of amino acids
    metric : str
        Metric to use: 'cost', 'deviation', or 'combined'
    
    Returns:
    --------
    results : dict
        Separation matrices and statistics
    """
    print("\nComputing pairwise separation...")
    
    n = len(AA_LIST)
    
    # Store raw distances
    cost_inter = np.zeros((n, n))
    dev_inter = np.zeros((n, n))
    cost_intra = {}
    dev_intra = {}
    
    # Compute all distances
    print("  Computing distances...")
    all_costs = defaultdict(list)
    all_devs = defaultdict(list)
    
    for i, cent_aa in enumerate(AA_LIST):
        print(f"    Centroid {cent_aa}...")
        for j, trace_aa in enumerate(AA_LIST):
            costs, devs = [], []
            
            for trace in data_by_aa[trace_aa]:
                c, d = compute_trace_to_centroid(trace, centroids[cent_aa])
                costs.append(c)
                devs.append(d)
            
            all_costs[(cent_aa, trace_aa)] = costs
            all_devs[(cent_aa, trace_aa)] = devs
            
            cost_inter[i, j] = np.median(costs)
            dev_inter[i, j] = np.median(devs)
    
    # Extract intra-class (diagonal)
    for i, aa in enumerate(AA_LIST):
        cost_intra[aa] = cost_inter[i, i]
        dev_intra[aa] = dev_inter[i, i]
    
    print("  Computing separation ratios...")
    
    cost_sep = np.zeros((n, n))
    dev_sep = np.zeros((n, n))
    combined_sep = np.zeros((n, n))
    
    for i, cent_aa in enumerate(AA_LIST):
        for j, trace_aa in enumerate(AA_LIST):
            # Separation = inter / intra (higher = better)
            if cost_intra[cent_aa] > 0:
                cost_sep[i, j] = cost_inter[i, j] / cost_intra[cent_aa]
            else:
                cost_sep[i, j] = 1.0
            
            if dev_intra[cent_aa] > 0:
                dev_sep[i, j] = dev_inter[i, j] / dev_intra[cent_aa]
            else:
                dev_sep[i, j] = 1.0
            
            # Combined metric
            combined_inter = cost_inter[i, j] + dev_inter[i, j]
            combined_intra = cost_intra[cent_aa] + dev_intra[cent_aa]
            if combined_intra > 0:
                combined_sep[i, j] = combined_inter / combined_intra
            else:
                combined_sep[i, j] = 1.0
    
    results = {
        'cost_separation': cost_sep,
        'dev_separation': dev_sep,
        'combined_separation': combined_sep,
        'cost_raw': cost_inter,
        'dev_raw': dev_inter,
        'cost_intra': cost_intra,
        'dev_intra': dev_intra,
        'all_costs': dict(all_costs),
        'all_devs': dict(all_devs)
    }
    
    return results


def get_separation_statistics(separation_matrix, AA_LIST):
    """
    Compute summary statistics for separation matrix
    
    Parameters:
    -----------
    separation_matrix : np.ndarray
        Separation ratio matrix
    AA_LIST : list
        Amino acid labels
    
    Returns:
    --------
    stats : dict
        Summary statistics
    """
    n = len(AA_LIST)
    
    # Get lower triangle (excluding diagonal)
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    vals = separation_matrix[mask]
    
    stats = {
        'mean': float(np.mean(vals)),
        'median': float(np.median(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'std': float(np.std(vals)),
        'percent_above_1': float(100 * np.mean(vals > 1))
    }
    
    return stats


def get_best_worst_pairs(separation_matrix, AA_LIST, n_pairs=5):
    """
    Find best and worst separated pairs
    
    Parameters:
    -----------
    separation_matrix : np.ndarray
        Separation ratio matrix
    AA_LIST : list
        Amino acid labels
    n_pairs : int
        Number of pairs to return
    
    Returns:
    --------
    best_pairs : list
        Best separated pairs
    worst_pairs : list
        Worst separated pairs
    """
    n = len(AA_LIST)
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    
    # Best pairs
    best_pairs = []
    temp_matrix = separation_matrix.copy()
    for _ in range(n_pairs):
        idx = np.unravel_index(
            np.argmax(temp_matrix * mask),
            temp_matrix.shape
        )
        if mask[idx]:
            best_pairs.append({
                'centroid': AA_LIST[idx[0]],
                'trace': AA_LIST[idx[1]],
                'separation': float(separation_matrix[idx])
            })
            temp_matrix[idx] = -np.inf
    
    # Worst pairs
    worst_pairs = []
    temp_matrix = separation_matrix.copy()
    temp_matrix[~mask] = np.inf
    
    for _ in range(n_pairs):
        idx = np.unravel_index(np.argmin(temp_matrix), temp_matrix.shape)
        if mask[idx] and temp_matrix[idx] < np.inf:
            worst_pairs.append({
                'centroid': AA_LIST[idx[0]],
                'trace': AA_LIST[idx[1]],
                'separation': float(separation_matrix[idx])
            })
            temp_matrix[idx] = np.inf
    
    return best_pairs, worst_pairs