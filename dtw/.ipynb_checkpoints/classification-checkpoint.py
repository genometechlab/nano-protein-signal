"""
DTW-based classification and grid search
"""

import numpy as np
from typing import List, Tuple
from tslearn.metrics import dtw as ts_dtw
from sklearn.metrics import accuracy_score, confusion_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def seg_cost(a, b):
    """DTW cost between two segments"""
    raw = ts_dtw(np.asarray(a).reshape(-1, 1), np.asarray(b).reshape(-1, 1))
    return raw / ((len(a) + len(b)) / 2)


def cost_matrix(A: List[np.ndarray], B: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise cost matrix"""
    m, n = len(A), len(B)
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            C[i, j] = seg_cost(A[i], B[j])
    return C


def dtw_mean_and_dev(C: np.ndarray) -> Tuple[float, float]:
    """Compute DTW alignment cost and path deviation"""
    r, c = C.shape
    acc = np.full((r, c), np.inf)
    acc[0, 0] = C[0, 0]
    
    # Forward pass
    for i in range(r):
        for j in range(c):
            if i == j == 0:
                continue
            acc[i, j] = C[i, j] + min(
                acc[i-1, j] if i else np.inf,
                acc[i, j-1] if j else np.inf,
                acc[i-1, j-1] if i and j else np.inf
            )
    
    # Backtrack for deviation
    i, j = r-1, c-1
    path_len = 1
    dev_sum = 0
    
    while i or j:
        dev_sum += abs(i - j)
        opts = []
        if i and j:
            opts.append((acc[i-1, j-1], i-1, j-1))
        if i:
            opts.append((acc[i-1, j], i-1, j))
        if j:
            opts.append((acc[i, j-1], i, j-1))
        _, i, j = min(opts, key=lambda x: x[0])
        path_len += 1
    
    return acc[-1, -1] / path_len, dev_sum / path_len


def classify_trace_batch(args):
    """Classify batch of traces"""
    traces, centroids, AA_LIST, alpha, beta = args
    results = []
    
    for true_aa, trace in traces:
        best_score, best_class = float("inf"), None
        for cand_aa in AA_LIST:
            C = cost_matrix([np.array(s) for s in centroids[cand_aa]], trace)
            cost, dev = dtw_mean_and_dev(C)
            score = alpha * cost + beta * dev
            if score < best_score:
                best_score, best_class = score, cand_aa
        results.append((true_aa, best_class))
    
    return results


def evaluate_grid_point(args):
    """Evaluate single grid point"""
    alpha, beta, all_traces, centroids, AA_LIST = args
    
    batch_size = max(10, len(all_traces) // mp.cpu_count())
    batches = [all_traces[i:i+batch_size] for i in range(0, len(all_traces), batch_size)]
    
    true_labels, pred_labels = [], []
    
    for batch in batches:
        batch_results = classify_trace_batch((batch, centroids, AA_LIST, alpha, beta))
        for true_aa, pred_aa in batch_results:
            true_labels.append(true_aa)
            pred_labels.append(pred_aa)
    
    acc = accuracy_score(true_labels, pred_labels)
    return alpha, beta, acc, true_labels, pred_labels


def grid_search_optimization(norm_by_aa, centroids, AA_LIST, 
                             coarse_grid, fine_grid=None):
    """
    Grid search for optimal alpha/beta parameters
    
    Parameters:
    -----------
    norm_by_aa : dict
        Normalized traces by AA
    centroids : dict
        DBA centroids
    AA_LIST : list
        List of amino acids
    coarse_grid : list of tuples
        Coarse grid (alpha, beta) pairs
    fine_grid : list of tuples, optional
        Fine grid for refinement
    
    Returns:
    --------
    best_params : tuple
        (alpha, beta, true_labels, pred_labels)
    best_acc : float
        Best accuracy
    """
    print("\nStarting grid search optimization...")
    
    # Prepare traces
    all_traces = []
    for aa in AA_LIST:
        for trace in norm_by_aa[aa]:
            all_traces.append((aa, trace))
    
    print(f"Total traces for classification: {len(all_traces)}")
    
    # Coarse grid search
    print(f"Coarse grid search with {len(coarse_grid)} points...")
    best_acc = 0
    best_params = None
    
    n_processes = min(mp.cpu_count(), len(coarse_grid))
    print(f"Using {n_processes} processes")
    
    grid_args = [(alpha, beta, all_traces, centroids, AA_LIST) 
                 for alpha, beta in coarse_grid]
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(evaluate_grid_point, args) for args in grid_args]
        
        for future in as_completed(futures):
            alpha, beta, acc, true_labels, pred_labels = future.result()
            print(f"  alpha={alpha:.2f} beta={beta:.2f} acc={acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_params = (alpha, beta, true_labels, pred_labels)
    
    # Fine grid search if provided
    if fine_grid is not None and best_params is not None:
        best_alpha = best_params[0]
        print(f"\nFine grid search around alpha={best_alpha:.2f}...")
        
        alpha_range = np.linspace(max(0, best_alpha - 0.15), min(1, best_alpha + 0.15), 11)
        fine_grid_points = [(a, 1.0 - a) for a in alpha_range]
        
        fine_grid_args = [(alpha, beta, all_traces, centroids, AA_LIST) 
                         for alpha, beta in fine_grid_points]
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = [executor.submit(evaluate_grid_point, args) for args in fine_grid_args]
            
            for future in as_completed(futures):
                alpha, beta, acc, true_labels, pred_labels = future.result()
                print(f"  alpha={alpha:.3f} beta={beta:.3f} acc={acc:.4f}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_params = (alpha, beta, true_labels, pred_labels)
    
    return best_params, best_acc