"""
Utility functions for DTW alignment visualization
"""

import numpy as np
from typing import List, Tuple, Sequence, Union
from tslearn.metrics import dtw as ts_dtw

ArrayLike = Union[Sequence[float], np.ndarray]


def dtw_matrix(s: Sequence[ArrayLike], t: Sequence[ArrayLike], 
               cost_mat: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute DTW alignment path from cost matrix
    
    Parameters:
    -----------
    s : sequence of arrays
        First sequence (barycenter segments)
    t : sequence of arrays
        Second sequence (test segments)
    cost_mat : np.ndarray
        Pairwise cost matrix
    
    Returns:
    --------
    dp : np.ndarray
        Accumulated cost matrix
    path : list of tuples
        Optimal alignment path
    """
    n, m = cost_mat.shape
    dp = np.full((n+1, m+1), np.inf)
    dp[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i, j] = cost_mat[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    
    # Backtrack
    i, j = n, m
    path = [(i-1, j-1)]
    while i > 1 or j > 1:
        opts = [dp[i-1, j-1], dp[i-1, j], dp[i, j-1]]
        step = int(np.argmin(opts))
        if step == 0:
            i, j = i-1, j-1
        elif step == 1:
            i -= 1
        else:
            j -= 1
        path.append((i-1, j-1))
    path.reverse()
    
    return dp, path


def zscore_trace(trace: List[np.ndarray]) -> List[np.ndarray]:
    """Z-score normalize trace"""
    allpts = np.concatenate(trace)
    mu = allpts.mean()
    sig = allpts.std() + 1e-8
    return [(seg - mu) / sig for seg in trace]


def zscore_both(trace1: List[np.ndarray], 
                trace2: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Z-score normalize both traces independently"""
    return zscore_trace(trace1), zscore_trace(trace2)


def compute_cost_matrix(segs1: List[np.ndarray], segs2: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise DTW cost matrix between segment lists
    
    Parameters:
    -----------
    segs1 : list of arrays
        First sequence segments
    segs2 : list of arrays
        Second sequence segments
    
    Returns:
    --------
    cost_mat : np.ndarray
        Cost matrix
    """
    n_i, n_t = len(segs1), len(segs2)
    cost_mat = np.zeros((n_i, n_t))
    
    for i in range(n_i):
        for j in range(n_t):
            cost_mat[i, j] = ts_dtw(
                segs1[i].reshape(-1, 1),
                segs2[j].reshape(-1, 1)
            )
    
    return cost_mat


def compute_alignment_metrics(dp: np.ndarray, path: List[Tuple[int, int]]) -> dict:
    """
    Compute alignment quality metrics
    
    Parameters:
    -----------
    dp : np.ndarray
        Accumulated cost matrix
    path : list of tuples
        Alignment path
    
    Returns:
    --------
    metrics : dict
        Dictionary of metrics
    """
    total_cost = dp[-1, -1]
    mean_cost = total_cost / len(path)
    deviation = np.mean([abs(i - j) for i, j in path])
    
    return {
        'total_cost': total_cost,
        'mean_cost': mean_cost,
        'deviation': deviation,
        'path_length': len(path)
    }
    
def compute_cost_matrix(segs1: List[np.ndarray], segs2: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise DTW cost matrix between segment lists
    
    Parameters:
    -----------
    segs1 : list of arrays
        First sequence segments
    segs2 : list of arrays
        Second sequence segments
    
    Returns:
    --------
    cost_mat : np.ndarray
        Cost matrix
    """
    from tslearn.metrics import dtw as ts_dtw
    
    n_i, n_t = len(segs1), len(segs2)
    cost_mat = np.zeros((n_i, n_t))
    
    for i in range(n_i):
        for j in range(n_t):
            cost_mat[i, j] = ts_dtw(
                segs1[i].reshape(-1, 1),
                segs2[j].reshape(-1, 1)
            )
    
    return cost_mat


def dtw_mean_and_dev(C: np.ndarray) -> Tuple[float, float]:
    """
    Compute DTW alignment cost and path deviation
    
    Parameters:
    -----------
    C : np.ndarray
        Cost matrix
    
    Returns:
    --------
    mean_cost : float
        Mean cost per aligned segment
    mean_deviation : float
        Mean deviation from diagonal
    """
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