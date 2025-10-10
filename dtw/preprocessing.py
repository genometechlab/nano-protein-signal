"""
Preprocessing utilities for DTW analysis
"""

import numpy as np
from typing import List, Union, Sequence

ArrayLike = Union[np.ndarray, Sequence[float]]


def zscore_segments(seglist: List[np.ndarray]) -> List[np.ndarray]:
    """
    Z-score normalize segments
    
    Parameters:
    -----------
    seglist : list of arrays
        List of segments
    
    Returns:
    --------
    normalized : list of arrays
        Normalized segments
    """
    x = np.concatenate(seglist)
    mu, sd = np.mean(x), np.std(x) + 1e-8
    return [(s - mu) / sd for s in seglist]


def interp_segment(seg: ArrayLike, target_length: int) -> np.ndarray:
    """
    Interpolate segment to fixed length
    
    Parameters:
    -----------
    seg : array-like
        Input segment
    target_length : int
        Target length for interpolation
    
    Returns:
    --------
    interpolated : np.ndarray
        Interpolated segment
    """
    old_idx = np.linspace(0, 1, len(seg))
    new_idx = np.linspace(0, 1, target_length)
    return np.interp(new_idx, old_idx, seg)


def filter_traces_by_criteria(traces, min_segments=None, max_segments=None,
                              min_length=None, max_length=None):
    """
    Filter traces based on segment count and length criteria
    
    Parameters:
    -----------
    traces : list
        List of traces (each trace is list of segments)
    min_segments : int, optional
        Minimum number of segments
    max_segments : int, optional
        Maximum number of segments
    min_length : int, optional
        Minimum total trace length
    max_length : int, optional
        Maximum total trace length
    
    Returns:
    --------
    filtered_traces : list
        Filtered traces
    """
    filtered = []
    
    for trace in traces:
        n_segments = len(trace)
        total_length = sum(len(seg) for seg in trace)
        
        # Check segment count
        if min_segments is not None and n_segments < min_segments:
            continue
        if max_segments is not None and n_segments > max_segments:
            continue
        
        # Check total length
        if min_length is not None and total_length < min_length:
            continue
        if max_length is not None and total_length > max_length:
            continue
        
        filtered.append(trace)
    
    return filtered


def load_and_preprocess_data(pickle_file, target_aas=None, min_segments=None,
                             max_segments=None, min_length=None, max_length=None):
    """
    Load and preprocess segmented data for DBA
    
    Parameters:
    -----------
    pickle_file : str
        Path to segmented pickle file
    target_aas : list of str, optional
        Specific amino acids to process (None for all)
    min_segments : int, optional
        Minimum segments per trace
    max_segments : int, optional
        Maximum segments per trace
    min_length : int, optional
        Minimum trace length
    max_length : int, optional
        Maximum trace length
    
    Returns:
    --------
    norm_by_aa : dict
        Normalized traces by amino acid
    AA_LIST : list
        List of amino acids
    """
    import pickle
    from collections import defaultdict
    
    print("Loading and preprocessing data...")
    
    with open(pickle_file, "rb") as f:
        entries = pickle.load(f)
    
    raw_by_aa = defaultdict(list)
    for e in entries:
        aa = e["variable_region"]
        segments = e.get("segments", e.get("cleaned_segments", []))
        raw_by_aa[aa].append(segments)
    
    # Filter by target AAs if specified
    if target_aas is not None:
        raw_by_aa = {aa: traces for aa, traces in raw_by_aa.items() if aa in target_aas}
    
    AA_LIST = sorted(raw_by_aa.keys())
    print(f"Processing {len(AA_LIST)} amino acids: {AA_LIST}")
    
    # Filter and normalize traces
    norm_by_aa = {}
    for aa, traces in raw_by_aa.items():
        # Filter traces
        filtered_traces = filter_traces_by_criteria(
            traces, min_segments, max_segments, min_length, max_length
        )
        
        print(f"  {aa}: {len(filtered_traces)}/{len(traces)} traces after filtering")
        
        # Normalize
        norm_by_aa[aa] = [zscore_segments(tr) for tr in filtered_traces]
    
    total_traces = sum(len(traces) for traces in norm_by_aa.values())
    print(f"Total traces for DBA: {total_traces}")
    
    return norm_by_aa, AA_LIST