"""
PELT segmentation for nanopore protein signals
"""

import numpy as np
import pickle
from collections import defaultdict
from joblib import Parallel, delayed
import ruptures as rpt

import sys
sys.path.append('..')
from config.config import *
from segmentation.filters import apply_bessel_filter
from segmentation.cost_functions import CustomCost
from utils.data_loader import load_pastor_data


def segment_trace_pelt(signal, penalty=PELT_PENALTY, min_size=PELT_MIN_SIZE, 
                       scale=PELT_SCALE, cost_function='custom'):
    """
    Segment signal using PELT algorithm
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    penalty : float
        Penalty parameter for PELT
    min_size : int
        Minimum segment size
    scale : float
        Scale parameter for custom cost
    cost_function : str
        Cost function: 'custom', 'l1', 'l2', 'rbf'
    
    Returns:
    --------
    bkps : list
        Breakpoint indices
    """
    if cost_function == 'custom':
        custom_cost = CustomCost(scale=scale, min_size=min_size)
        algo = rpt.Pelt(custom_cost=custom_cost, min_size=min_size).fit(signal)
    else:
        algo = rpt.Pelt(model=cost_function, min_size=min_size).fit(signal)
    
    bkps = algo.predict(pen=penalty)
    
    # Ensure full coverage
    if bkps[0] != 0:
        bkps.insert(0, 0)
    if bkps[-1] != len(signal):
        bkps.append(len(signal))
    
    return bkps


def process_trace_pelt(pastor, aa, group, counter, raw_data, channels, run, 
                       filter_order=FILTER_ORDER, cutoff=CUTOFF_FREQUENCY, 
                       sampling_rate=SAMPLING_RATE, penalty=PELT_PENALTY,
                       min_size=PELT_MIN_SIZE, scale=PELT_SCALE,
                       cost_function='custom', min_length=MIN_LENGTH, 
                       max_length=MAX_LENGTH):
    """Process single trace with PELT segmentation"""
    
    idx = group[pastor.index(aa)]
    
    if idx not in raw_data:
        return None
    
    sig = np.asarray(raw_data[idx], dtype=float)
    
    # Length filtering (optional)
    if min_length is not None and max_length is not None:
        if not (min_length <= len(sig) <= max_length):
            return None
    
    try:
        # Filter signal
        filtered = apply_bessel_filter(sig, filter_order, cutoff, sampling_rate)
        
        # Segment
        bkps = segment_trace_pelt(filtered, penalty, min_size, scale, cost_function)
        
        # Extract segments
        segments = []
        for a1, a2 in zip(bkps[:-1], bkps[1:]):
            seg = filtered[a1:a2]
            if len(seg) >= 2:
                segments.append(seg.tolist())
        
        if not segments:
            return None
        
        return {
            "variable_region": aa,
            "label": AA_CLASS_MAP[aa],
            "segments": segments,
            "breakpoints": bkps,
            "metadata": f"{pastor}_{counter[(pastor, aa)]}",
            "df_index": idx,
            "channel": channels.get(idx),
            "run": run.get(idx)
        }
    
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return None


def run_pelt_segmentation(data_path=DATA_PATH, output_path=None, n_jobs=N_JOBS, **kwargs):
    """
    Run PELT segmentation on all traces
    
    Parameters:
    -----------
    data_path : str
        Path to input JSON file
    output_path : str
        Path to save output pickle file
    n_jobs : int
        Number of parallel jobs
    **kwargs : dict
        Additional parameters for segmentation
    """
    
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/pelt_segmented.pkl"
    
    # Load data
    pastor_groups, aa_info, raw_data, channels, run = load_pastor_data(data_path)
    
    # Build job list
    joblist = []
    count_dict = defaultdict(int)
    
    for pastor in PASTORS:
        for aa in pastor:
            for channel, groups in pastor_groups[pastor].items():
                for group in groups:
                    idx = group[pastor.index(aa)]
                    if idx in raw_data:
                        count_dict[(pastor, aa)] += 1
                        joblist.append((pastor, aa, group, count_dict.copy()))
    
    print(f"Processing {len(joblist)} traces with PELT...")
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_trace_pelt)(p, a, g, c, raw_data, channels, run, **kwargs)
        for p, a, g, c in joblist
    )
    
    structured = [r for r in results if r is not None]
    
    # Save results
    with open(output_path, "wb") as f:
        pickle.dump(structured, f)
    
    print(f"Saved {len(structured)} segmented traces to {output_path}")
    
    return structured


if __name__ == "__main__":
    run_pelt_segmentation()