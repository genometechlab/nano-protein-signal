"""
DTW Barycenter Averaging for amino acid templates
"""

import numpy as np
import json
import time
from tslearn.barycenters import dtw_barycenter_averaging

import sys
sys.path.append('..')
from dtw.preprocessing import interp_segment


def build_centroids(norm_by_aa, AA_LIST, fixed_seg_len=80):
    """
    Build DBA centroids for each amino acid
    
    Parameters:
    -----------
    norm_by_aa : dict
        Normalized traces by amino acid
    AA_LIST : list
        List of amino acids
    fixed_seg_len : int
        Fixed length for segment interpolation
    
    Returns:
    --------
    centroids : dict
        Centroid for each amino acid
    """
    print("\nBuilding DBA centroids...")
    centroids = {}
    
    for aa in AA_LIST:
        traces = norm_by_aa[aa]
        start_time = time.time()
        
        interpolated_traces = []
        for tr in traces:
            segs = [interp_segment(seg, fixed_seg_len) for seg in tr]
            interpolated_traces.append(np.stack(segs))
        
        data = np.stack(interpolated_traces)
        centroid = dtw_barycenter_averaging(data, barycenter_size=data.shape[1])
        centroids[aa] = [centroid[i].tolist() for i in range(len(centroid))]
        
        elapsed = time.time() - start_time
        print(f"  {aa}: centroid built ({len(traces)} traces) in {elapsed:.1f}s")
    
    return centroids


def save_centroids(centroids, output_path):
    """
    Save centroids to JSON file
    
    Parameters:
    -----------
    centroids : dict
        Centroid dictionary
    output_path : str
        Output JSON path
    """
    with open(output_path, "w") as f:
        json.dump(centroids, f, indent=2)
    print(f"Centroids saved to: {output_path}")


def load_centroids(centroid_path):
    """
    Load centroids from JSON file
    
    Parameters:
    -----------
    centroid_path : str
        Path to centroid JSON
    
    Returns:
    --------
    centroids : dict
        Loaded centroids
    """
    with open(centroid_path, "r") as f:
        centroids = json.load(f)
    return centroids