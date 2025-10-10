"""
Feature extraction from segmented traces
"""

import numpy as np
import pickle
from scipy.stats import skew, kurtosis

import sys
sys.path.append('..')
from config.config import AA_CLASS_MAP


def extract_segment_features(segment):
    """
    Extract statistical features from a single segment
    
    Parameters:
    -----------
    segment : array-like
        Segment signal
    
    Returns:
    --------
    features : list
        Feature vector [skew, kurtosis, std, max, min, median, mean, length]
    """
    seg = np.asarray(segment)
    
    if len(seg) < 2:
        return None
    
    features = [
        skew(seg),
        kurtosis(seg),
        np.std(seg),
        np.max(seg),
        np.min(seg),
        np.median(seg),
        np.mean(seg),
        len(seg)
    ]
    
    return features


def extract_features_from_file(input_pickle, output_pickle=None):
    """
    Extract features from segmented data file
    
    Parameters:
    -----------
    input_pickle : str
        Path to segmented data pickle file
    output_pickle : str
        Path to save features pickle file
    
    Returns:
    --------
    featured_data : list
        List of dicts with features added
    """
    
    with open(input_pickle, "rb") as f:
        data = pickle.load(f)
    
    featured_data = []
    
    for entry in data:
        segments = entry.get("segments", [])
        
        features = []
        for seg in segments:
            feat = extract_segment_features(seg)
            if feat is not None:
                features.append(feat)
        
        if not features:
            continue
        
        entry["features"] = np.array(features)
        featured_data.append(entry)
    
    if output_pickle:
        with open(output_pickle, "wb") as f:
            pickle.dump(featured_data, f)
        print(f"Saved {len(featured_data)} entries with features to {output_pickle}")
    
    return featured_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: extract_features.py <input_pickle> [output_pickle]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.pkl', '_features.pkl')
    
    extract_features_from_file(input_file, output_file)