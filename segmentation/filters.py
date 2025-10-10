"""
Signal filtering functions for nanopore data
"""

import numpy as np
from scipy.signal import bessel, filtfilt


def apply_bessel_filter(signal, order=1, cutoff=1500, sampling_rate=3012):
    """
    Apply low-pass Bessel filter to signal
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    order : int
        Filter order
    cutoff : float
        Cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    
    Returns:
    --------
    filtered_signal : np.ndarray
        Filtered signal
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = bessel(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)