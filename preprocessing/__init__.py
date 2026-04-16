"""
nanoclean
==============
Modular multi-pass signal cleaning pipeline for nanopore trace data.

Quick start:
    from nanoclean import clean_signal, load_traces

    # Clean a raw numpy array
    cleaned = clean_signal(raw_signal)

    # Load and clean from files
    traces = load_traces("data.fast5")   # or "data.json"
    results = clean_signal(traces[0]["signal"])
"""

__version__ = "1.0.0"

from nanoclean.core.config import CleanerConfig
from nanoclean.core.trace import TraceData
from nanoclean.processing.cleaner import SignalCleaner
from nanoclean.processing.batch import BatchCleaner
from nanoclean.io.loader import (
    load_traces,
    load_fast5,
    load_json,
    save_results,
    load_results,
    validate_trace_data,
    batch_process_files,
)

# Convenience function
def clean_signal(
    signal,
    first_pass: str = "isolation",
    second_pass: str = "tv",
    third_pass: bool = True,
    **kwargs,
):
    """Clean a signal array with sensible defaults.

    Parameters
    ----------
    signal : array-like
        Raw signal data.
    first_pass : str
        Spike detection method: 'cwt_huber', 'hampel', 'ransac', 'isolation'.
    second_pass : str
        Smoothing method: 'lowpass', 'bilateral', 'tv', 'kalman', 'wavelet', 'none'.
    third_pass : bool
        Apply CWT+Huber refinement pass after smoothing.
    **kwargs
        Override any parameter in CleanerConfig (e.g. contamination=0.05).

    Returns
    -------
    numpy.ndarray
        Cleaned signal.
    """
    import numpy as np

    config = CleanerConfig(
        first_pass_method=first_pass,
        second_pass_method=second_pass,
        third_pass_cwt=third_pass,
        **kwargs,
    )
    cleaner = SignalCleaner(config)
    trace = TraceData(raw_signal=np.asarray(signal, dtype=float))
    result = cleaner.process(trace)
    return result.cleaned_signal


__all__ = [
    "clean_signal",
    "CleanerConfig",
    "TraceData",
    "SignalCleaner",
    "BatchCleaner",
    "load_traces",
    "load_fast5",
    "load_json",
    "save_results",
    "load_results",
    "validate_trace_data",
    "batch_process_files",
]