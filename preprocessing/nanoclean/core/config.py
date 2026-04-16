"""
signal_cleaner.core.config
===========================
Single source of truth for all pipeline parameters.

Every tuneable value lives here.  Parameters are grouped by the pipeline
stage they belong to so it's easy to find what you need.
"""

from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class CleanerConfig:
    """Complete configuration for the signal cleaning pipeline.

    Override any parameter at construction time::

        cfg = CleanerConfig(contamination=0.05, weight=0.2)
    """

    # ------------------------------------------------------------------
    # Pass 1 – Spike detection / removal
    # ------------------------------------------------------------------
    first_pass_method: str = "isolation"
    """One of 'cwt_huber', 'hampel', 'ransac', 'isolation'."""

    # Shared
    sampling_rate: float = 3012.0
    spike_window_size: int = 10
    """Window half-width for local spike operations."""

    # CWT + Huber
    threshold_factor: float = 0.15
    dilation_size: int = 3
    epsilon: float = 2.0
    alpha: float = 0.01

    # Hampel
    n_sigmas: float = 3.0

    # Isolation Forest
    contamination: float = 0.1

    # ------------------------------------------------------------------
    # Pass 2 – Smoothing
    # ------------------------------------------------------------------
    second_pass_method: str = "tv"
    """One of 'lowpass', 'bilateral', 'tv', 'kalman', 'wavelet', 'none'."""

    # Lowpass
    cutoff_freq: float = 1500.0
    filter_type: str = "bessel"
    filter_order: int = 2

    # Bilateral
    spatial_sigma: float = 2.0
    range_sigma: float = 5.0

    # TV denoising
    weight: float = 0.1
    tv_iterations: int = 100

    # Kalman
    process_variance: float = 1e-5
    measurement_variance: float = 0.01

    # Wavelet
    denoise_wavelet: str = "db4"
    denoise_level: Optional[int] = None

    # ------------------------------------------------------------------
    # Pass 3 – Optional CWT + Huber refinement
    # ------------------------------------------------------------------
    third_pass_cwt: bool = True
    third_pass_threshold_factor: float = 0.25
    third_pass_window_size: int = 5
    third_pass_epsilon: float = 1.0
    third_pass_dilation_size: int = 1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = "./output"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_filter_params(self) -> dict:
        """Export a flat dict compatible with the filter functions."""
        return {
            "sampling_rate": self.sampling_rate,
            "threshold_factor": self.threshold_factor,
            "window_size": self.spike_window_size,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "dilation_size": self.dilation_size,
            "cutoff_freq": self.cutoff_freq,
            "filter_type": self.filter_type,
            "filter_order": self.filter_order,
            "contamination": self.contamination,
            "n_sigmas": self.n_sigmas,
            "spatial_sigma": self.spatial_sigma,
            "range_sigma": self.range_sigma,
            "weight": self.weight,
            "process_variance": self.process_variance,
            "measurement_variance": self.measurement_variance,
            "denoise_wavelet": self.denoise_wavelet,
            "denoise_level": self.denoise_level,
            "third_pass_threshold_factor": self.third_pass_threshold_factor,
            "third_pass_window_size": self.third_pass_window_size,
            "third_pass_epsilon": self.third_pass_epsilon,
            "third_pass_dilation_size": self.third_pass_dilation_size,
        }

    def __post_init__(self):
        valid_first = {"cwt_huber", "hampel", "ransac", "isolation"}
        valid_second = {"lowpass", "bilateral", "tv", "kalman", "wavelet", "none"}
        if self.first_pass_method not in valid_first:
            raise ValueError(
                f"first_pass_method must be one of {valid_first}, "
                f"got '{self.first_pass_method}'"
            )
        if self.second_pass_method not in valid_second:
            raise ValueError(
                f"second_pass_method must be one of {valid_second}, "
                f"got '{self.second_pass_method}'"
            )
