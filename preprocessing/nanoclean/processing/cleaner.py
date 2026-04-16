"""
nanoclean.processing.cleaner
==================================
High-level signal cleaning orchestrator.

Runs a configurable multi-pass pipeline:

    Pass 1  →  Spike detection & removal
    Pass 2  →  Smoothing / denoising
    Pass 3  →  (optional) CWT + Huber refinement
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_dilation

from nanoclean.core.config import CleanerConfig
from nanoclean.core.trace import TraceData
from nanoclean.processing import filters

logger = logging.getLogger(__name__)


class SignalCleaner:
    """Configurable multi-pass signal cleaner.

    Parameters
    ----------
    config : CleanerConfig
        All tuneable parameters.  See :class:`CleanerConfig` for the full
        list with defaults.

    Example
    -------
    >>> from nanoclean import CleanerConfig, SignalCleaner, TraceData
    >>> cfg = CleanerConfig(first_pass_method="isolation",
    ...                     second_pass_method="tv",
    ...                     third_pass_cwt=True)
    >>> cleaner = SignalCleaner(cfg)
    >>> trace = TraceData(raw_signal=my_array)
    >>> result = cleaner.process(trace)
    >>> result.cleaned_signal
    """

    def __init__(self, config: CleanerConfig | None = None):
        self.config = config or CleanerConfig()
        self.params = self.config.to_filter_params()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, trace: TraceData) -> TraceData:
        """Run all configured passes and return the same TraceData (mutated).

        On unrecoverable error the cleaned_signal falls back to a simple
        moving-average smooth so downstream code always gets *something*.
        """
        try:
            signal = trace.raw_signal.astype(float)

            # Pass 1 — spike detection / removal
            signal = self._first_pass(signal)

            # Pass 2 — smoothing
            signal = self._second_pass(signal)

            # Pass 3 — optional CWT + Huber refinement
            if self.config.third_pass_cwt:
                signal = self._third_pass(signal)

            trace.cleaned_signal = signal
            self._record_metrics(trace)

        except Exception as e:
            logger.error("Signal cleaning failed, falling back to simple smooth: %s", e)
            trace.cleaned_signal = self._fallback_smooth(trace.raw_signal)

        return trace

    # ------------------------------------------------------------------
    # Pass 1 — Spike detection & removal
    # ------------------------------------------------------------------

    def _first_pass(self, signal: np.ndarray) -> np.ndarray:
        method = self.config.first_pass_method

        if method == "cwt_huber":
            return self._cwt_huber(signal)
        if method == "hampel":
            return self._hampel(signal)
        if method == "ransac":
            return self._ransac(signal)
        if method == "isolation":
            return self._isolation(signal)

        # Should never happen after config validation, but just in case.
        raise ValueError(f"Unknown first_pass_method: {method}")

    def _cwt_huber(self, signal: np.ndarray) -> np.ndarray:
        mask = filters.detect_spikes_cwt(
            signal,
            self.params["sampling_rate"],
            "mexh",
            self.params["threshold_factor"],
        )
        mask = binary_dilation(mask, structure=np.ones(self.params["dilation_size"]))
        return filters.remove_spikes_huber(
            signal,
            mask,
            self.params["window_size"],
            self.params["epsilon"],
            self.params["alpha"],
        )

    def _hampel(self, signal: np.ndarray) -> np.ndarray:
        cleaned, _ = filters.hampel_filter_numba(
            signal,
            self.params["window_size"],
            self.params["n_sigmas"],
        )
        return cleaned

    def _ransac(self, signal: np.ndarray) -> np.ndarray:
        cleaned, _ = filters.detect_and_remove_spikes_ransac(
            signal,
            window_size=self.params["window_size"],
            min_samples=0.5,
        )
        return cleaned

    def _isolation(self, signal: np.ndarray) -> np.ndarray:
        cleaned, _ = filters.detect_and_remove_spikes_isolation(
            signal,
            contamination=self.params["contamination"],
            window_size=self.params["window_size"],
        )
        return cleaned

    # ------------------------------------------------------------------
    # Pass 2 — Smoothing
    # ------------------------------------------------------------------

    def _second_pass(self, signal: np.ndarray) -> np.ndarray:
        method = self.config.second_pass_method

        if method == "none":
            return signal
        if method == "lowpass":
            return filters.apply_lowpass(
                signal,
                self.params["sampling_rate"],
                self.params["cutoff_freq"],
                self.params["filter_type"],
                self.params["filter_order"],
            )
        if method == "bilateral":
            return filters.bilateral_filter_numba(
                signal,
                self.params["spatial_sigma"],
                self.params["range_sigma"],
            )
        if method == "tv":
            return filters.tv_denoise_numba(
                signal,
                self.params["weight"],
                self.config.tv_iterations,
            )
        if method == "kalman":
            return filters.kalman_filter_numba(
                signal,
                self.params["process_variance"],
                self.params["measurement_variance"],
            )
        if method == "wavelet":
            return filters.apply_wavelet_denoising(signal)

        raise ValueError(f"Unknown second_pass_method: {method}")

    # ------------------------------------------------------------------
    # Pass 3 — CWT + Huber refinement
    # ------------------------------------------------------------------

    def _third_pass(self, signal: np.ndarray) -> np.ndarray:
        mask = filters.detect_spikes_cwt(
            signal,
            self.params["sampling_rate"],
            "mexh",
            self.params["third_pass_threshold_factor"],
        )
        mask = binary_dilation(
            mask,
            structure=np.ones(self.params["third_pass_dilation_size"]),
        )
        return filters.remove_spikes_huber(
            signal,
            mask,
            self.params["third_pass_window_size"],
            self.params["third_pass_epsilon"],
            self.params["alpha"],
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _record_metrics(self, trace: TraceData) -> None:
        raw_std = np.std(np.diff(trace.raw_signal))
        cln_std = np.std(np.diff(trace.cleaned_signal))
        reduction = (1.0 - cln_std / raw_std) * 100.0 if raw_std > 0 else 0.0

        trace.metadata["noise_reduction"] = round(reduction, 2)
        trace.metadata["original_std"] = float(raw_std)
        trace.metadata["cleaned_std"] = float(cln_std)

        method = f"{self.config.first_pass_method}+{self.config.second_pass_method}"
        if self.config.third_pass_cwt:
            method += "+cwt_huber"
        trace.metadata["processing_method"] = method

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
        kernel = np.ones(window) / window
        smoothed = np.convolve(signal, kernel, mode="same")
        half = window // 2
        smoothed[:half] = signal[:half]
        smoothed[-half:] = signal[-half:]
        return smoothed
