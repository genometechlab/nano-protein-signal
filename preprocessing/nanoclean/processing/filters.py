"""
nanoclean.processing.filters
===================================
Low-level optimized filtering functions with Numba acceleration.

This module contains every atomic signal-processing operation used by the
pipeline.  Functions are stateless and operate on plain numpy arrays so they
can be used independently of the pipeline classes.
"""

from __future__ import annotations

import os

# Prevent OpenMP / MKL fork-safety issues when used with multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import threading
import warnings
from functools import lru_cache
from typing import Optional, Tuple

import numba as nb
import numpy as np
import pywt
from numba import njit, prange
from scipy.ndimage import binary_dilation
from scipy.signal import bessel, butter, sosfiltfilt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")


# =====================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =====================================================================


@njit(cache=True, fastmath=True)
def compute_mad(data: np.ndarray) -> float:
    """Median Absolute Deviation."""
    median = np.median(data)
    return np.median(np.abs(data - median))


@njit(cache=True, fastmath=True)
def compute_spike_energy_cwt(coeffs: np.ndarray) -> np.ndarray:
    """Spike energy from the first 5 CWT scale rows."""
    high_freq = coeffs[:5, :]
    return np.sum(np.abs(high_freq) ** 2, axis=0)


@njit(cache=True, fastmath=True, parallel=True)
def hampel_filter_numba(
    data: np.ndarray, window_size: int, n_sigmas: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated Hampel filter.

    Returns (cleaned_signal, boolean_spike_mask).
    """
    n = len(data)
    cleaned = data.copy()
    spike_mask = np.zeros(n, dtype=nb.boolean)
    k = 1.4826  # consistency constant for Gaussian

    for i in prange(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = data[start:end]
        median = np.median(window)
        mad = k * compute_mad(window)

        if np.abs(data[i] - median) > n_sigmas * mad:
            spike_mask[i] = True
            cleaned[i] = median

    return cleaned, spike_mask


@njit(cache=True, fastmath=True)
def tv_denoise_numba(
    data: np.ndarray, weight: float, n_iter: int = 100
) -> np.ndarray:
    """Total-Variation denoising (Rudin-Osher-Fatemi style)."""
    n = len(data)
    denoised = data.copy()
    lo = data.min() - 10.0
    hi = data.max() + 10.0

    for _ in range(n_iter):
        grad = np.zeros_like(denoised)
        grad[0] = denoised[1] - denoised[0]
        for i in range(1, n - 1):
            grad[i] = (denoised[i + 1] - denoised[i - 1]) / 2.0
        grad[n - 1] = denoised[n - 1] - denoised[n - 2]

        grad_norm = np.sqrt(grad * grad + 1e-10)
        ng = grad / grad_norm

        div = np.zeros_like(ng)
        div[0] = ng[1] - ng[0]
        for i in range(1, n - 1):
            div[i] = (ng[i + 1] - ng[i - 1]) / 2.0
        div[n - 1] = ng[n - 1] - ng[n - 2]

        denoised = denoised - weight * (denoised - data - div)
        for i in range(n):
            if denoised[i] < lo:
                denoised[i] = lo
            elif denoised[i] > hi:
                denoised[i] = hi

    return denoised


@njit(cache=True, fastmath=True)
def bilateral_filter_numba(
    data: np.ndarray, spatial_sigma: float, range_sigma: float
) -> np.ndarray:
    """Edge-preserving bilateral filter."""
    n = len(data)
    filtered = np.zeros_like(data)
    window = int(4 * spatial_sigma)
    sf = -0.5 / (spatial_sigma * spatial_sigma)
    rf = -0.5 / (range_sigma * range_sigma)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        tw = 0.0
        ws = 0.0
        for j in range(start, end):
            sd = (j - i) * (j - i)
            sw = np.exp(sd * sf)
            vd = data[j] - data[i]
            rw = np.exp(vd * vd * rf)
            w = sw * rw
            tw += w
            ws += data[j] * w
        filtered[i] = ws / tw if tw > 0 else data[i]

    return filtered


@njit(cache=True, fastmath=True)
def kalman_filter_numba(
    data: np.ndarray, process_var: float, measurement_var: float
) -> np.ndarray:
    """Simple scalar Kalman filter."""
    n = len(data)
    filtered = np.zeros(n)
    x_post = data[0]
    p_post = 1.0

    for i in range(n):
        x_pri = x_post
        p_pri = p_post + process_var
        k = p_pri / (p_pri + measurement_var)
        x_post = x_pri + k * (data[i] - x_pri)
        p_post = (1.0 - k) * p_pri
        filtered[i] = x_post

    return filtered


# =====================================================================
# HIGH-LEVEL SPIKE DETECTION & REMOVAL
# =====================================================================


class _HuberCache:
    """Thread-safe Huber regressor + polynomial feature cache."""

    def __init__(self):
        self._poly_cache: dict = {}
        self._local = threading.local()

    def get_poly(self, degree: int = 2) -> PolynomialFeatures:
        if degree not in self._poly_cache:
            self._poly_cache[degree] = PolynomialFeatures(
                degree=degree, include_bias=False
            )
        return self._poly_cache[degree]

    def get_huber(self, epsilon: float, alpha: float) -> HuberRegressor:
        if not hasattr(self._local, "huber"):
            self._local.huber = HuberRegressor(
                epsilon=epsilon, alpha=alpha, max_iter=1000
            )
        else:
            self._local.huber.epsilon = epsilon
            self._local.huber.alpha = alpha
        return self._local.huber


_huber_cache = _HuberCache()


@lru_cache(maxsize=32)
def _cwt_scales(sampling_rate: float, n_scales: int = 20) -> np.ndarray:
    return np.arange(1, n_scales)


def detect_spikes_cwt(
    data: np.ndarray,
    sampling_rate: float = 3012.0,
    wavelet: str = "mexh",
    threshold_factor: float = 2.0,
) -> np.ndarray:
    """Detect spikes via Continuous Wavelet Transform energy.

    Returns a boolean mask (True = spike).
    """
    scales = _cwt_scales(sampling_rate)
    coeffs, _ = pywt.cwt(data, scales, wavelet, sampling_period=1.0 / sampling_rate)
    energy = compute_spike_energy_cwt(coeffs)
    median_e = np.median(energy)
    mad_e = compute_mad(energy)
    threshold = median_e + threshold_factor * mad_e * 1.4826
    return energy > threshold


def remove_spikes_huber(
    data: np.ndarray,
    spike_mask: np.ndarray,
    window_size: int = 15,
    epsilon: float = 2.0,
    alpha: float = 0.01,
) -> np.ndarray:
    """Replace detected spikes with Huber-robust polynomial predictions."""
    cleaned = data.copy()
    spike_indices = np.where(spike_mask)[0]
    if len(spike_indices) == 0:
        return cleaned

    epsilon = max(1.0, epsilon)
    poly = _huber_cache.get_poly(degree=2)
    huber = _huber_cache.get_huber(epsilon, alpha)

    for idx in spike_indices:
        start = max(0, idx - window_size)
        end = min(len(data), idx + window_size + 1)
        window_mask = spike_mask[start:end]
        X = (np.arange(start, end) - idx).reshape(-1, 1)
        y = data[start:end]
        weights = (~window_mask).astype(float)
        weights[weights == 0] = 0.1

        try:
            X_poly = poly.fit_transform(X / (np.max(np.abs(X)) + 1e-10))
            huber.fit(X_poly, y, sample_weight=weights)
            cleaned[idx] = huber.predict(poly.transform([[0]]))[0]
        except Exception:
            clean_vals = y[~window_mask]
            cleaned[idx] = (
                np.median(clean_vals) if len(clean_vals) > 0 else np.median(y)
            )

    return cleaned


def remove_spikes_ransac(
    data: np.ndarray,
    spike_mask: np.ndarray,
    window_size: int = 15,
    min_samples: float = 0.5,
    residual_threshold_multiplier: float = 2.5,
) -> np.ndarray:
    """Replace detected spikes using RANSAC polynomial regression."""
    cleaned = data.copy()
    spike_indices = np.where(spike_mask)[0]
    if len(spike_indices) == 0:
        return cleaned

    for idx in spike_indices:
        start = max(0, idx - window_size)
        end = min(len(data), idx + window_size + 1)
        window_mask = spike_mask[start:end]
        X = (np.arange(start, end) - idx).reshape(-1, 1)
        y = data[start:end]

        if np.sum(~window_mask) < 5:
            cleaned[idx] = (
                np.median(y[~window_mask]) if np.any(~window_mask) else np.median(y)
            )
            continue

        try:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            threshold = np.std(y[~window_mask]) * residual_threshold_multiplier
            ransac = RANSACRegressor(
                min_samples=max(3, int(len(y) * min_samples)),
                residual_threshold=threshold,
                max_trials=100,
                random_state=0,
            )
            ransac.fit(X_poly, y)
            cleaned[idx] = ransac.predict(poly.transform([[0]]))[0]
        except Exception:
            clean_vals = y[~window_mask]
            cleaned[idx] = (
                np.median(clean_vals) if len(clean_vals) > 0 else np.median(y)
            )

    return cleaned


def detect_and_remove_spikes_ransac(
    data: np.ndarray,
    window_size: int = 15,
    min_samples: float = 0.5,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window RANSAC spike detection + removal."""
    cleaned = data.copy()
    spike_mask = np.zeros(len(data), dtype=bool)
    half = window_size // 2

    for i in range(half, len(data) - half, half):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size)
        window_data = data[start:end]
        X = np.arange(len(window_data)).reshape(-1, 1)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        try:
            ransac = RANSACRegressor(
                min_samples=min_samples,
                residual_threshold=np.std(window_data) * 2,
                random_state=0,
            )
            ransac.fit(X_poly, window_data)
            outliers = ~ransac.inlier_mask_
            spike_mask[start:end] |= outliers
            preds = ransac.predict(X_poly)
            wc = window_data.copy()
            wc[outliers] = preds[outliers]
            cleaned[start:end] = wc
        except Exception:
            continue

    return cleaned, spike_mask


def detect_and_remove_spikes_isolation(
    data: np.ndarray,
    contamination: float = 0.1,
    window_size: int = 5,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Isolation Forest anomaly detection + Huber replacement."""
    n = len(data)
    grad = np.gradient(data)
    features_list = []

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = data[start:end]
        features_list.append([
            data[i],
            np.mean(window),
            np.std(window),
            data[i] - np.mean(window),
            grad[i] if 0 < i < n - 1 else 0.0,
        ])

    features_array = np.array(features_list)
    iso = IsolationForest(contamination=contamination, random_state=0)
    predictions = iso.fit_predict(features_array)
    spike_mask = predictions == -1
    cleaned = remove_spikes_huber(data, spike_mask, window_size)
    return cleaned, spike_mask


# =====================================================================
# SMOOTHING FILTERS (non-Numba wrappers)
# =====================================================================


def apply_lowpass(
    data: np.ndarray,
    sampling_rate: float = 3012.0,
    cutoff_freq: float = 1500.0,
    filter_type: str = "bessel",
    order: int = 2,
) -> np.ndarray:
    """Apply a lowpass filter (Butterworth or Bessel)."""
    nyquist = sampling_rate / 2.0
    norm_cutoff = cutoff_freq / nyquist
    if filter_type == "butterworth":
        sos = butter(order, norm_cutoff, btype="low", output="sos")
    elif filter_type == "bessel":
        sos = bessel(order, norm_cutoff, btype="low", output="sos")
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return sosfiltfilt(sos, data)


def apply_wavelet_denoising(
    data: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
) -> np.ndarray:
    """Soft-threshold wavelet denoising."""
    if level is None:
        level = min(pywt.dwt_max_level(len(data), wavelet), 5)

    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs_t = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]
    denoised = pywt.waverec(coeffs_t, wavelet)
    return denoised[: len(data)]


# =====================================================================
# PRE-COMPILATION
# =====================================================================

def precompile():
    """Warm the Numba JIT cache with dummy data."""
    d = np.random.randn(100)
    _ = hampel_filter_numba(d, 5, 3.0)
    _ = tv_denoise_numba(d, 0.1, 10)
    _ = compute_mad(d)
    _ = compute_spike_energy_cwt(np.random.randn(5, 100))
    _ = bilateral_filter_numba(d, 2.0, 5.0)
    _ = kalman_filter_numba(d, 1e-5, 0.01)


# Run on import so the first real call isn't slow.
precompile()
