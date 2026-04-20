"""
nano_extract.detection.dip_detector
=====================================
Detect sustained YY dip regions in nanoclean-processed signals.

A YY dip is a contiguous region where the signal drops below a
threshold and *stays low* for a minimum number of samples.  Each dip
is a distinct YY boundary in the construct — no merging is performed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import groupby
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper if not args or not callable(args[0]) else args[0]

    prange = range

from nano_extract.core.config import ExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class DipRegion:
    """A single sustained YY dip."""

    start: int
    """First sample index of the dip."""

    end: int
    """One-past-last sample index (like a Python slice)."""

    width: int
    """Number of samples in the dip (end - start)."""

    min_value: float
    """Minimum signal value within the dip."""

    min_index: int
    """Index of the minimum value (absolute, not relative)."""

    depth: float
    """How far below the threshold the minimum sits."""

    score: float
    """Ranking metric (width × depth) used to select the best dips."""


@njit(cache=True, fastmath=True)
def _find_min_in_region(
    signal: np.ndarray, start: int, end: int
) -> Tuple[int, float]:
    """Numba-optimised minimum finder."""
    min_val = signal[start]
    min_idx = start
    for i in range(start + 1, end):
        if signal[i] < min_val:
            min_val = signal[i]
            min_idx = i
    return min_idx, min_val


@njit(cache=True, fastmath=True, parallel=True)
def _find_mins_batch(
    signal: np.ndarray, regions: np.ndarray
) -> np.ndarray:
    """Find minimums in multiple regions in parallel.

    Parameters
    ----------
    signal : 1-D float array
    regions : (N, 2) int array of [start, end) pairs

    Returns
    -------
    (N, 2) float array of [min_idx, min_val] per region
    """
    n = regions.shape[0]
    out = np.empty((n, 2), dtype=np.float64)
    for i in prange(n):
        idx, val = _find_min_in_region(signal, regions[i, 0], regions[i, 1])
        out[i, 0] = idx
        out[i, 1] = val
    return out


class DipDetector:
    """Find sustained YY dip regions in a cleaned signal.

    Each detected dip corresponds to one YY boundary in the construct.
    No merging is performed — every sustained low-current region that
    meets the width threshold is treated as a distinct dip.

    If more dips are found than ``n_expected_dips``, the top-scoring
    ones (by width × depth) are kept.

    Parameters
    ----------
    config : ExtractionConfig
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.cfg = config or ExtractionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, signal: np.ndarray) -> List[DipRegion]:
        """Detect YY dips and return exactly ``n_expected_dips`` regions.

        Parameters
        ----------
        signal : np.ndarray
            Cleaned signal (output of nanoclean).

        Returns
        -------
        list of DipRegion
            Sorted by start index.
        """
        smoothed = self._smooth(signal)
        threshold = np.percentile(smoothed, self.cfg.dip_threshold_percentile)
        candidates = self._find_sustained_regions(smoothed, threshold)

        if len(candidates) == 0:
            logger.warning(
                "No sustained dips found in signal of length %d", len(signal)
            )
            return []

        n = self.cfg.n_expected_dips

        if len(candidates) < n:
            logger.warning(
                "Found only %d sustained dips (expected %d)",
                len(candidates),
                n,
            )
            return candidates

        if len(candidates) > n:
            # Keep top N by score, then re-sort by position
            candidates.sort(key=lambda d: d.score, reverse=True)
            candidates = sorted(candidates[:n], key=lambda d: d.start)

        return candidates

    def detect_with_metadata(self, signal: np.ndarray) -> dict:
        """Like :meth:`detect` but returns additional diagnostics.

        Returns
        -------
        dict with keys:
            dips : list of DipRegion
            smoothed : np.ndarray
            threshold : float
            n_candidates : int (before trimming)
            success : bool
        """
        smoothed = self._smooth(signal)
        threshold = np.percentile(smoothed, self.cfg.dip_threshold_percentile)
        all_candidates = self._find_sustained_regions(smoothed, threshold)
        n_candidates = len(all_candidates)

        n = self.cfg.n_expected_dips
        if len(all_candidates) > n:
            all_candidates.sort(key=lambda d: d.score, reverse=True)
            dips = sorted(all_candidates[:n], key=lambda d: d.start)
        else:
            dips = all_candidates

        return {
            "dips": dips,
            "smoothed": smoothed,
            "threshold": threshold,
            "n_candidates": n_candidates,
            "success": len(dips) == n,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing for dip detection."""
        win = self.cfg.smoothing_window
        win = min(win, len(signal) // 2 * 2 - 1)
        if win < 5:
            return signal.copy()
        return savgol_filter(signal, win, self.cfg.smoothing_polyorder)

    def _find_sustained_regions(
        self, smoothed: np.ndarray, threshold: float
    ) -> List[DipRegion]:
        """Find contiguous below-threshold regions >= min_dip_width."""
        is_low = smoothed < threshold
        regions: List[DipRegion] = []
        pos = 0

        for val, group in groupby(is_low):
            length = sum(1 for _ in group)
            if val and length >= self.cfg.min_dip_width:
                start = pos
                end = pos + length
                region_signal = smoothed[start:end]
                min_idx_rel = int(np.argmin(region_signal))
                min_val = float(region_signal[min_idx_rel])
                depth = threshold - min_val

                regions.append(
                    DipRegion(
                        start=start,
                        end=end,
                        width=length,
                        min_value=min_val,
                        min_index=start + min_idx_rel,
                        depth=depth,
                        score=length * depth,
                    )
                )
            pos += length

        return regions
