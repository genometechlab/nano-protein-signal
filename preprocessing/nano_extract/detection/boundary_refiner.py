"""
nano_extract.detection.boundary_refiner
=========================================
Refine the edges of detected YY dip regions to find the exact
boundary points (the minimum-current sample at each dip edge).

This is the updated version of the original BoundaryFinder, rewired
to work on nanoclean-processed signals and DipRegion objects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper if not args or not callable(args[0]) else args[0]

from nano_extract.core.config import ExtractionConfig
from nano_extract.detection.dip_detector import DipRegion, _find_min_in_region

logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True)
def _refine_edge(signal: np.ndarray, center: int, padding: int, sig_len: int) -> Tuple[int, float]:
    """Find the minimum within ±padding of center."""
    start = max(0, center - padding)
    end = min(sig_len, center + padding)
    if start >= end:
        return center, signal[center]
    return _find_min_in_region(signal, start, end)


class BoundaryRefiner:
    """Refine dip edges to precise boundary positions.

    Given a list of DipRegion objects (coarse boundaries), this class
    searches a small window around each dip edge to find the exact
    sample where the signal reaches its minimum — that's the true
    YY boundary point.

    Parameters
    ----------
    config : ExtractionConfig
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.cfg = config or ExtractionConfig()

    def refine(
        self,
        signal: np.ndarray,
        dips: List[DipRegion],
    ) -> List[Dict]:
        """Refine boundaries for a list of dip regions.

        Parameters
        ----------
        signal : np.ndarray
            The cleaned signal (same one used for detection).
        dips : list of DipRegion
            Dip regions from :class:`DipDetector`.

        Returns
        -------
        list of dict
            One dict per dip, each containing:
            - ``left_idx``, ``left_val`` : refined left boundary
            - ``right_idx``, ``right_val`` : refined right boundary
            - ``min_idx``, ``min_val`` : deepest point in the dip
            - ``dip`` : the original DipRegion
        """
        padding = self.cfg.refinement_padding
        n = len(signal)
        results = []

        for dip in dips:
            # Refine left edge
            left_idx, left_val = _refine_edge(signal, dip.start, padding, n)

            # Refine right edge
            right_idx, right_val = _refine_edge(signal, dip.end - 1, padding, n)

            # The deepest point is already known from detection
            results.append({
                "left_idx": int(left_idx),
                "left_val": float(left_val),
                "right_idx": int(right_idx),
                "right_val": float(right_val),
                "min_idx": dip.min_index,
                "min_val": dip.min_value,
                "dip": dip,
            })

        return results

    def get_segment_boundaries(
        self,
        signal: np.ndarray,
        dips: List[DipRegion],
    ) -> List[Tuple[int, int]]:
        """Get the (start, end) pairs for segments between consecutive dips.

        The segment boundaries are the refined minimum points of
        adjacent dips: segment i runs from dip[i].min_index to
        dip[i+1].min_index.

        Parameters
        ----------
        signal : np.ndarray
            Cleaned signal.
        dips : list of DipRegion
            Detected dip regions (must be sorted by start).

        Returns
        -------
        list of (start, end)
            One pair per inter-dip segment.  With 5 dips you get
            4 segments (plus optionally flanks).
        """
        refined = self.refine(signal, dips)
        segments = []

        # Optionally include left flank (before first dip)
        if self.cfg.include_flanks and refined[0]["min_idx"] > self.cfg.min_segment_length:
            segments.append((0, refined[0]["min_idx"]))

        # Inter-dip segments
        for i in range(len(refined) - 1):
            seg_start = refined[i]["min_idx"]
            seg_end = refined[i + 1]["min_idx"]
            if seg_end - seg_start >= self.cfg.min_segment_length:
                segments.append((seg_start, seg_end))

        # Optionally include right flank (after last dip)
        if self.cfg.include_flanks:
            last_min = refined[-1]["min_idx"]
            if len(signal) - last_min > self.cfg.min_segment_length:
                segments.append((last_min, len(signal)))

        return segments
