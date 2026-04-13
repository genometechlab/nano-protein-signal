"""Signal segmentation functionality."""

import logging
from typing import Dict, List, Optional, Any, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class SegmentVarianceCollector:
    """Collects segment variances across multiple signals."""

    def __init__(self, expected_segments: Optional[int] = None) -> None:
        self.expected_segments = expected_segments
        self.variance_lists: List[List[float]] = (
            [[] for _ in range(expected_segments)] if expected_segments else []
        )
        self._max_observed = 0

    def add_signal_variances(self, variances: List[float]) -> None:
        """Add variances from a signal, growing the list if needed."""
        while len(self.variance_lists) < len(variances):
            self.variance_lists.append([])

        for i, var in enumerate(variances):
            self.variance_lists[i].append(var)

        self._max_observed = max(self._max_observed, len(variances))

    def get_average_variances(
        self,
        max_samples: int = 10
    ) -> List[npt.NDArray[np.float64]]:
        return [
            np.array(var_list[:max_samples], dtype=np.float64)
            for var_list in self.variance_lists
        ]

    @property
    def num_segments(self) -> int:
        return len(self.variance_lists)


class Segmenter:
    """Segments signals and extracts per-segment statistics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            from vrhmm.config import CONFIG
            self.seg_config = CONFIG['segmentation']
            self.filter_config = CONFIG['filtering']
        else:
            self.seg_config = config.get('segmentation', {})
            self.filter_config = config.get('filtering', {})

        self.penalty = self.seg_config.get('penalty', 1)
        self.scale = self.seg_config.get('scale', 5)
        self.min_size = self.seg_config.get('min_size', 25)
        self.num_bkps = self.seg_config.get('num_bkps', 35)

    def segment(
        self,
        signal: Union[npt.NDArray[np.float64], List[float]],
        seg_mode: str = 'dynp'
    ) -> Dict[str, Any]:
        """Segment a signal and extract per-segment statistics.

        Breakpoints are detected on the Bessel-filtered signal, but segment
        statistics (mean, variance) are computed on the original signal so
        that filtering artifacts don't distort the emission distributions.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=np.float64)

        filtered = apply_bessel_filter(
            signal,
            order=self.filter_config.get('order', 1),
            cutoff=self.filter_config.get('cutoff', 3000),
            sampling_rate=self.filter_config.get('sampling_rate', 10000)
        )

        bkps = self._detect_breakpoints(filtered, seg_mode)

        if bkps and bkps[0] != 0:
            bkps = [0] + bkps

        means = []
        variances = []
        start = 0
        for bkp in bkps[1:]:
            seg = signal[start:bkp]
            if len(seg) > 0:
                means.append(float(np.mean(seg)))
                variances.append(float(np.var(seg)))
            start = bkp

        return {
            'breakpoints': bkps,
            'means': np.array(means, dtype=np.float64),
            'variances': np.array(variances, dtype=np.float64)
        }

    def _detect_breakpoints(
        self,
        filtered: npt.NDArray[np.float64],
        seg_mode: str
    ) -> List[int]:
        if seg_mode == 'set_window':
            return run_set_window_segmentation(filtered, num_bkps=self.num_bkps)
        elif seg_mode == 'dynp':
            return run_dynamic_segmentation(
                filtered,
                scale=self.scale,
                num_bkps=self.num_bkps,
                min_size=self.min_size
            )
        elif seg_mode == 'pelt':
            return run_pelt_segmentation(
                filtered,
                penalty=self.penalty,
                scale=self.scale,
                min_size=self.min_size
            )
        else:
            raise ValueError(f"Unknown segmentation mode: {seg_mode}")
            