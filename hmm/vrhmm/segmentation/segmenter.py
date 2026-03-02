"""Signal segmentation functionality."""

import logging
from typing import Dict, List, Optional, Any, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

class SegmentVarianceCollector:
    """Collects segment variances across multiple signals."""

    def __init__(self, expected_segments: Optional[int] = None) -> None:
        """
        Args:
            expected_segments: If None, grows dynamically. If set, pre-allocates.
        """
        self.expected_segments = expected_segments
        if expected_segments:
            self.variance_lists: List[List[float]] = [[] for _ in range(expected_segments)]
        else:
            self.variance_lists: List[List[float]] = []
        self._max_observed = 0

    def add_signal_variances(self, variances: List[float]) -> None:
        """Add variances from a signal, growing list if needed."""
        # Grow the list if we see more segments
        while len(self.variance_lists) < len(variances):
            self.variance_lists.append([])
        
        for i, var in enumerate(variances):
            self.variance_lists[i].append(var)
        
        self._max_observed = max(self._max_observed, len(variances))

    def get_average_variances(
            self,
            max_samples: int = 10
    ) -> List[npt.NDArray[np.float64]]:
        result = []
        for var_list in self.variance_lists:
            limited_list = var_list[:max_samples]
            result.append(np.array(limited_list, dtype=np.float64))
        return result
    
    @property
    def num_segments(self) -> int:
        """Return the number of segments observed."""
        return len(self.variance_lists)

class Segmenter:
    """Class for segmenting signals and extracting segment statistics."""

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

        logger.info(
            f"Initialized Segmenter with penalty={self.penalty}, "
            f"scale={self.scale}, min_size={self.min_size}"
        )

    def segment(
            self,
            signal: Union[npt.NDArray[np.float64], List[float]],
            seg_mode: str = 'dynp'
    ) -> Dict[str, Any]:
        """Segment a signal and extract statistics."""
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=np.float64)

        filtered = apply_bessel_filter(
            signal,
            order=self.filter_config.get('order', 1),
            cutoff=self.filter_config.get('cutoff', 3000),
            sampling_rate=self.filter_config.get('sampling_rate', 10000)
        )

        if seg_mode == 'set_window':
            bkps = run_set_window_segmentation(filtered, num_bkps=self.num_bkps)
        elif seg_mode == 'dynp':
            bkps = run_dynamic_segmentation(
                filtered,
                scale=self.scale,
                num_bkps=self.num_bkps,
                min_size=self.min_size
            )
        elif seg_mode == 'pelt':
            bkps = run_pelt_segmentation(
                filtered,
                penalty=self.penalty,
                scale=self.scale,
                min_size=self.min_size
            )
        else:
            raise ValueError(f"Unknown segmentation mode: {seg_mode}")

        if bkps and bkps[0] != 0:
            bkps = [0] + bkps

        stats = []
        means = []
        variances = []
        start = 0

        for bkp in bkps[1:]:
            segment = signal[start:bkp]
            if len(segment) > 0:
                seg_mean = float(np.mean(segment))
                seg_var = float(np.var(segment))
                stats.append({'mean': seg_mean, 'var': seg_var})
                means.append(seg_mean)
                variances.append(seg_var)
            start = bkp

        return {
            'stats': stats,
            'breakpoints': bkps,
            'means': np.array(means, dtype=np.float64),
            'variances': np.array(variances, dtype=np.float64)
        }