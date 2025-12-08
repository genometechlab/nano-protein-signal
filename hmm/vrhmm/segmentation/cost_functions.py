"""Cost functions for segmentation algorithms."""

from typing import Union

import numpy as np
import numpy.typing as npt
import numba
import ruptures as rpt

@numba.jit(nopython=True, fastmath=True)
def compute_segment_cost(
        cumsum_signal: npt.NDArray[np.float64],
        cumsum_signal_sq: npt.NDArray[np.float64],
        start: int,
        end: int,
        scale: float,
        min_size: int
) -> float:
    """Fast computation of segment cost using precomputed cumulative sums."""
    if end - start < min_size:
        return np.inf

    if start == 0:
        seg_sum = cumsum_signal[end - 1]
        seg_sum_sq = cumsum_signal_sq[end - 1]
    else:
        seg_sum = cumsum_signal[end - 1] - cumsum_signal[start - 1]
        seg_sum_sq = cumsum_signal_sq[end - 1] - cumsum_signal_sq[start - 1]

    n = end - start
    cost = scale * (seg_sum_sq - (seg_sum * seg_sum) / n)
    return cost

class CustomCost(rpt.base.BaseCost):
    """Numba-optimized custom cost function for segmentation."""

    def __init__(self, scale: float = 1.0, min_size: int = 1) -> None:
        
        super().__init__()
        self.scale = float(scale)
        self.min_size = int(min_size)

    def fit(
            self,
            signal: Union[npt.NDArray[np.float64], list]
    ) -> "CustomCost":
        """Precompute necessary values for the cost function."""
        self.signal = np.asarray(signal, dtype=np.float64)
        self.signal = self.signal - np.mean(self.signal)
        self.signal_sq = self.signal ** 2
        self.cumsum_signal = np.cumsum(self.signal)
        self.cumsum_signal_sq = np.cumsum(self.signal_sq)

        self.cumsum_signal = np.ascontiguousarray(self.cumsum_signal)
        self.cumsum_signal_sq = np.ascontiguousarray(self.cumsum_signal_sq)

        return self

    def error(self, start: int, end: int) -> float:
        """Compute the cost of a segment."""
        return compute_segment_cost(
            self.cumsum_signal,
            self.cumsum_signal_sq,
            start,
            end,
            self.scale,
            self.min_size
        )

    @property
    def model(self) -> str:
        """Model identifier."""
        return "custom_cost"

def warmup() -> None:
    """Precompile the Numba function."""
    dummy_signal = np.random.randn(100)
    dummy_cumsum = np.cumsum(dummy_signal)
    dummy_cumsum_sq = np.cumsum(dummy_signal ** 2)
    compute_segment_cost(dummy_cumsum, dummy_cumsum_sq, 0, 10, 1.0, 1)

# Run warmup on module load
warmup()