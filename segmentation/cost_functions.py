"""
Custom cost functions for change point detection
"""

import numpy as np
import ruptures as rpt


class CustomCost(rpt.base.BaseCost):
    """Variance-based custom cost function"""
    
    def __init__(self, scale=1, min_size=1):
        super().__init__()
        self.scale = scale
        self.min_size = min_size

    def fit(self, signal):
        self.signal = np.asarray(signal) - np.mean(signal)
        self.signal_sq = self.signal ** 2
        self.cumsum_signal = np.cumsum(self.signal)
        self.cumsum_signal_sq = np.cumsum(self.signal_sq)
        return self

    def error(self, start, end):
        if start == 0:
            seg_sum = self.cumsum_signal[end - 1]
            seg_sum_sq = self.cumsum_signal_sq[end - 1]
        else:
            seg_sum = self.cumsum_signal[end - 1] - self.cumsum_signal[start - 1]
            seg_sum_sq = self.cumsum_signal_sq[end - 1] - self.cumsum_signal_sq[start - 1]
        n = end - start
        return self.scale * (seg_sum_sq - (seg_sum ** 2) / n)

    @property
    def model(self):
        return "custom"