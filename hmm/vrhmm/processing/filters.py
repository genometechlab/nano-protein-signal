"""Signal filtering utilities."""

from typing import Union, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt, butter

def apply_bessel_filter(
        signal: Union[npt.NDArray[np.float64], List[float]],
        order: int = 1,
        cutoff: float = 1500,
        sampling_rate: float = 3012
) -> npt.NDArray[np.float64]:
    """Apply a low-pass Butterworth filter to a signal."""
    nyquist: float = 0.5 * sampling_rate
    normal_cutoff: float = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal