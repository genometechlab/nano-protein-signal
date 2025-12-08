"""Type definitions for vrhmm package."""

from typing import Dict, List, Tuple, Any, Optional, Union, TypedDict
import numpy as np
import numpy.typing as npt

# Type aliases
ProfileArray = npt.NDArray[np.float64]
SegmentStats = Dict[str, Tuple[float, float]]
SignalArray = npt.NDArray[np.float64]
BreakpointList = List[int]

class SegmentResult(TypedDict):
    """Structure for segmentation results."""
    stats: List[Dict[str, float]]
    breakpoints: BreakpointList
    means: ProfileArray
    variances: ProfileArray
    z_normalized_stats: Optional[Dict[str, Tuple[float, float]]]
    hmm_profile_stats: Optional[Dict[str, Tuple[float, float]]]

class ClassificationResult(TypedDict):
    """Structure for classification results."""
    predicted_category: str
    log_probability: float
    all_scores: Dict[str, float]
    state_sequence: List[str]
    full_path: List[str]

class SignalRecord(TypedDict):
    """Structure for signal data records."""
    run: str
    channel: int
    segment: int
    aa: str
    cleaned_segment: Union[str, List[float], ProfileArray]
    pretty: str