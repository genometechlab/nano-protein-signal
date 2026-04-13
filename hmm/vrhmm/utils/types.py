"""Type definitions for vrhmm package."""

from typing import Dict, List, Tuple, Optional, Union, TypedDict

import numpy as np
import numpy.typing as npt

ProfileArray = npt.NDArray[np.float64]
SignalArray = npt.NDArray[np.float64]
SegmentStats = Dict[str, Tuple[float, float]]
BreakpointList = List[int]


class SegmentResult(TypedDict):
    breakpoints: BreakpointList
    means: ProfileArray
    variances: ProfileArray
    z_normalized_stats: Optional[Dict[str, Tuple[float, float]]]


class ClassificationResult(TypedDict):
    predicted_category: str
    log_probability: float
    all_scores: Dict[str, float]
    state_sequence: List[str]
    full_path: List[str]


class SignalRecord(TypedDict):
    run: str
    channel: int
    segment: int
    aa: str
    cleaned_segment: Union[str, List[float], ProfileArray]
    pretty: str