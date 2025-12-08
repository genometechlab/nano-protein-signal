"""Segmentation algorithms implementation."""

import logging
from typing import List

import numpy as np
import numpy.typing as npt
import ruptures as rpt

from vrhmm.segmentation.cost_functions import CustomCost

logger = logging.getLogger(__name__)

def run_set_window_segmentation(
        signal: npt.NDArray[np.float64],
        num_bkps: int = 35
) -> List[int]:
    
    try:
        bkps = []
        step_size = len(signal) // num_bkps

        for i in range(1, num_bkps):
            bkps.append(i * step_size)

        if bkps and bkps[-1] != len(signal):
            bkps.append(len(signal))

        return bkps
    except Exception as e:
        logger.error(f"Error in set window segmentation: {e}")
        return [len(signal)]

def run_dynamic_segmentation(
        signal: npt.NDArray[np.float64],
        scale: float,
        num_bkps: int = 34,
        min_size: int = 1
) -> List[int]:
    
    try:
        cost = CustomCost(scale=scale, min_size=min_size).fit(signal)
        algo = rpt.Dynp(custom_cost=cost).fit(signal)
        bkps = algo.predict(n_bkps=num_bkps)
        return bkps
    except Exception as e:
        logger.error(f"Error in dynamic programming segmentation: {e}")
        return [len(signal)]

def run_pelt_segmentation(
        signal: npt.NDArray[np.float64],
        penalty: float,
        scale: float,
        min_size: int = 1
) -> List[int]:
    
    try:
        cost = CustomCost(scale=scale, min_size=min_size).fit(signal)
        algo = rpt.Pelt(custom_cost=cost).fit(signal)
        bkps = algo.predict(pen=penalty)
        return bkps
    except Exception as e:
        logger.error(f"Error in PELT segmentation: {e}")
        return [len(signal)]