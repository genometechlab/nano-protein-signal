"""
vrHMM: Variable Rate Hidden Markov Model for Nanopore Sequencing.
"""

__version__ = "1.0.0"

from vrhmm.segmentation import Segmenter, SegmentVarianceCollector
from vrhmm.core import HMMConstructor, HMMClassifier
from vrhmm.io import DataLoader  # Changed from vrhmm.utils
from vrhmm.processing import apply_bessel_filter
from vrhmm.utils.amino_acids import (
    get_classification_mode,
    get_amino_acid_category,
    get_all_categories,
    get_amino_acids_in_category,
)
from vrhmm.config import CONFIG

__all__ = [
    "Segmenter",
    "SegmentVarianceCollector",
    "HMMConstructor",
    "HMMClassifier",
    "DataLoader",
    "apply_bessel_filter",
    "get_classification_mode",
    "get_amino_acid_category",
    "get_all_categories",
    "get_amino_acids_in_category",
    "CONFIG",
]