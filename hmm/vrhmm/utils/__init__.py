"""Utility functions for vrHMM."""

from vrhmm.utils.amino_acids import (
    get_classification_mode,
    get_amino_acid_category,
    get_all_categories,
    get_amino_acids_in_category,
    AMINO_ACIDS_2WAY,
    AMINO_ACIDS_4WAY,
    AMINO_ACIDS_BIOLOGICAL,
    AMINO_ACIDS_20WAY
)

__all__ = [
    'get_classification_mode',
    'get_amino_acid_category',
    'get_all_categories',
    'get_amino_acids_in_category',
    'AMINO_ACIDS_2WAY',
    'AMINO_ACIDS_4WAY',
    'AMINO_ACIDS_BIOLOGICAL',
    'AMINO_ACIDS_20WAY',
]