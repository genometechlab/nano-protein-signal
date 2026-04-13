"""Amino acid grouping definitions for multi-way classification."""

from typing import Dict, List


AMINO_ACIDS_20WAY: List[str] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

AMINO_ACIDS_2WAY: Dict[str, List[str]] = {
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E']
}

AMINO_ACIDS_3WAY: Dict[str, List[str]] = {
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E'],
    'neutral': [
        'S', 'T', 'N', 'Q', 'C',
        'G', 'A', 'V', 'L', 'I', 'M', 'P',
        'F', 'W', 'Y'
    ]
}

AMINO_ACIDS_4WAY: Dict[str, List[str]] = {
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E'],
    'big': ['F', 'W', 'Y', 'L', 'I', 'M', 'V'],
    'small': ['G', 'A', 'S', 'T', 'C', 'N', 'Q', 'P']
}

AMINO_ACIDS_5WAY_SIZE: Dict[str, List[str]] = {
    'very_small': ['G', 'A', 'S'],
    'small': ['C', 'D', 'N', 'P', 'T', 'V'],
    'medium': ['E', 'I', 'L', 'Q'],
    'large': ['H', 'K', 'M', 'F'],
    'very_large': ['R', 'W', 'Y']
}

AMINO_ACIDS_BIOLOGICAL: Dict[str, List[str]] = {
    'non_polar': ['A', 'G', 'I', 'L', 'M', 'F', 'W', 'P', 'V'],
    'polar': ['S', 'T', 'N', 'Q', 'C', 'Y'],
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E']
}

_CLASSIFICATION_MODES: Dict[str, Dict[str, List[str]]] = {
    '2way': AMINO_ACIDS_2WAY,
    '3way': AMINO_ACIDS_3WAY,
    '4way': AMINO_ACIDS_4WAY,
    '5way_size': AMINO_ACIDS_5WAY_SIZE,
    'biological': AMINO_ACIDS_BIOLOGICAL,
}


def get_classification_mode(mode: str) -> Dict[str, List[str]]:
    """Return the category-to-amino-acids mapping for the given mode."""
    if mode == '20way':
        return {aa: [aa] for aa in AMINO_ACIDS_20WAY}

    if mode not in _CLASSIFICATION_MODES:
        raise ValueError(f"Unknown classification mode: {mode}")

    return _CLASSIFICATION_MODES[mode]


def get_amino_acid_category(amino_acid: str, mode: str) -> str:
    """Return the category a single amino acid belongs to under the given mode."""
    if mode == '20way':
        return amino_acid

    groups = get_classification_mode(mode)
    for category, aa_list in groups.items():
        if amino_acid in aa_list:
            return category

    raise ValueError(f"Amino acid {amino_acid} not found in {mode} classification")


def get_all_categories(mode: str) -> List[str]:
    """Return all category names for the given classification mode."""
    return list(get_classification_mode(mode).keys())


def get_amino_acids_in_category(category: str, mode: str) -> List[str]:
    """Return the amino acids belonging to a specific category."""
    groups = get_classification_mode(mode)
    if category not in groups:
        raise ValueError(f"Category {category} not found in {mode} classification")
    return groups[category]
    