"""Amino acid grouping definitions for multi-way classification."""

from typing import Dict, List

# 20-way classification: individual amino acids
AMINO_ACIDS_20WAY: List[str] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# 2-way classification: positive vs negative charged
AMINO_ACIDS_2WAY: Dict[str, List[str]] = {
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E']
}

# 4-way classification: positive, negative, big, small
AMINO_ACIDS_4WAY: Dict[str, List[str]] = {
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E'],
    'big': ['F', 'W', 'Y', 'L', 'I', 'M', 'V'],
    'small': ['G', 'A', 'S', 'T', 'C', 'N', 'Q', 'P']
}

# Biological classification: standard biochemical categories
AMINO_ACIDS_BIOLOGICAL: Dict[str, List[str]] = {
    'acidic': ['D', 'E'],
    'basic': ['K', 'R', 'H'],
    'polar_uncharged': ['S', 'T', 'N', 'Q', 'C'],
    'nonpolar_aliphatic': ['G', 'A', 'V', 'L', 'I', 'M', 'P'],
    'aromatic': ['F', 'W', 'Y']
}

def get_classification_mode(mode: str) -> Dict[str, List[str]]:
    
    if mode == '2way':
        return AMINO_ACIDS_2WAY
    elif mode == '4way':
        return AMINO_ACIDS_4WAY
    elif mode == 'biological':
        return AMINO_ACIDS_BIOLOGICAL
    elif mode == '20way':
        return {aa: [aa] for aa in AMINO_ACIDS_20WAY}
    else:
        raise ValueError(f"Unknown classification mode: {mode}")

def get_amino_acid_category(amino_acid: str, mode: str) -> str:
    
    if mode == '20way':
        return amino_acid

    groups = get_classification_mode(mode)

    for category, aa_list in groups.items():
        if amino_acid in aa_list:
            return category

    raise ValueError(f"Amino acid {amino_acid} not found in {mode} classification")

def get_all_categories(mode: str) -> List[str]:
    
    groups = get_classification_mode(mode)
    return list(groups.keys())

def get_amino_acids_in_category(category: str, mode: str) -> List[str]:
    
    groups = get_classification_mode(mode)

    if category not in groups:
        raise ValueError(f"Category {category} not found in {mode} classification")

    return groups[category]