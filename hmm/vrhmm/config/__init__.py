"""Configuration module for vrHMM."""

from vrhmm.config.default_config import CONFIG

# Extract the sub-configs from CONFIG for backward compatibility
FILTERING = CONFIG.get('filtering', {})
SEGMENTATION = CONFIG.get('segmentation', {})
HMM = CONFIG.get('hmm', {})

__all__ = ['CONFIG', 'FILTERING', 'SEGMENTATION', 'HMM']