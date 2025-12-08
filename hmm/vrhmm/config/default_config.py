"""Default configuration for vrhmm system."""

from typing import Dict, Any, Union, List

class Config:
    """Configuration container for vrhmm parameters."""

    # Variance scaling factor
    VARIANCE_SCALE: float = 80.0

    # Filtering parameters
    FILTERING: Dict[str, Union[int, float]] = {
        'order': 1,
        'cutoff': 3000,
        'sampling_rate': 10000
    }

    # Segmentation parameters
    SEGMENTATION: Dict[str, Union[int, float]] = {
        'penalty': 1.0,
        'scale': 5.0,
        'min_size': 25,
        'num_bkps': 35
    }

    # HMM transition probabilities
    HMM_TRANSITIONS: Dict[str, float] = {
        'match_self_loop': 0.013513513513513514,
        'forward': 0.675675675675675,
        'to_skip': 0.2027027027027027,
        'to_slip': 0.06756756756756757,
        'to_end': 0.04054054054054054,
        'to_insert': 0.03,
        'skip_to_match': 0.95,
        'skip_continue': 0.05,
        'insert_self_loop': 0.05,
        'insert_to_match': 0.95,
        'slip_to_match': 0.92,
        'slip_continue': 0.08
    }

    # Classification modes
    CLASSIFICATION_MODES: List[str] = ['2way', '4way', 'biological', '20way']
    VARIANCE_MODES: List[str] = ['barycenter', 'segment']

    @classmethod
    def to_dict(cls) -> Dict[str, Dict[str, Any]]:
        """Convert configuration to dictionary format."""
        return {
            'filtering': cls.FILTERING,
            'segmentation': cls.SEGMENTATION,
            'hmm': {
                'variance_scale': cls.VARIANCE_SCALE,
                'transitions': cls.HMM_TRANSITIONS
            },
            'classification': {
                'modes': cls.CLASSIFICATION_MODES,
                'variance_modes': cls.VARIANCE_MODES,
                'segment_variance_samples': 10,
                'expected_segments': 35
            }
        }

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration parameters."""
        if cls.FILTERING['cutoff'] <= 0:
            raise ValueError("Cutoff frequency must be positive")
        if cls.FILTERING['sampling_rate'] <= 0:
            raise ValueError("Sampling rate must be positive")
        if cls.SEGMENTATION['min_size'] < 1:
            raise ValueError("Minimum segment size must be at least 1")
        if cls.SEGMENTATION['scale'] <= 0:
            raise ValueError("Scale factor must be positive")

        for key, value in cls.HMM_TRANSITIONS.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Transition probability {key}={value} must be in [0, 1]")

        return True

CONFIG = Config.to_dict()
Config.validate()