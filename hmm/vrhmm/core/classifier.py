"""Multi-way classification using HMM profiles."""

import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import numpy.typing as npt

from vrhmm.utils.amino_acids import (
    get_amino_acid_category,
    get_all_categories,
    get_amino_acids_in_category
)

logger = logging.getLogger(__name__)

class HMMClassifier:
    """Multi-way classifier using HMM profiles for amino acid prediction."""

    def __init__(
            self,
            classification_mode: str = '20way',
            hmm_models: Optional[Dict[str, Any]] = None,
            use_length_normalization: bool = False
    ) -> None:
        
        self.classification_mode = classification_mode
        self.hmm_models = hmm_models or {}
        self.categories = get_all_categories(classification_mode)
        self.use_length_normalization = use_length_normalization

        logger.info(f"Initialized HMMClassifier with {classification_mode} mode")

    def add_model(self, identifier: str, model: Any) -> None:
        
        self.hmm_models[identifier] = model
        logger.debug(f"Added model for {identifier}")

    def predict(
            self,
            observation_sequence: npt.NDArray[np.float64]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict the category for an observation sequence."""
        if not self.hmm_models:
            raise ValueError("No HMM models available for prediction")

        scores = {}
        for identifier, model in self.hmm_models.items():
            try:
                log_prob = model.log_probability(observation_sequence)

                if self.use_length_normalization:
                    scores[identifier] = log_prob / len(observation_sequence)
                else:
                    scores[identifier] = log_prob

            except Exception as e:
                logger.warning(f"Error scoring with model {identifier}: {e}")
                scores[identifier] = float('-inf')

        if self.classification_mode != '20way':
            category_scores = self._aggregate_scores_by_category(scores)
            best_category = max(category_scores.keys(),
                                key=lambda k: category_scores[k])
            return best_category, category_scores[best_category], category_scores
        else:
            best_aa = max(scores.keys(), key=lambda k: scores[k])
            return best_aa, scores[best_aa], scores

    def _aggregate_scores_by_category(
            self,
            aa_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate amino acid scores into category scores."""
        category_scores = {}

        for category in self.categories:
            aa_list = get_amino_acids_in_category(category, self.classification_mode)
            relevant_scores = [
                aa_scores[aa] for aa in aa_list
                if aa in aa_scores
            ]

            if relevant_scores:
                category_scores[category] = max(relevant_scores)
            else:
                category_scores[category] = float('-inf')

        return category_scores