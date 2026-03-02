"""Multi-way classification using HMM profiles with coverage-weighted scoring."""

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


def analyze_viterbi_path_for_classification(
    path: List[Any],
    n_expected_matches: int
) -> Dict[str, Any]:
    """
    Analyze a Viterbi path for classification metrics.
    
    Args:
        path: List of (idx, state) tuples or state objects from Viterbi
        n_expected_matches: Number of match states in the model
    
    Returns:
        Dictionary with coverage, efficiency, and irregularity counts
    """
    match_indices = []
    prev_match_idx = -1
    
    n_skips = 0
    n_slips = 0
    n_self_loops = 0
    n_inserts = 0
    prev_state_name = None
    
    for item in path:
        # Handle different path formats
        if item is None:
            continue
        
        if hasattr(item, 'name'):
            state_name = item.name
        elif isinstance(item, tuple) and len(item) >= 2:
            state_obj = item[1]
            state_name = getattr(state_obj, 'name', None)
        else:
            continue
        
        if state_name is None:
            continue
        
        if 'Match' in state_name:
            try:
                idx = int(state_name.split('_')[1])
                match_indices.append(idx)
                
                if prev_match_idx >= 0:
                    if idx > prev_match_idx + 1:
                        n_skips += (idx - prev_match_idx - 1)
                    elif idx < prev_match_idx:
                        n_slips += 1
                    elif idx == prev_match_idx and state_name == prev_state_name:
                        n_self_loops += 1
                
                prev_match_idx = idx
                prev_state_name = state_name
                
            except (ValueError, IndexError):
                continue
                
        elif 'Insert' in state_name:
            n_inserts += 1
    
    n_unique = len(set(match_indices))
    n_total = len(match_indices)
    
    coverage = n_unique / n_expected_matches if n_expected_matches > 0 else 0.0
    efficiency = n_unique / n_total if n_total > 0 else 0.0
    
    total_emissions = n_total + n_inserts
    irregularities = n_skips + n_slips + 0.5 * n_self_loops
    smoothness = max(0.0, 1.0 - irregularities / max(1, total_emissions))
    
    return {
        'coverage': coverage,
        'efficiency': efficiency,
        'smoothness': smoothness,
        'n_unique_matches': n_unique,
        'n_total_matches': n_total,
        'n_expected_matches': n_expected_matches,
        'n_skips': n_skips,
        'n_slips': n_slips,
        'n_self_loops': n_self_loops,
        'n_inserts': n_inserts,
        'match_indices': match_indices
    }


class HMMClassifier:
    """
    HMM-based classifier with optional coverage-weighted scoring.
    
    Supports:
    - Standard log-likelihood scoring
    - Coverage-weighted scoring (penalizes models that use skips/inserts excessively)
    - Multiple classification modes (20-way, 4-way, etc.)
    """
    
    def __init__(
        self,
        classification_mode: str = '20way',
        hmm_models: Optional[Dict[str, Any]] = None,
        use_length_normalization: bool = False,
        coverage_weight: float = 0.0
    ) -> None:
        """
        Args:
            classification_mode: '20way', '4way', '3way', '2way', or 'biological'
            hmm_models: Pre-built HMM models {aa: model}
            use_length_normalization: Normalize LL by sequence length
            coverage_weight: If > 0, weight scores by coverage^weight
        """
        self.classification_mode = classification_mode
        self.hmm_models = hmm_models or {}
        self.model_lengths: Dict[str, int] = {}
        self.categories = get_all_categories(classification_mode)
        self.use_length_normalization = use_length_normalization
        self.coverage_weight = coverage_weight

    def add_model(self, identifier: str, model: Any) -> None:
        """Add an HMM model for an amino acid."""
        self.hmm_models[identifier] = model
        
        # Track model length (number of match states)
        match_count = sum(
            1 for state in model.states 
            if hasattr(state, 'name') and state.name and 'Match' in state.name
        )
        self.model_lengths[identifier] = match_count
        logger.debug(f"Added model for {identifier} with {match_count} match states")
    
    def get_model_length(self, identifier: str) -> int:
        """Get number of match states for a model."""
        return self.model_lengths.get(identifier, 35)

    def predict(
        self,
        observation_sequence: npt.NDArray[np.float64],
        return_path_metrics: bool = False
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict the category for an observation sequence.
        
        Args:
            observation_sequence: Z-normalized segment means
            return_path_metrics: If True, include path metrics in returned scores
        
        Returns:
            Tuple of (predicted_category, score, all_scores)
        """
        if not self.hmm_models:
            raise ValueError("No HMM models available for prediction")

        scores = {}
        path_metrics = {}
        
        for identifier, model in self.hmm_models.items():
            try:
                log_prob = model.log_probability(observation_sequence)

                if self.use_length_normalization:
                    log_prob = log_prob / len(observation_sequence)
                
                # Apply coverage weighting if enabled
                if self.coverage_weight > 0:
                    _, path = model.viterbi(observation_sequence)
                    n_expected = self.get_model_length(identifier)
                    metrics = analyze_viterbi_path_for_classification(path, n_expected)
                    
                    coverage_factor = metrics['coverage'] ** self.coverage_weight
                    weighted_score = log_prob * coverage_factor
                    
                    scores[identifier] = weighted_score
                    path_metrics[identifier] = metrics
                else:
                    scores[identifier] = log_prob

            except Exception as e:
                logger.warning(f"Error scoring with model {identifier}: {e}")
                scores[identifier] = float('-inf')

        # Aggregate by category if needed
        if self.classification_mode != '20way':
            category_scores = self._aggregate_scores_by_category(scores)
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            return best_category, category_scores[best_category], category_scores
        else:
            best_aa = max(scores.keys(), key=lambda k: scores[k])
            
            # Include path metrics if requested
            if return_path_metrics and path_metrics:
                scores['_path_metrics'] = path_metrics
            
            return best_aa, scores[best_aa], scores

    def predict_with_details(
        self,
        observation_sequence: npt.NDArray[np.float64]
    ) -> Dict[str, Any]:
        """
        Predict with detailed path analysis for all models.
        
        Returns comprehensive information about each model's fit.
        """
        if not self.hmm_models:
            raise ValueError("No HMM models available for prediction")
        
        results = {
            'scores': {},
            'path_metrics': {},
            'raw_log_probs': {},
            'coverage_factors': {}
        }
        
        for identifier, model in self.hmm_models.items():
            try:
                # Get log probability
                log_prob = model.log_probability(observation_sequence)
                results['raw_log_probs'][identifier] = log_prob
                
                # Get Viterbi path and analyze
                _, path = model.viterbi(observation_sequence)
                n_expected = self.get_model_length(identifier)
                metrics = analyze_viterbi_path_for_classification(path, n_expected)
                results['path_metrics'][identifier] = metrics
                
                # Compute weighted score
                if self.coverage_weight > 0:
                    coverage_factor = metrics['coverage'] ** self.coverage_weight
                    results['coverage_factors'][identifier] = coverage_factor
                    results['scores'][identifier] = log_prob * coverage_factor
                else:
                    results['coverage_factors'][identifier] = 1.0
                    results['scores'][identifier] = log_prob
                    
            except Exception as e:
                logger.warning(f"Error with model {identifier}: {e}")
                results['scores'][identifier] = float('-inf')
                results['raw_log_probs'][identifier] = float('-inf')
        
        # Find best prediction
        best_aa = max(results['scores'].keys(), key=lambda k: results['scores'][k])
        results['prediction'] = best_aa
        results['best_score'] = results['scores'][best_aa]
        
        # Category-level results if applicable
        if self.classification_mode != '20way':
            category_scores = self._aggregate_scores_by_category(results['scores'])
            results['category_scores'] = category_scores
            results['category_prediction'] = max(category_scores.keys(), key=lambda k: category_scores[k])
        
        return results

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
                if aa in aa_scores and not aa.startswith('_')
            ]

            if relevant_scores:
                category_scores[category] = max(relevant_scores)
            else:
                category_scores[category] = float('-inf')

        return category_scores
    
    def set_coverage_weight(self, weight: float) -> None:
        """Update the coverage weight for scoring."""
        self.coverage_weight = weight
        logger.info(f"Coverage weight set to {weight}")
    
    def get_confusion_analysis(
        self,
        true_aa: str,
        observation_sequence: npt.NDArray[np.float64]
    ) -> Dict[str, Any]:
        """
        Analyze why a prediction might be confused with the true amino acid.
        
        Useful for debugging classification errors.
        """
        details = self.predict_with_details(observation_sequence)
        predicted = details['prediction']
        
        analysis = {
            'true_aa': true_aa,
            'predicted_aa': predicted,
            'correct': true_aa == predicted,
            'true_aa_rank': None,
            'score_gap': None,
            'coverage_comparison': {}
        }
        
        # Rank the true AA
        sorted_scores = sorted(details['scores'].items(), key=lambda x: x[1], reverse=True)
        for rank, (aa, _) in enumerate(sorted_scores, 1):
            if aa == true_aa:
                analysis['true_aa_rank'] = rank
                break
        
        # Score gap
        if true_aa in details['scores']:
            analysis['score_gap'] = details['scores'][predicted] - details['scores'][true_aa]
        
        # Coverage comparison
        for aa in [true_aa, predicted]:
            if aa in details['path_metrics']:
                analysis['coverage_comparison'][aa] = {
                    'coverage': details['path_metrics'][aa]['coverage'],
                    'smoothness': details['path_metrics'][aa]['smoothness'],
                    'log_prob': details['raw_log_probs'].get(aa, float('-inf'))
                }
        
        return analysis