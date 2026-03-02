"""
Multi-objective training functions for HMM optimization.

Combines:
- Log-likelihood (model fit)
- Match state coverage (observed-to-expected ratio)
- Path smoothness (penalize irregularities)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for multi-objective training."""
    
    # Objective weights
    alpha: float = 1.0    # Log-likelihood weight
    beta: float = 0.5     # Coverage weight
    gamma: float = 0.3    # Smoothness weight
    
    # Normalization options
    normalize_ll_by_length: bool = True
    coverage_penalty_mode: str = 'linear'  # 'linear', 'quadratic', 'threshold'
    coverage_threshold: float = 0.8  # For threshold mode
    
    # Path quality options
    skip_penalty: float = 1.0
    slip_penalty: float = 1.0
    self_loop_penalty: float = 0.5
    insert_penalty: float = 0.3


@dataclass
class PathMetrics:
    """Metrics extracted from a Viterbi path."""
    
    coverage: float = 0.0          # Unique matches / expected matches
    efficiency: float = 0.0        # Unique matches / total match visits
    smoothness: float = 0.0        # 1 - (irregularities / emissions)
    
    n_unique_matches: int = 0
    n_total_matches: int = 0
    n_expected_matches: int = 0
    n_skips: int = 0
    n_slips: int = 0
    n_self_loops: int = 0
    n_inserts: int = 0
    
    match_indices: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'coverage': self.coverage,
            'efficiency': self.efficiency,
            'smoothness': self.smoothness,
            'n_unique_matches': self.n_unique_matches,
            'n_total_matches': self.n_total_matches,
            'n_expected_matches': self.n_expected_matches,
            'n_skips': self.n_skips,
            'n_slips': self.n_slips,
            'n_self_loops': self.n_self_loops,
            'n_inserts': self.n_inserts
        }


def analyze_viterbi_path(
    path: List[str],
    n_expected_matches: int,
    config: Optional[TrainingConfig] = None
) -> PathMetrics:
    """
    Analyze a Viterbi path for quality metrics.
    
    Args:
        path: List of state names from Viterbi decoding
        n_expected_matches: Number of match states in the model
        config: Training configuration (for penalty weights)
    
    Returns:
        PathMetrics with coverage, efficiency, smoothness, and raw counts
    """
    if config is None:
        config = TrainingConfig()
    
    metrics = PathMetrics(n_expected_matches=n_expected_matches)
    
    match_indices = []
    prev_match_idx = -1
    prev_state = None
    
    for state in path:
        if state is None:
            continue
            
        if 'Match' in state:
            try:
                idx = int(state.split('_')[1])
                match_indices.append(idx)
                
                if prev_match_idx >= 0:
                    if idx > prev_match_idx + 1:
                        # Skip: jumped forward more than 1
                        metrics.n_skips += (idx - prev_match_idx - 1)
                    elif idx < prev_match_idx:
                        # Slip: jumped backward
                        metrics.n_slips += 1
                    elif idx == prev_match_idx and state == prev_state:
                        # Self-loop: same state repeated
                        metrics.n_self_loops += 1
                
                prev_match_idx = idx
                prev_state = state
                
            except (ValueError, IndexError):
                continue
                
        elif 'Insert' in state:
            metrics.n_inserts += 1
    
    metrics.match_indices = match_indices
    metrics.n_unique_matches = len(set(match_indices))
    metrics.n_total_matches = len(match_indices)
    
    # Coverage: fraction of expected match states visited
    if n_expected_matches > 0:
        metrics.coverage = metrics.n_unique_matches / n_expected_matches
    
    # Efficiency: unique visits / total visits (1.0 = no revisits)
    if metrics.n_total_matches > 0:
        metrics.efficiency = metrics.n_unique_matches / metrics.n_total_matches
    
    # Smoothness: penalize irregularities
    total_emissions = metrics.n_total_matches + metrics.n_inserts
    if total_emissions > 0:
        weighted_irregularities = (
            config.skip_penalty * metrics.n_skips +
            config.slip_penalty * metrics.n_slips +
            config.self_loop_penalty * metrics.n_self_loops +
            config.insert_penalty * metrics.n_inserts
        )
        metrics.smoothness = max(0.0, 1.0 - weighted_irregularities / total_emissions)
    
    return metrics


def compute_coverage_score(
    metrics: PathMetrics,
    config: TrainingConfig
) -> float:
    """
    Compute coverage score based on configuration.
    
    Different modes:
    - 'linear': Direct coverage ratio
    - 'quadratic': Squared coverage (penalizes low coverage more)
    - 'threshold': Binary score based on threshold
    """
    if config.coverage_penalty_mode == 'linear':
        return metrics.coverage
    
    elif config.coverage_penalty_mode == 'quadratic':
        return metrics.coverage ** 2
    
    elif config.coverage_penalty_mode == 'threshold':
        if metrics.coverage >= config.coverage_threshold:
            return 1.0
        else:
            # Linear interpolation below threshold
            return metrics.coverage / config.coverage_threshold
    
    else:
        return metrics.coverage


class MultiObjectiveTrainer:
    """
    Trainer for multi-objective HMM optimization.
    
    Evaluates models based on:
    1. Log-likelihood of traces under their own AA model
    2. Match state coverage
    3. Path smoothness
    """
    
    def __init__(
        self,
        model_builder: Any,
        profile_stats: Dict[str, Dict[str, Tuple[float, float]]]
    ):
        """
        Args:
            model_builder: OptimizationModelBuilder instance
            profile_stats: Pre-loaded profile statistics {aa: {state: (mean, std)}}
        """
        self.model_builder = model_builder
        self.profile_stats = profile_stats
    
    def compute_objective(
        self,
        variance_scales: Dict[str, float],
        transitions: Dict[str, Dict[str, float]],
        test_traces: Dict[str, List[np.ndarray]],
        config: Optional[TrainingConfig] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute multi-objective training score.
        
        Args:
            variance_scales: Per-AA variance scales
            transitions: Per-AA transition probabilities
            test_traces: {aa: [trace1, trace2, ...]} test data
            config: Training configuration
        
        Returns:
            Tuple of (total_score, detailed_metrics)
        """
        if config is None:
            config = TrainingConfig()
        
        total_score = 0.0
        metrics = {
            'per_aa': {},
            'total_ll': 0.0,
            'total_coverage': 0.0,
            'total_smoothness': 0.0,
            'total_efficiency': 0.0,
            'n_traces': 0,
            'n_aa': 0
        }
        
        for aa in variance_scales.keys():
            if aa not in test_traces or not test_traces[aa]:
                continue
            
            # Get transitions for this AA
            aa_transitions = transitions.get(aa, transitions.get(list(transitions.keys())[0], {}))
            
            # Build model
            try:
                model = self.model_builder.build_model(
                    aa=aa,
                    variance_scale=variance_scales[aa],
                    transitions=aa_transitions
                )
            except Exception as e:
                logger.warning(f"Failed to build model for {aa}: {e}")
                continue
            
            n_expected = len(self.profile_stats.get(aa, {}))
            
            aa_ll = 0.0
            aa_coverage = 0.0
            aa_smoothness = 0.0
            aa_efficiency = 0.0
            aa_traces = 0
            
            for trace in test_traces[aa]:
                try:
                    # 1. Log-likelihood
                    ll = model.log_probability(trace)
                    
                    if config.normalize_ll_by_length:
                        ll = ll / max(1, len(trace))
                    
                    # 2. Viterbi path analysis
                    _, path_raw = model.viterbi(trace)
                    
                    # Extract state names from path
                    path = []
                    if path_raw:
                        for item in path_raw:
                            if hasattr(item, 'name'):
                                path.append(item.name)
                            elif isinstance(item, tuple) and len(item) >= 2:
                                state_obj = item[1]
                                if hasattr(state_obj, 'name'):
                                    path.append(state_obj.name)
                    
                    path_metrics = analyze_viterbi_path(path, n_expected, config)
                    
                    # 3. Compute component scores
                    coverage_score = compute_coverage_score(path_metrics, config)
                    
                    # 4. Combine into trace score
                    trace_score = (
                        config.alpha * ll +
                        config.beta * coverage_score +
                        config.gamma * path_metrics.smoothness
                    )
                    
                    total_score += trace_score
                    aa_ll += ll
                    aa_coverage += path_metrics.coverage
                    aa_smoothness += path_metrics.smoothness
                    aa_efficiency += path_metrics.efficiency
                    aa_traces += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing trace for {aa}: {e}")
                    continue
            
            if aa_traces > 0:
                metrics['per_aa'][aa] = {
                    'mean_ll': aa_ll / aa_traces,
                    'mean_coverage': aa_coverage / aa_traces,
                    'mean_smoothness': aa_smoothness / aa_traces,
                    'mean_efficiency': aa_efficiency / aa_traces,
                    'n_traces': aa_traces
                }
                
                metrics['total_ll'] += aa_ll
                metrics['total_coverage'] += aa_coverage
                metrics['total_smoothness'] += aa_smoothness
                metrics['total_efficiency'] += aa_efficiency
                metrics['n_traces'] += aa_traces
                metrics['n_aa'] += 1
        
        # Compute means
        if metrics['n_traces'] > 0:
            metrics['mean_ll'] = metrics['total_ll'] / metrics['n_traces']
            metrics['mean_coverage'] = metrics['total_coverage'] / metrics['n_traces']
            metrics['mean_smoothness'] = metrics['total_smoothness'] / metrics['n_traces']
            metrics['mean_efficiency'] = metrics['total_efficiency'] / metrics['n_traces']
        else:
            metrics['mean_ll'] = 0.0
            metrics['mean_coverage'] = 0.0
            metrics['mean_smoothness'] = 0.0
            metrics['mean_efficiency'] = 0.0
        
        return total_score, metrics
    
    def compute_classification_accuracy(
        self,
        variance_scales: Dict[str, float],
        transitions: Dict[str, Dict[str, float]],
        test_traces: Dict[str, List[np.ndarray]],
        coverage_weight: float = 0.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute classification accuracy (for monitoring, not optimization).
        
        Args:
            variance_scales: Per-AA variance scales
            transitions: Per-AA transition probabilities
            test_traces: {aa: [trace1, trace2, ...]} test data
            coverage_weight: If > 0, weight scores by coverage
        
        Returns:
            Tuple of (accuracy, detailed_results)
        """
        # Build all models
        models = {}
        n_expected = {}
        
        for aa in variance_scales.keys():
            aa_transitions = transitions.get(aa, transitions.get(list(transitions.keys())[0], {}))
            
            try:
                models[aa] = self.model_builder.build_model(
                    aa=aa,
                    variance_scale=variance_scales[aa],
                    transitions=aa_transitions
                )
                n_expected[aa] = len(self.profile_stats.get(aa, {}))
            except Exception as e:
                logger.warning(f"Failed to build model for {aa}: {e}")
        
        correct = 0
        total = 0
        results = []
        
        for true_aa, traces in test_traces.items():
            if true_aa not in models:
                continue
            
            for trace in traces:
                scores = {}
                
                for aa, model in models.items():
                    try:
                        ll = model.log_probability(trace)
                        
                        if coverage_weight > 0:
                            _, path_raw = model.viterbi(trace)
                            path = []
                            if path_raw:
                                for item in path_raw:
                                    if hasattr(item, 'name'):
                                        path.append(item.name)
                                    elif isinstance(item, tuple) and len(item) >= 2:
                                        state_obj = item[1]
                                        if hasattr(state_obj, 'name'):
                                            path.append(state_obj.name)
                            
                            path_metrics = analyze_viterbi_path(path, n_expected[aa])
                            scores[aa] = ll * (path_metrics.coverage ** coverage_weight)
                        else:
                            scores[aa] = ll
                            
                    except Exception:
                        scores[aa] = float('-inf')
                
                predicted = max(scores, key=scores.get)
                is_correct = predicted == true_aa
                
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    'true_aa': true_aa,
                    'predicted_aa': predicted,
                    'correct': is_correct,
                    'scores': scores
                })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy, {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }