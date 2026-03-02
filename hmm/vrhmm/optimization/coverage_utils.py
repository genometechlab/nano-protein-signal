"""
Coverage-based utilities for HMM classification and optimization.

Provides tools for:
- Analyzing path quality
- Coverage-weighted classification
- Match state distribution analysis
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def compute_match_state_distribution(
    path: List[str],
    n_expected: int
) -> Dict[str, Any]:
    """
    Compute detailed match state distribution from a Viterbi path.
    
    Returns:
        Dictionary with:
        - visited: set of visited match state indices
        - visit_counts: {idx: count} for each visited state
        - gaps: list of (start, end) tuples for unvisited ranges
        - distribution_entropy: measure of how evenly states are visited
    """
    visit_counts = defaultdict(int)
    
    for state in path:
        if state and 'Match' in state:
            try:
                idx = int(state.split('_')[1])
                visit_counts[idx] += 1
            except (ValueError, IndexError):
                continue
    
    visited = set(visit_counts.keys())
    
    # Find gaps (unvisited ranges)
    gaps = []
    gap_start = None
    
    for i in range(n_expected):
        if i not in visited:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gaps.append((gap_start, i - 1))
                gap_start = None
    
    if gap_start is not None:
        gaps.append((gap_start, n_expected - 1))
    
    # Compute distribution entropy
    total_visits = sum(visit_counts.values())
    if total_visits > 0:
        probs = np.array(list(visit_counts.values())) / total_visits
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(visited)) if len(visited) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0
    
    return {
        'visited': visited,
        'visit_counts': dict(visit_counts),
        'gaps': gaps,
        'distribution_entropy': normalized_entropy,
        'n_visited': len(visited),
        'n_expected': n_expected,
        'coverage': len(visited) / n_expected if n_expected > 0 else 0.0
    }


def compute_path_transition_stats(
    path: List[str]
) -> Dict[str, Any]:
    """
    Compute transition statistics from a Viterbi path.
    
    Returns detailed breakdown of transition types observed.
    """
    stats = {
        'forward_1': 0,      # Normal forward transition
        'forward_skip': 0,   # Skipped states
        'skip_sizes': [],    # Size of each skip
        'backward': 0,       # Backslips
        'slip_sizes': [],    # Size of each backslip
        'self_loop': 0,      # Self-loops
        'total_match': 0,
        'total_insert': 0,
        'insert_lengths': [] # Consecutive insert runs
    }
    
    prev_match_idx = -1
    current_insert_run = 0
    
    for state in path:
        if state is None:
            continue
        
        if 'Match' in state:
            # End any insert run
            if current_insert_run > 0:
                stats['insert_lengths'].append(current_insert_run)
                current_insert_run = 0
            
            try:
                idx = int(state.split('_')[1])
                stats['total_match'] += 1
                
                if prev_match_idx >= 0:
                    diff = idx - prev_match_idx
                    
                    if diff == 0:
                        stats['self_loop'] += 1
                    elif diff == 1:
                        stats['forward_1'] += 1
                    elif diff > 1:
                        stats['forward_skip'] += 1
                        stats['skip_sizes'].append(diff - 1)
                    else:  # diff < 0
                        stats['backward'] += 1
                        stats['slip_sizes'].append(abs(diff))
                
                prev_match_idx = idx
                
            except (ValueError, IndexError):
                continue
                
        elif 'Insert' in state:
            stats['total_insert'] += 1
            current_insert_run += 1
    
    # Finalize any remaining insert run
    if current_insert_run > 0:
        stats['insert_lengths'].append(current_insert_run)
    
    return stats


def compute_coverage_weighted_score(
    log_prob: float,
    coverage: float,
    weight: float = 0.5,
    mode: str = 'multiplicative'
) -> float:
    """
    Compute coverage-weighted score.
    
    Args:
        log_prob: Log probability from HMM
        coverage: Coverage ratio (0-1)
        weight: Weight for coverage term
        mode: 'multiplicative', 'additive', or 'threshold'
    
    Returns:
        Weighted score
    """
    if mode == 'multiplicative':
        # Score = LL * coverage^weight
        return log_prob * (coverage ** weight)
    
    elif mode == 'additive':
        # Score = LL + weight * coverage
        # Need to scale coverage to be comparable to LL
        return log_prob + weight * 100 * coverage  # Scale factor of 100
    
    elif mode == 'threshold':
        # Full score if coverage above threshold, penalized otherwise
        threshold = 0.8
        if coverage >= threshold:
            return log_prob
        else:
            penalty = (threshold - coverage) / threshold
            return log_prob * (1 - weight * penalty)
    
    else:
        return log_prob


def rank_models_by_fit(
    trace: np.ndarray,
    models: Dict[str, Any],
    model_lengths: Dict[str, int],
    coverage_weight: float = 0.5
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Rank all models by their fit to a trace.
    
    Returns list of (aa, score, details) sorted by score descending.
    """
    results = []
    
    for aa, model in models.items():
        try:
            log_prob = model.log_probability(trace)
            _, path_raw = model.viterbi(trace)
            
            # Extract state names
            path = []
            for item in path_raw:
                if hasattr(item, 'name'):
                    path.append(item.name)
                elif isinstance(item, tuple) and len(item) >= 2:
                    state_obj = item[1]
                    if hasattr(state_obj, 'name'):
                        path.append(state_obj.name)
            
            n_expected = model_lengths.get(aa, 35)
            distribution = compute_match_state_distribution(path, n_expected)
            transition_stats = compute_path_transition_stats(path)
            
            weighted_score = compute_coverage_weighted_score(
                log_prob, distribution['coverage'], coverage_weight
            )
            
            results.append((aa, weighted_score, {
                'log_prob': log_prob,
                'coverage': distribution['coverage'],
                'n_visited': distribution['n_visited'],
                'n_expected': n_expected,
                'gaps': distribution['gaps'],
                'n_skips': transition_stats['forward_skip'],
                'n_slips': transition_stats['backward'],
                'n_self_loops': transition_stats['self_loop']
            }))
            
        except Exception as e:
            logger.debug(f"Error scoring {aa}: {e}")
            results.append((aa, float('-inf'), {'error': str(e)}))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def diagnose_classification_error(
    trace: np.ndarray,
    true_aa: str,
    predicted_aa: str,
    models: Dict[str, Any],
    model_lengths: Dict[str, int]
) -> Dict[str, Any]:
    """
    Diagnose why a classification error occurred.
    
    Provides detailed comparison between true and predicted model fits.
    """
    diagnosis = {
        'true_aa': true_aa,
        'predicted_aa': predicted_aa,
        'models': {}
    }
    
    for aa in [true_aa, predicted_aa]:
        if aa not in models:
            diagnosis['models'][aa] = {'error': 'Model not found'}
            continue
        
        model = models[aa]
        
        try:
            log_prob = model.log_probability(trace)
            _, path_raw = model.viterbi(trace)
            
            path = []
            for item in path_raw:
                if hasattr(item, 'name'):
                    path.append(item.name)
                elif isinstance(item, tuple) and len(item) >= 2:
                    state_obj = item[1]
                    if hasattr(state_obj, 'name'):
                        path.append(state_obj.name)
            
            n_expected = model_lengths.get(aa, 35)
            distribution = compute_match_state_distribution(path, n_expected)
            transition_stats = compute_path_transition_stats(path)
            
            diagnosis['models'][aa] = {
                'log_prob': log_prob,
                'normalized_ll': log_prob / len(trace),
                'coverage': distribution['coverage'],
                'n_visited': distribution['n_visited'],
                'n_expected': n_expected,
                'gaps': distribution['gaps'],
                'visit_counts': distribution['visit_counts'],
                'n_skips': transition_stats['forward_skip'],
                'skip_sizes': transition_stats['skip_sizes'],
                'n_slips': transition_stats['backward'],
                'slip_sizes': transition_stats['slip_sizes'],
                'n_self_loops': transition_stats['self_loop'],
                'n_inserts': transition_stats['total_insert']
            }
            
        except Exception as e:
            diagnosis['models'][aa] = {'error': str(e)}
    
    # Compute comparison metrics
    if true_aa in diagnosis['models'] and predicted_aa in diagnosis['models']:
        true_data = diagnosis['models'][true_aa]
        pred_data = diagnosis['models'][predicted_aa]
        
        if 'error' not in true_data and 'error' not in pred_data:
            diagnosis['comparison'] = {
                'll_gap': pred_data['log_prob'] - true_data['log_prob'],
                'coverage_gap': pred_data['coverage'] - true_data['coverage'],
                'true_has_better_coverage': true_data['coverage'] > pred_data['coverage'],
                'true_has_fewer_skips': true_data['n_skips'] < pred_data['n_skips'],
                'true_has_fewer_slips': true_data['n_slips'] < pred_data['n_slips']
            }
            
            # Suggest potential fix
            if diagnosis['comparison']['true_has_better_coverage']:
                diagnosis['suggestion'] = 'Increase coverage_weight to favor better path coverage'
            elif diagnosis['comparison']['ll_gap'] < 5:
                diagnosis['suggestion'] = 'Models are close; may need more training data'
            else:
                diagnosis['suggestion'] = 'True model has significantly worse LL; check profile quality'
    
    return diagnosis