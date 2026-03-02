"""
Model builder for optimization - direct Python HMM construction.

Builds HMM models without subprocess calls for faster optimization.
"""

import logging
from typing import Dict, Tuple, Any, Optional, List

import numpy as np

from vrhmm.yahmm import yahmm

logger = logging.getLogger(__name__)


class OptimizationModelBuilder:
    """
    Builds HMM models directly for optimization.
    
    This bypasses the CLI and subprocess calls for much faster
    model construction during optimization.
    """
    
    def __init__(
        self,
        profile_stats: Dict[str, Dict[str, Tuple[float, float]]],
        default_transitions: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            profile_stats: {aa: {state_idx: (mean, std)}} pre-loaded profiles
            default_transitions: Default transition probabilities
        """
        self.profile_stats = profile_stats
        
        if default_transitions is None:
            self.default_transitions = {
                'match_self_loop': 0.05,
                'forward': 0.65,
                'to_skip': 0.15,
                'to_slip': 0.07,
                'to_insert': 0.03,
                'to_end': 0.05,
                'insert_self_loop': 0.3,
                'insert_to_match': 0.7,
                'skip_to_match': 0.9,
                'skip_continue': 0.1,
                'slip_to_match': 0.92,
                'slip_continue': 0.08
            }
        else:
            self.default_transitions = default_transitions
        
        # Cache for built models
        self._model_cache: Dict[str, Any] = {}
    
    def build_model(
        self,
        aa: str,
        variance_scale: float = 1.0,
        transitions: Optional[Dict[str, float]] = None,
        use_cache: bool = False
    ) -> yahmm.Model:
        """
        Build an HMM model for a specific amino acid.
        
        Args:
            aa: Amino acid single-letter code
            variance_scale: Variance scaling factor
            transitions: Transition probabilities (uses defaults if None)
            use_cache: Whether to cache and reuse models
        
        Returns:
            yahmm.Model ready for scoring
        """
        # Check cache
        if use_cache:
            cache_key = f"{aa}_{variance_scale}_{hash(frozenset((transitions or {}).items()))}"
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]
        
        if aa not in self.profile_stats:
            raise ValueError(f"No profile stats for amino acid: {aa}")
        
        profile = self.profile_stats[aa]
        
        # Merge transitions with defaults
        trans = self.default_transitions.copy()
        if transitions:
            trans.update(transitions)
        
        # Normalize match state transitions
        trans = self._normalize_transitions(trans)
        
        # Build the model
        model = self._create_profile_hmm(aa, profile, variance_scale, trans)
        
        if use_cache:
            self._model_cache[cache_key] = model
        
        return model
    
    def _normalize_transitions(self, trans: Dict[str, float]) -> Dict[str, float]:
        """Normalize transition probabilities."""
        normalized = trans.copy()
        
        # Match state outgoing transitions must sum to 1
        match_keys = ['match_self_loop', 'forward', 'to_skip', 'to_slip', 'to_insert', 'to_end']
        match_total = sum(trans.get(k, 0) for k in match_keys)
        
        if match_total > 0:
            for key in match_keys:
                if key in trans:
                    normalized[key] = trans[key] / match_total
        
        return normalized
    
    def _create_profile_hmm(
        self,
        aa: str,
        profile: Dict[str, Tuple[float, float]],
        variance_scale: float,
        transitions: Dict[str, float]
    ) -> yahmm.Model:
        """
        Create a profile HMM with Match, Insert, Skip, and Slip states.
        
        Matches the structure from hmm_builder.py with proper required regions.
        """
        model_name = f"HMM_{aa}_opt"
        model = yahmm.Model(name=model_name)
        
        # Sort states by index
        state_indices = sorted([int(k) for k in profile.keys()])
        n_states = len(state_indices)
        
        # Create state containers
        match_states = []
        insert_states = []
        skip_states = []
        slip_states = []
        
        # Build states
        for i, state_idx in enumerate(state_indices):
            mean, std = profile[str(state_idx)]
            
            # Apply variance scaling
            scaled_std = std * np.sqrt(variance_scale)
            if scaled_std < 1e-10:
                scaled_std = 1.0
            
            # Match state
            match_dist = yahmm.NormalDistribution(mean, scaled_std)
            match_state = yahmm.State(match_dist, name=f"Match_{i}")
            model.add_state(match_state)
            match_states.append(match_state)
            
            # Insert state (between this and next match)
            if i < n_states - 1:
                next_mean, next_std = profile[str(state_indices[i + 1])]
                next_scaled_std = next_std * np.sqrt(variance_scale)
                
                insert_mean = (mean + next_mean) / 2.0
                insert_var = (1/4) * ((mean - next_mean) ** 2) + (1/2) * (scaled_std**2 + next_scaled_std**2)
                insert_std = np.sqrt(insert_var)
                
                if insert_std < 1e-6:
                    insert_std = max(scaled_std, next_scaled_std, 0.1)
                
                insert_dist = yahmm.NormalDistribution(insert_mean, insert_std)
                insert_state = yahmm.State(insert_dist, name=f"Insert_{i}_{i+1}")
                model.add_state(insert_state)
                insert_states.append(insert_state)
            else:
                insert_states.append(None)
            
            # Skip state (silent, for skipping forward)
            # Required regions: first 2 and last 2 states cannot be skipped
            is_required = (i < 2 or i >= n_states - 2)
            if i < n_states - 1 and not is_required:
                skip_state = yahmm.State(None, name=f"Skip_{i}")
                model.add_state(skip_state)
                skip_states.append(skip_state)
            else:
                skip_states.append(None)
            
            # Slip state (silent, for going backward)
            # Only available after position 1
            if i > 1:
                slip_state = yahmm.State(None, name=f"Slip_{i}")
                model.add_state(slip_state)
                slip_states.append(slip_state)
            else:
                slip_states.append(None)
        
        # Add transitions
        self._add_transitions(
            model, match_states, insert_states, skip_states, slip_states, 
            transitions, n_states
        )
        
        # Finalize
        model.bake()
        
        return model
    
    def _add_transitions(
        self,
        model: yahmm.Model,
        match_states: List[yahmm.State],
        insert_states: List[Optional[yahmm.State]],
        skip_states: List[Optional[yahmm.State]],
        slip_states: List[Optional[yahmm.State]],
        probs: Dict[str, float],
        n_states: int
    ) -> None:
        """Add all transitions to the model matching hmm_builder.py logic."""
        
        # Start -> first match
        model.add_transition(model.start, match_states[0], 1.0)
        
        for i in range(n_states):
            match_i = match_states[i]
            is_last = (i == n_states - 1)
            is_required_start = (i < 2)
            is_required_end = (i >= n_states - 2)
            
            # Self-loop
            model.add_transition(match_i, match_i, probs['match_self_loop'])
            
            # Forward to next match
            if not is_last:
                if is_required_start or is_required_end:
                    # In required regions, combine forward + insert probability
                    forward_prob = probs['forward'] + probs['to_insert']
                else:
                    forward_prob = probs['forward']
                model.add_transition(match_i, match_states[i + 1], forward_prob)
            
            # To insert (only in non-required regions)
            if insert_states[i] is not None and not is_required_start and not is_required_end:
                model.add_transition(match_i, insert_states[i], probs['to_insert'])
            
            # To skip (only in non-required regions and not too close to end)
            if skip_states[i] is not None and not is_required_start and not is_required_end:
                if i < n_states - 3:
                    model.add_transition(match_i, skip_states[i], probs['to_skip'])
            
            # To slip (only after position 1 and not in required start)
            if i > 1 and slip_states[i] is not None and not is_required_start:
                model.add_transition(match_i, slip_states[i], probs['to_slip'])
            
            # To end
            if is_last:
                model.add_transition(match_i, model.end, 1.0 - probs['match_self_loop'])
            elif not is_required_start and not is_required_end and i > 1 and i < n_states - 3:
                model.add_transition(match_i, model.end, probs['to_end'] * 0.1)
        
        # Insert transitions
        for i in range(n_states - 1):
            if insert_states[i] is not None:
                insert_i = insert_states[i]
                model.add_transition(insert_i, insert_i, probs['insert_self_loop'])
                model.add_transition(insert_i, match_states[i + 1], probs['insert_to_match'])
        
        # Skip transitions
        for i in range(n_states - 1):
            if skip_states[i] is not None:
                skip_i = skip_states[i]
                if i + 1 < n_states:
                    if i + 1 < n_states - 2:
                        model.add_transition(skip_i, match_states[i + 1], probs['skip_to_match'])
                    else:
                        model.add_transition(skip_i, match_states[i + 1], 1.0)
                
                if i + 1 < n_states - 3 and skip_states[i + 1] is not None:
                    model.add_transition(skip_i, skip_states[i + 1], probs['skip_continue'])
        
        # Slip transitions
        for i in range(2, n_states):
            if slip_states[i] is not None:
                slip_i = slip_states[i]
                model.add_transition(slip_i, match_states[i - 1], probs['slip_to_match'])
                
                if i > 2 and slip_states[i - 1] is not None:
                    model.add_transition(slip_i, slip_states[i - 1], probs['slip_continue'])
    
    def get_model_length(self, aa: str) -> int:
        """Get the number of match states for an amino acid."""
        if aa in self.profile_stats:
            return len(self.profile_stats[aa])
        return 35  # Default
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()
    
    def get_profile_stats(self, aa: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """Get profile statistics for an amino acid."""
        return self.profile_stats.get(aa)