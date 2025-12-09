"""HMM model construction with Match, Insert, Skip, and Slip states.

Supports variable-length profiles per amino acid.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from vrhmm.yahmm import yahmm

logger = logging.getLogger(__name__)


class HMMConstructor:
    """Constructs HMM models for nanopore data with variable rate states."""

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            variance_mode: str = 'barycenter',
            variance_scale: float = 80.0,
            enforce_length: bool = False,  # NEW: Option to enforce fixed length
            default_length: int = 35       # NEW: Default only used if enforce_length=True
    ) -> None:
        if config is None:
            from vrhmm.config import CONFIG
            self.config = CONFIG['hmm']
        else:
            self.config = config

        self.variance_mode = variance_mode
        self.variance_scale = variance_scale
        self.enforce_length = enforce_length
        self.default_length = default_length

        logger.info(f"Initialized HMMConstructor with variance_mode={variance_mode}, "
                    f"enforce_length={enforce_length}")

    def build_hmm_from_arrays(
            self,
            amino_acid: str,
            profile_arrays: List[npt.NDArray[np.float64]],
            segment_variances: Optional[List[npt.NDArray[np.float64]]] = None,
            model_name: Optional[str] = None,
            expected_length: Optional[int] = None  # Now optional - uses profile length if None
    ) -> yahmm.Model:
        """Build an HMM model from profile arrays.
        
        Args:
            amino_acid: Single letter amino acid code
            profile_arrays: List of segment arrays from barycenter
            segment_variances: Optional empirical variances per segment
            model_name: Optional custom model name
            expected_length: If None, uses len(profile_arrays). If set, will pad/truncate.
        
        Returns:
            yahmm.Model with match states equal to len(profile_arrays) or expected_length
        """
        # Determine actual number of states from profile
        actual_length = len(profile_arrays)
        
        # Decide target length
        if expected_length is not None:
            target_length = expected_length
        elif self.enforce_length:
            target_length = self.default_length
        else:
            target_length = actual_length  # Use natural profile length
        
        segment_dict = {}

        for i, segment_array in enumerate(profile_arrays):
            seg_mean = float(np.mean(segment_array))

            if self.variance_mode == 'segment' and segment_variances is not None:
                if i < len(segment_variances):
                    variance_array = segment_variances[i]
                    if len(variance_array) > 0:
                        seg_var = float(np.mean(variance_array))
                        seg_std = float(np.sqrt(seg_var))
                    else:
                        seg_std = float(np.std(segment_array))
                else:
                    seg_std = float(np.std(segment_array))
            else:
                seg_var = float(np.var(segment_array))
                scaled_var = seg_var * self.variance_scale
                seg_std = float(np.sqrt(scaled_var))

            if seg_std < 1e-10:
                seg_std = 1.0

            segment_dict[str(i)] = (seg_mean, seg_std)

        if model_name is None:
            model_name = f"HMM_{amino_acid}_{self.variance_mode}_n{actual_length}"

        return self._build_hmm(segment_dict, model_name, target_length, amino_acid)

    def _build_hmm(
            self,
            segment_dict: Dict[str, Tuple[float, float]],
            model_name: Optional[str] = None,
            target_length: Optional[int] = None,
            amino_acid: str = "?"
    ) -> yahmm.Model:
        """Build HMM with specified segment statistics.
        
        Now respects the actual profile length unless explicitly told to enforce a length.
        """
        segment_list = list(segment_dict.keys())
        actual_length = len(segment_list)
        
        # If no target specified, use actual length
        if target_length is None:
            target_length = actual_length

        if actual_length != target_length:
            if self.enforce_length:
                # Old behavior: pad or truncate
                logger.warning(
                    f"[{amino_acid}] Profile has {actual_length} segments, "
                    f"enforcing {target_length} (enforce_length=True)"
                )
                if actual_length < target_length:
                    # Pad with last segment stats
                    last_key = segment_list[-1]
                    last_stats = segment_dict[last_key]
                    for i in range(actual_length, target_length):
                        new_key = str(i)
                        segment_dict[new_key] = last_stats
                        segment_list.append(new_key)
                else:
                    # Truncate
                    segment_list = segment_list[:target_length]
            else:
                # NEW behavior: accept the natural length, just log it
                logger.info(
                    f"[{amino_acid}] Building HMM with {actual_length} match states "
                    f"(profile-defined length)"
                )
                target_length = actual_length

        probs = self.config.get('transitions', {})
        return self._create_profile_hmm(segment_list, segment_dict, probs, model_name)

    def _create_profile_hmm(
            self,
            segment_list: List[str],
            segment_dict: Dict[str, Tuple[float, float]],
            probabilities: Dict[str, float],
            model_name: Optional[str] = None
    ) -> yahmm.Model:
        model = yahmm.Model(name=model_name or "Profile_HMM")
        n_states = len(segment_list)

        # Extract and normalize probabilities
        probs = self._get_normalized_probabilities(probabilities)

        # Create states
        match_states, insert_states, skip_states, slip_states = self._create_states(
            model, segment_list, segment_dict
        )

        # Add transitions
        self._add_transitions(
            model, match_states, insert_states, skip_states, slip_states, probs
        )

        model.bake()

        # Count non-None states for logging
        n_insert = len([s for s in insert_states if s is not None])
        n_skip = len([s for s in skip_states if s is not None])
        n_slip = len([s for s in slip_states if s is not None])

        logger.info(
            f"Profile HMM '{model_name}' created with {len(match_states)} match, "
            f"{n_insert} insert, {n_skip} skip, and {n_slip} slip states"
        )

        return model

    def _get_normalized_probabilities(
            self,
            probabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract and normalize transition probabilities."""
        probs = {
            'match_self_loop': probabilities.get('match_self_loop', 0.05),
            'match_forward': probabilities.get('forward', 0.80),
            'match_to_skip': probabilities.get('to_skip', 0.03),
            'match_to_slip': probabilities.get('to_slip', 0.02),
            'match_to_insert': probabilities.get('to_insert', 0.03),
            'match_to_end': probabilities.get('to_end', 0.02),
            'insert_self_loop': probabilities.get('insert_self_loop', 0.3),
            'insert_to_match': probabilities.get('insert_to_match', 0.7),
            'skip_continue': probabilities.get('skip_continue', 0.1),
            'skip_to_match': probabilities.get('skip_to_match', 0.9),
            'slip_continue': probabilities.get('slip_continue', 0.1),
            'slip_to_match': probabilities.get('slip_to_match', 0.9)
        }

        # Normalize match transitions
        match_sum = sum([
            probs['match_self_loop'], probs['match_forward'],
            probs['match_to_skip'], probs['match_to_slip'],
            probs['match_to_insert'], probs['match_to_end']
        ])

        if match_sum > 0:
            for key in ['match_self_loop', 'match_forward', 'match_to_skip',
                        'match_to_slip', 'match_to_insert', 'match_to_end']:
                probs[key] /= match_sum

        return probs

    def _create_states(
            self,
            model: yahmm.Model,
            segment_list: List[str],
            segment_dict: Dict[str, Tuple[float, float]]
    ) -> Tuple[List[Any], List[Optional[Any]], List[Optional[Any]], List[Optional[Any]]]:
        n_states = len(segment_list)
        match_states = []
        insert_states = []
        skip_states = []
        slip_states = []

        for i, seg_label in enumerate(segment_list):
            seg_mean, seg_std = segment_dict[seg_label]

            # Match state
            match_st = yahmm.State(
                yahmm.NormalDistribution(seg_mean, seg_std),
                name=f"Match_{i}"
            )
            model.add_state(match_st)
            match_states.append(match_st)

            # Insert state (between current and next)
            if i < n_states - 1:
                next_seg_mean, next_seg_std = segment_dict[segment_list[i + 1]]
                insert_mean = (seg_mean + next_seg_mean) / 2.0
                insert_variance = (1 / 4) * ((seg_mean - next_seg_mean) ** 2) + \
                                  (1 / 2) * (seg_std ** 2 + next_seg_std ** 2)
                insert_std = np.sqrt(insert_variance)

                if insert_std < 1e-6:
                    insert_std = max(seg_std, next_seg_std, 0.1)

                insert_st = yahmm.State(
                    yahmm.NormalDistribution(insert_mean, insert_std),
                    name=f"Insert_{i}_{i + 1}"
                )
                model.add_state(insert_st)
                insert_states.append(insert_st)
            else:
                insert_states.append(None)

            # Skip state - dynamic boundary check based on actual n_states
            # Protect first 2 and last 2 states from being skipped
            is_required = (i < 2 or i >= n_states - 2)
            if i < n_states - 1 and not is_required:
                skip_st = yahmm.State(None, name=f"Skip_{i}")
                model.add_state(skip_st)
                skip_states.append(skip_st)
            else:
                skip_states.append(None)

            # Slip state (backslip from current to previous)
            if i > 1:
                slip_st = yahmm.State(None, name=f"Slip_{i}")
                model.add_state(slip_st)
                slip_states.append(slip_st)
            else:
                slip_states.append(None)

        return match_states, insert_states, skip_states, slip_states

    def _add_transitions(
            self,
            model: yahmm.Model,
            match_states: List[Any],
            insert_states: List[Optional[Any]],
            skip_states: List[Optional[Any]],
            slip_states: List[Optional[Any]],
            probs: Dict[str, float]
    ) -> None:
        n_states = len(match_states)

        # Start transition
        model.add_transition(model.start, match_states[0], 1.0)

        # Match state transitions
        for i in range(n_states):
            match_i = match_states[i]
            is_last = (i == n_states - 1)
            
            # Dynamic boundary protection based on actual model size
            is_required_start = (i < 2)
            is_required_end = (i >= n_states - 2)

            # Self-loop
            model.add_transition(match_i, match_i, probs['match_self_loop'])

            # Forward to next match
            if not is_last:
                if is_required_start or is_required_end:
                    forward_prob = probs['match_forward'] + probs['match_to_insert']
                else:
                    forward_prob = probs['match_forward']
                model.add_transition(match_i, match_states[i + 1], forward_prob)

            # To insert state
            if insert_states[i] is not None and not is_required_start and not is_required_end:
                model.add_transition(match_i, insert_states[i], probs['match_to_insert'])

            # To skip state - check bounds dynamically
            if skip_states[i] is not None and not is_required_start and not is_required_end:
                if i < n_states - 3:  # Can't skip too close to end
                    model.add_transition(match_i, skip_states[i], probs['match_to_skip'])

            # To slip state
            if i > 1 and slip_states[i] is not None and not is_required_start:
                model.add_transition(match_i, slip_states[i], probs['match_to_slip'])

            # To end
            if is_last:
                model.add_transition(match_i, model.end, 1.0 - probs['match_self_loop'])
            elif not is_required_start and not is_required_end and i > 1 and i < n_states - 3:
                model.add_transition(match_i, model.end, probs['match_to_end'] * 0.1)

        # Insert state transitions
        for i in range(n_states - 1):
            if insert_states[i] is not None:
                insert_i = insert_states[i]
                model.add_transition(insert_i, insert_i, probs['insert_self_loop'])
                model.add_transition(insert_i, match_states[i + 1], probs['insert_to_match'])

        # Skip state transitions
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

        # Slip state transitions
        for i in range(2, n_states):
            if slip_states[i] is not None:
                slip_i = slip_states[i]
                model.add_transition(slip_i, match_states[i - 1], probs['slip_to_match'])

                if i > 2 and slip_states[i - 1] is not None:
                    model.add_transition(slip_i, slip_states[i - 1], probs['slip_continue'])

    def get_model_length(self, model: yahmm.Model) -> int:
        """Get the number of match states in a model."""
        match_count = sum(1 for state in model.states 
                         if hasattr(state, 'name') and state.name and 'Match' in state.name)
        return match_count