"""HMM-based segment reorganization."""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check for DTW availability
try:
    from dtaidistance import dtw_barycenter

    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    logger.warning("dtaidistance not available. DTW averaging disabled.")

class HMMSegmentReorganizer:
    """Reorganize segments based on HMM alignment."""

    def __init__(self, backslip_mode: str = 'ignore') -> None:
        
        self.backslip_mode = backslip_mode
        if backslip_mode == 'average' and not DTW_AVAILABLE:
            logger.warning("DTW averaging requested but not available. Using 'delete' mode.")
            self.backslip_mode = 'delete'

    def reorganize_segments(
            self,
            original_segments: List[np.ndarray],
            full_path: List[str],
            segment_results: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
        """Reorganize segments based on HMM alignment."""
        segment_to_state_mapping = self._map_segments_to_states(full_path)
        grouped_segments = self._group_segments_by_match_state(
            original_segments, segment_to_state_mapping
        )
        reorganized = self._process_grouped_segments(grouped_segments)

        match_indices = sorted(reorganized.keys())
        reorganized_segments = []

        for idx in range(35):
            if idx in reorganized:
                reorganized_segments.append(reorganized[idx])
            else:
                reorganized_segments.append(np.array([]))

        metadata = {
            'original_count': len(original_segments),
            'reorganized_count': len(reorganized_segments),
            'backslip_mode': self.backslip_mode,
            'match_indices': match_indices,
            'original_path': full_path,
            'segment_mapping': segment_to_state_mapping
        }

        return reorganized_segments, match_indices, metadata

    def _map_segments_to_states(
            self,
            full_path: List[str]
    ) -> List[Tuple[str, int, int]]:
        """Map each observation to its aligned state."""
        mapping = []
        obs_idx = 0

        for state in full_path:
            if 'Match' in state:
                match_idx = int(state.split('_')[1])
                mapping.append(('Match', match_idx, obs_idx))
                obs_idx += 1
            elif 'Insert' in state:
                parts = state.split('_')
                insert_pos = int(parts[1]) if len(parts) > 1 else obs_idx
                mapping.append(('Insert', insert_pos, obs_idx))
                obs_idx += 1

        return mapping

    def _group_segments_by_match_state(
            self,
            segments: List[np.ndarray],
            mapping: List[Tuple[str, int, int]]
    ) -> Dict[int, List[Tuple[np.ndarray, int]]]:
        """Group segments by their match state alignment."""
        grouped = {}

        for state_type, state_idx, seg_idx in mapping:
            if seg_idx >= len(segments):
                break

            if state_type == 'Match':
                if state_idx not in grouped:
                    grouped[state_idx] = []
                grouped[state_idx].append((segments[seg_idx], seg_idx))
            elif state_type == 'Insert':
                insert_key = f"Insert_{state_idx}"
                if insert_key not in grouped:
                    grouped[insert_key] = []
                grouped[insert_key].append((segments[seg_idx], seg_idx))

        return grouped

    def _process_grouped_segments(
            self,
            grouped: Dict[int, List[Tuple[np.ndarray, int]]]
    ) -> Dict[int, np.ndarray]:
        """Process grouped segments based on backslip mode."""
        processed = {}

        for key, segment_list in grouped.items():
            if isinstance(key, str) and 'Insert' in key:
                continue

            match_idx = key

            if len(segment_list) == 1:
                processed[match_idx] = segment_list[0][0]
            elif len(segment_list) > 1:
                segments = [s[0] for s in segment_list]
                indices = [s[1] for s in segment_list]

                is_consecutive = all(
                    indices[i + 1] == indices[i] + 1
                    for i in range(len(indices) - 1)
                )

                if is_consecutive:
                    processed[match_idx] = np.concatenate(segments)
                else:
                    processed[match_idx] = self._handle_backslip(segments, match_idx)

        return processed

    def _handle_backslip(
            self,
            segment_list: List[np.ndarray],
            match_idx: int
    ) -> np.ndarray:
        """Handle multiple non-consecutive alignments to same state."""
        if self.backslip_mode == 'ignore':
            return np.concatenate(segment_list)
        elif self.backslip_mode == 'delete':
            return segment_list[0]
        elif self.backslip_mode == 'average' and DTW_AVAILABLE:
            try:
                series = [seg.astype(np.float64) for seg in segment_list]
                barycenter = dtw_barycenter.dba(series, use_c=True)
                return np.array(barycenter)
            except Exception as e:
                logger.error(f"DTW averaging failed: {e}")
                return self._simple_average(segment_list)
        else:
            return self._simple_average(segment_list)

    def _simple_average(self, segment_list: List[np.ndarray]) -> np.ndarray:
        """Simple averaging fallback when DTW is not available."""
        max_len = max(len(s) for s in segment_list)
        padded = []

        for seg in segment_list:
            if len(seg) < max_len:
                padded_seg = np.pad(
                    seg, (0, max_len - len(seg)),
                    mode='constant',
                    constant_values=seg[-1] if len(seg) > 0 else 0
                )
                padded.append(padded_seg)
            else:
                padded.append(seg)

        return np.mean(padded, axis=0)