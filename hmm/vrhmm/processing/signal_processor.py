"""Signal processing pipeline."""

import ast
import json
import logging
from typing import Dict, Any, List, Union, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

class SignalProcessor:
    """Handles signal processing operations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        
        self.config = config

    def process_signal(
            self,
            record: Dict[str, Any],
            segmenter: Any,
            classifier: Any,
            seg_mode: str = 'dynp'
    ) -> Dict[str, Any]:
        """Process a single signal through the pipeline."""
        # Parse signal
        raw_data = record['cleaned_segment']
        signal, is_presegmented = self._parse_signal_data(raw_data)

        # Segment if needed
        if is_presegmented:
            segment_results = self._process_presegmented(raw_data)
        else:
            segment_results = segmenter.segment(signal, seg_mode)

        # Normalize
        z_means = self._z_normalize_means(segment_results['means'])
        segment_results['z_normalized_stats'] = {
            str(i): (float(z_means[i]), 0.0)
            for i in range(len(z_means))
        }

        # Z-normalize full signal
        z_signal = self._z_normalize_signal(signal)

        # Classify
        pred_category, log_prob, all_scores = classifier.predict(z_means)

        # Get state path
        best_model_aa = max(all_scores.keys(), key=lambda k: all_scores[k])
        best_model = classifier.hmm_models[best_model_aa]

        state_sequence = []
        full_path = []

        try:
            _, path = best_model.viterbi(z_means)
            if path:
                for _, state_obj in path:
                    if state_obj.name and 'start' not in state_obj.name.lower() \
                            and 'end' not in state_obj.name.lower():
                        full_path.append(state_obj.name)
                        if 'Match' in state_obj.name:
                            state_sequence.append(state_obj.name)
        except Exception as e:
            logger.warning(f"Could not get Viterbi path: {e}")

        return {
            'signal': signal,
            'signal_length': len(signal),
            'amino_acid': record.get('aa', 'unknown'),
            'segment_results': segment_results,
            'z_normalized_signal': z_signal,
            'log_probability': log_prob,
            'state_sequence': state_sequence,
            'full_path': full_path,
            'num_segments': len(segment_results['means']),
            'predicted_category': pred_category,
            'all_scores': all_scores
        }

    def parse_signal(
            self,
            signal_value: Any
    ) -> npt.NDArray[np.float64]:
        """Parse signal data from various formats."""
        signal, _ = self._parse_signal_data(signal_value)
        return signal

    def _parse_signal_data(
            self,
            signal_value: Any
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """Parse signal and detect if presegmented."""
        is_presegmented = False

        if isinstance(signal_value, str):
            try:
                parsed = ast.literal_eval(signal_value)
            except:
                try:
                    parsed = json.loads(signal_value)
                except:
                    parsed = [float(x.strip()) for x in signal_value.split(',')]
            signal = np.array(parsed, dtype=np.float64)
        elif isinstance(signal_value, list):
            if len(signal_value) > 0 and isinstance(signal_value[0], (list, np.ndarray)):
                is_presegmented = True
                signal_parts = []
                for seg in signal_value:
                    if seg is not None:
                        seg_array = np.array(seg).flatten()
                        if len(seg_array) > 0:
                            signal_parts.append(seg_array)
                signal = np.concatenate(signal_parts) if signal_parts else np.array([])
            else:
                signal = np.array(signal_value, dtype=np.float64)
        elif isinstance(signal_value, np.ndarray):
            signal = signal_value.astype(np.float64)
        else:
            raise ValueError(f"Cannot parse signal data of type {type(signal_value)}")

        return signal, is_presegmented

    def _process_presegmented(
            self,
            raw_segments: List[Any]
    ) -> Dict[str, Any]:
        """Process pre-segmented data."""
        means = []
        variances = []
        breakpoints = [0]
        current_pos = 0

        for segment in raw_segments:
            if isinstance(segment, (list, np.ndarray)):
                segment_array = np.array(segment).flatten()
            else:
                segment_array = np.array([segment])

            means.append(float(np.mean(segment_array)))
            variances.append(float(np.var(segment_array)))

            current_pos += len(segment_array)
            breakpoints.append(current_pos)

        return {
            'means': np.array(means),
            'variances': np.array(variances),
            'breakpoints': breakpoints
        }

    def _z_normalize_means(
            self,
            means: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Z-normalize segment means."""
        mean_val = np.mean(means)
        std_val = np.std(means, ddof=1)

        if std_val == 0:
            return means - mean_val
        return (means - mean_val) / std_val

    def _z_normalize_signal(
            self,
            signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Z-normalize entire signal."""
        mean_val = np.mean(signal)
        std_val = np.std(signal)

        if std_val > 0:
            return (signal - mean_val) / std_val
        return signal - mean_val