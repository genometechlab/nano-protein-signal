"""Signal processing pipeline."""

import ast
import json
import logging
from typing import Dict, Any, List

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class SignalProcessor:

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def process_signal(
        self,
        record: Dict[str, Any],
        segmenter: Any,
        classifier: Any,
        seg_mode: str = 'dynp'
    ) -> Dict[str, Any]:
        """Run the full processing pipeline on a single signal record."""
        raw_data = record['cleaned_segment']
        signal, is_presegmented = self._parse_signal_data(raw_data)

        if is_presegmented:
            segment_results = self._process_presegmented(raw_data)
        else:
            segment_results = segmenter.segment(signal, seg_mode)

        z_means = self._z_normalize_means(segment_results['means'])
        segment_results['z_normalized_stats'] = {
            str(i): (float(z_means[i]), 0.0)
            for i in range(len(z_means))
        }

        z_signal = self._z_normalize_signal(signal)

        pred_category, log_prob, all_scores = classifier.predict(z_means)

        # In 20way mode, pred_category is already the best AA.
        # For grouped modes, find the best individual AA from model scores.
        if classifier.classification_mode == '20way':
            best_model_aa = pred_category
        else:
            best_model_aa = max(
                classifier.hmm_models,
                key=lambda aa: classifier.hmm_models[aa].log_probability(z_means)
            )

        state_sequence, full_path = self._extract_viterbi_path(
            classifier.hmm_models[best_model_aa], z_means
        )

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
            'all_scores': all_scores,
            'best_aa_model': best_model_aa
        }

    def _extract_viterbi_path(
        self,
        model: Any,
        z_means: npt.NDArray[np.float64]
    ) -> tuple[List[str], List[str]]:
        """Run Viterbi on the best model and extract state/match sequences."""
        state_sequence = []
        full_path = []

        try:
            _, path = model.viterbi(z_means)
            if path:
                for _, state_obj in path:
                    name = state_obj.name
                    if not name or 'start' in name.lower() or 'end' in name.lower():
                        continue
                    full_path.append(name)
                    if 'Match' in name:
                        state_sequence.append(name)
        except Exception as e:
            logger.warning(f"Could not get Viterbi path: {e}")

        return state_sequence, full_path

    def parse_signal(self, signal_value: Any) -> npt.NDArray[np.float64]:
        signal, _ = self._parse_signal_data(signal_value)
        return signal

    def _parse_signal_data(
        self,
        signal_value: Any
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """Parse raw signal data into a flat numpy array.

        Returns (signal_array, is_presegmented).
        """
        if isinstance(signal_value, np.ndarray):
            return signal_value.astype(np.float64), False

        if isinstance(signal_value, str):
            parsed = self._parse_string_signal(signal_value)
            return np.array(parsed, dtype=np.float64), False

        if isinstance(signal_value, list):
            if len(signal_value) > 0 and isinstance(signal_value[0], (list, np.ndarray)):
                signal_parts = [
                    np.array(seg).flatten()
                    for seg in signal_value
                    if seg is not None and len(np.array(seg).flatten()) > 0
                ]
                signal = np.concatenate(signal_parts) if signal_parts else np.array([])
                return signal, True
            else:
                return np.array(signal_value, dtype=np.float64), False

        raise ValueError(f"Cannot parse signal data of type {type(signal_value)}")

    @staticmethod
    def _parse_string_signal(value: str) -> list:
        """Parse a string-encoded signal (Python literal, JSON, or CSV)."""
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        return [float(x.strip()) for x in value.split(',')]

    def _process_presegmented(self, raw_segments: List[Any]) -> Dict[str, Any]:
        """Convert pre-segmented data to the format expected downstream."""
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

    @staticmethod
    def _z_normalize_means(means: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        mean_val = np.mean(means)
        std_val = np.std(means, ddof=1)
        if std_val < 1e-10:
            return means - mean_val
        return (means - mean_val) / std_val

    @staticmethod
    def _z_normalize_signal(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val < 1e-10:
            return signal - mean_val
        return (signal - mean_val) / std_val