"""Result writing utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import pandas as pd

from vrhmm.utils.amino_acids import get_amino_acid_category

logger = logging.getLogger(__name__)


class ResultWriter:
    """Handles writing results to various formats."""

    def __init__(self, output_dir: Path, timestamp: str) -> None:
        self.output_dir = Path(output_dir)
        self.timestamp = timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        results: Dict[str, Dict[str, Any]],
        df: pd.DataFrame,
        args: Any
    ) -> None:
        """Save all results."""
        is_cross_validation = (
            args.test_aa is not None
            and args.model_aa is not None
            and args.test_aa != args.model_aa
        )

        mode_suffix = self._get_mode_suffix(args, is_cross_validation)

        self._save_json_results(results, mode_suffix)
        self._save_csv_summary(df, mode_suffix)

        if is_cross_validation:
            self._save_cross_validation_metrics(df, args, mode_suffix)
        else:
            self._save_metrics(df, args.classification_mode, mode_suffix)

    def _get_mode_suffix(self, args: Any, is_cross_validation: bool) -> str:
        """Generate suffix for output files."""
        if is_cross_validation:
            return f"cross_val_{args.model_aa}_vs_{args.test_aa}_{args.variance_mode}"
        elif args.variance_mode == 'barycenter':
            return f"{args.classification_mode}_{args.variance_mode}_scale{args.variance_scale}"
        else:
            return f"{args.classification_mode}_{args.variance_mode}"

    def _save_csv_summary(self, df: pd.DataFrame, mode_suffix: str) -> None:
        """Save summary as CSV."""
        summary_path = self.output_dir / f"summary_{mode_suffix}_{self.timestamp}.csv"
        df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to: {summary_path}")

    def _save_json_results(
        self,
        results: Dict[str, Dict[str, Any]],
        mode_suffix: str
    ) -> None:
        """Save detailed results as JSON."""
        results_path = self.output_dir / f"results_{mode_suffix}_{self.timestamp}.json"

        json_results = {}
        for key, result in results.items():
            json_results[key] = {
                'amino_acid': result['amino_acid'],
                'num_segments': result['num_segments'],
                'signal_length': result['signal_length'],
                'predicted_category': result.get('predicted_category', ''),
                'log_probability': result.get('log_probability', 0.0),
                'state_sequence': result.get('state_sequence', []),
                'full_path': result.get('full_path', []),
                'all_scores': {
                    aa: float(score)
                    for aa, score in result.get('all_scores', {}).items()
                },
                'best_aa_model': result.get('best_aa_model', ''),
            }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"Saved results to: {results_path}")

    def _compute_accuracy(
        self,
        df: pd.DataFrame,
        classification_mode: str
    ) -> pd.DataFrame:
        """Add true_category column and return the modified dataframe."""
        if classification_mode != '20way':
            df['true_category'] = df['true_aa'].apply(
                lambda x: get_amino_acid_category(x, classification_mode)
            )
        else:
            df['true_category'] = df['true_aa']
        return df

    def _save_metrics(
        self,
        df: pd.DataFrame,
        classification_mode: str,
        mode_suffix: str
    ) -> None:
        """Save full classification metrics."""
        if 'predicted_category' not in df.columns or len(df) == 0:
            return

        df = self._compute_accuracy(df, classification_mode)

        correct = (df['true_category'] == df['predicted_category']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0

        confusion = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            confusion[row['true_category']][row['predicted_category']] += 1

        metrics = {
            'accuracy': accuracy,
            'correct': int(correct),
            'total': int(total),
            'confusion_matrix': {k: dict(v) for k, v in confusion.items()},
            'classification_mode': classification_mode
        }

        metrics_path = self.output_dir / f"metrics_{mode_suffix}_{self.timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to: {metrics_path}")
        logger.info(f"Classification Accuracy: {accuracy:.2%}")

    def _save_cross_validation_metrics(
        self,
        df: pd.DataFrame,
        args: Any,
        mode_suffix: str
    ) -> None:
        """Save minimal metrics for cross-validation."""
        if 'predicted_category' not in df.columns or len(df) == 0:
            return

        df = self._compute_accuracy(df, args.classification_mode)

        correct = (df['true_category'] == df['predicted_category']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0

        metrics = {
            'cross_validation': True,
            'model_aa': args.model_aa,
            'test_aa': args.test_aa,
            'accuracy': accuracy,
            'correct': int(correct),
            'total': int(total),
            'classification_mode': args.classification_mode,
            'variance_mode': args.variance_mode
        }

        metrics_path = self.output_dir / f"cross_val_metrics_{mode_suffix}_{self.timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved cross-validation metrics to: {metrics_path}")
        