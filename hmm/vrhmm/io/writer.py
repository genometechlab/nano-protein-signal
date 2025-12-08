"""Result writing utilities."""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

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
        # Check if this is cross-validation
        is_cross_validation = (
                hasattr(args, 'test_aa') and
                hasattr(args, 'model_aa') and
                args.test_aa is not None and
                args.model_aa is not None and
                args.test_aa != args.model_aa
        )

        mode_suffix = self._get_mode_suffix(args, is_cross_validation)

        self._save_json_results(results, mode_suffix)
        self._save_csv_summary(df, mode_suffix)

        # Only save full metrics if NOT cross-validation
        if not is_cross_validation:
            self._save_metrics(df, args.classification_mode, mode_suffix)
        else:
            # Save minimal cross-validation metrics
            self._save_cross_validation_metrics(df, args, mode_suffix)

    def _get_mode_suffix(self, args: Any, is_cross_validation: bool) -> str:
        """Generate suffix for output files."""
        if is_cross_validation:
            return f"cross_val_{args.model_aa}_vs_{args.test_aa}_{args.variance_mode}"
        elif args.variance_mode == 'barycenter':
            return f"{args.classification_mode}_{args.variance_mode}_scale{args.variance_scale}"
        else:
            return f"{args.classification_mode}_{args.variance_mode}"

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
                'full_path': result.get('full_path', [])
            }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"Saved results to: {results_path}")

    def _save_csv_summary(self, df: pd.DataFrame, mode_suffix: str) -> None:
        """Save summary as CSV."""
        summary_path = self.output_dir / f"summary_{mode_suffix}_{self.timestamp}.csv"
        df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to: {summary_path}")

    def _save_metrics(
            self,
            df: pd.DataFrame,
            classification_mode: str,
            mode_suffix: str
    ) -> None:
        """Save full classification metrics."""
        if 'predicted_category' not in df.columns or len(df) == 0:
            return

        from vrhmm.utils.amino_acids import get_amino_acid_category

        if classification_mode != '20way':
            df['true_category'] = df['true_aa'].apply(
                lambda x: get_amino_acid_category(x, classification_mode)
            )
        else:
            df['true_category'] = df['true_aa']

        correct = (df['true_category'] == df['predicted_category']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0

        from collections import defaultdict
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

        from vrhmm.utils.amino_acids import get_amino_acid_category

        if args.classification_mode != '20way':
            df['true_category'] = df['true_aa'].apply(
                lambda x: get_amino_acid_category(x, args.classification_mode)
            )
        else:
            df['true_category'] = df['true_aa']

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