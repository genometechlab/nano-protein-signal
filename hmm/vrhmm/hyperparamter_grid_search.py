#!/usr/bin/env python3
"""
hyperparameter_grid_search.py
Grid search for optimal HMM transition probabilities using segment-based or profile-based variance.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from datetime import datetime
import itertools
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import logging
from collections import defaultdict

# Set up for HPC multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from vrhmm.segmentation.segmenter import Segmenter, SegmentVarianceCollector
from vrhmm.core.hmm_builder import HMMConstructor
from vrhmm.core.classifier import HMMClassifier
from vrhmm.io.loader import DataLoader, parse_signal_data, process_pre_segmented_data
from vrhmm.config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_transitions(trans_params: Dict[str, float]) -> Dict[str, float]:
    """Normalize transition probabilities to ensure they sum to 1."""
    normalized = trans_params.copy()

    # Match state outgoing transitions
    match_keys = ['match_self_loop', 'forward', 'to_skip', 'to_slip', 'to_insert', 'to_end']
    match_total = sum(trans_params.get(k, 0) for k in match_keys)

    if match_total > 0:
        for key in match_keys:
            if key in trans_params:
                normalized[key] = trans_params[key] / match_total

    # Insert state transitions
    insert_total = trans_params.get('insert_self_loop', 0.3) + trans_params.get('insert_to_match', 0.7)
    if insert_total > 0:
        normalized['insert_self_loop'] = trans_params.get('insert_self_loop', 0.3) / insert_total
        normalized['insert_to_match'] = trans_params.get('insert_to_match', 0.7) / insert_total

    # Skip state transitions
    skip_total = trans_params.get('skip_to_match', 0.9) + trans_params.get('skip_continue', 0.1)
    if skip_total > 0:
        normalized['skip_to_match'] = trans_params.get('skip_to_match', 0.9) / skip_total
        normalized['skip_continue'] = trans_params.get('skip_continue', 0.1) / skip_total

    # Slip state transitions
    slip_total = trans_params.get('slip_to_match', 0.92) + trans_params.get('slip_continue', 0.08)
    if slip_total > 0:
        normalized['slip_to_match'] = trans_params.get('slip_to_match', 0.92) / slip_total
        normalized['slip_continue'] = trans_params.get('slip_continue', 0.08) / slip_total

    return normalized


def collect_segment_variances(
        signal_data: List[Dict[str, Any]],
        amino_acids: List[str],
        segmenter: Segmenter,
        use_pickle: bool = False,
        max_signals_per_aa: int = 10
) -> Dict[str, SegmentVarianceCollector]:
    """Collect segment variances from signals for each amino acid."""
    logger.info("Collecting segment variances from signals...")

    collectors = {aa: SegmentVarianceCollector() for aa in amino_acids}

    # Group signals by amino acid
    aa_signals = defaultdict(list)
    for record in signal_data:
        aa = record.get('aa', '')
        if aa in amino_acids:
            aa_signals[aa].append(record)

    for aa in amino_acids:
        signals = aa_signals[aa][:max_signals_per_aa]

        if not signals:
            logger.warning(f"No signals found for {aa}")
            continue

        processed = 0
        for record in signals:
            try:
                raw_data = record.get('cleaned_segment')
                if raw_data is None:
                    continue

                # Check if pre-segmented
                if use_pickle or _is_presegmented(raw_data):
                    # Pre-segmented: extract variances directly
                    variances = []
                    for seg in raw_data:
                        if seg is not None:
                            seg_array = np.array(seg).flatten()
                            if len(seg_array) > 0:
                                variances.append(float(np.var(seg_array)))

                    if len(variances) == 35:
                        collectors[aa].add_signal_variances(variances)
                        processed += 1
                else:
                    # Raw signal: segment first
                    signal = parse_signal_data(raw_data)
                    result = segmenter.segment(signal, 'dynp')
                    collectors[aa].add_signal_variances(result['variances'].tolist())
                    processed += 1

            except Exception as e:
                logger.debug(f"Error processing signal for variance: {e}")
                continue

        if processed > 0:
            logger.info(f"  {aa}: Collected variances from {processed} signals")

    return collectors


def _is_presegmented(data) -> bool:
    """Check if data is pre-segmented."""
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, (list, np.ndarray)):
            return True
        if 20 <= len(data) <= 40:
            return True
    return False


def build_classifier_with_transitions(
        barycenters: Optional[Dict[str, List[np.ndarray]]],
        profile_stats: Optional[Dict[str, Dict[int, Tuple[float, float]]]],
        variance_mode: str,
        variance_collectors: Optional[Dict[str, SegmentVarianceCollector]],
        variance_scales: Optional[Dict[str, float]],
        transition_params: Dict[str, float],
        base_config: Dict[str, Any]
) -> HMMClassifier:
    """Build HMM classifier using specified variance mode and transitions."""

    # Update config with new transitions
    config = base_config.copy()
    config['hmm'] = config.get('hmm', {}).copy()
    config['hmm']['transitions'] = transition_params

    classifier = HMMClassifier(classification_mode='20way', use_length_normalization=False)

    # If using profile stats directly
    if profile_stats is not None:
        constructor = HMMConstructor(
            config=config['hmm'],
            variance_mode='profile',
            variance_scale=1.0  # Will be overridden per-AA
        )
        
        for aa, profile_dict in profile_stats.items():
            try:
                # Set variance scale for this AA
                if variance_scales:
                    constructor.variance_scale = variance_scales.get(aa, 1.0)
                else:
                    constructor.variance_scale = 1.0
                
                # Build from profile stats - returns (model, profile_stats) tuple
                result = constructor.build_hmm_from_profile_stats(
                    amino_acid=aa,
                    profile_stats=profile_dict,
                    model_name=f"HMM_{aa}_from_profile"
                )
                
                # Unpack the tuple
                if isinstance(result, tuple):
                    model, _ = result
                else:
                    model = result
                
                classifier.add_model(aa, model)
            except Exception as e:
                logger.warning(f"Error building model for {aa}: {e}")
        
        return classifier
    
    # Otherwise use barycenters
    if variance_mode == 'segment':
        constructor = HMMConstructor(
            config=config['hmm'],
            variance_mode='segment',
            variance_scale=1.0
        )
    else:  # profile mode
        constructor = HMMConstructor(
            config=config['hmm'],
            variance_mode='profile',
            variance_scale=1.0  # Will be overridden per-AA
        )

    for aa, profile_arrays in barycenters.items():
        try:
            if variance_mode == 'segment':
                # Get empirical variances for this amino acid
                segment_variances = None
                if aa in variance_collectors:
                    segment_variances = variance_collectors[aa].get_average_variances()

                model = constructor.build_hmm_from_arrays(
                    amino_acid=aa,
                    profile_arrays=profile_arrays,
                    segment_variances=segment_variances,
                    model_name=f"HMM_{aa}_segment_var",
                    expected_length=35
                )
            else:  # profile mode
                # Set the variance scale for this AA
                constructor.variance_scale = variance_scales.get(aa, 1.0)

                model = constructor.build_hmm_from_arrays(
                    amino_acid=aa,
                    profile_arrays=profile_arrays,
                    segment_variances=None,
                    model_name=f"HMM_{aa}_profile_var",
                    expected_length=35
                )

            classifier.add_model(aa, model)

        except Exception as e:
            logger.warning(f"Error building model for {aa}: {e}")

    return classifier


def evaluate_transitions(
        param_combo: Tuple,
        param_names: List[str],
        base_config: Dict[str, Any],
        test_data: List[Dict[str, Any]],
        barycenters: Optional[Dict[str, List[np.ndarray]]],
        profile_stats: Optional[Dict[str, Dict[int, Tuple[float, float]]]],
        variance_mode: str,
        variance_collectors: Optional[Dict[str, SegmentVarianceCollector]],
        variance_scales: Optional[Dict[str, float]]
) -> Dict[str, Any]:
    """Evaluate a single transition parameter configuration."""

    # Build transition params from combo
    trans_params = base_config['hmm']['transitions'].copy()
    for name, value in zip(param_names, param_combo):
        trans_params[name] = value

    trans_params = normalize_transitions(trans_params)

    # Build classifier
    classifier = build_classifier_with_transitions(
        barycenters,
        profile_stats,
        variance_mode,
        variance_collectors,
        variance_scales,
        trans_params,
        base_config
    )

    # Evaluate
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    per_aa_results = defaultdict(lambda: {'correct': 0, 'total': 0})

    for sample in test_data:
        obs = sample['observation']
        true_aa = sample['true_aa']

        try:
            pred_aa, log_prob, scores = classifier.predict(obs)

            if pred_aa == true_aa:
                correct += 1
                per_aa_results[true_aa]['correct'] += 1

            total += 1
            per_aa_results[true_aa]['total'] += 1
            confusion[true_aa][pred_aa] += 1

        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0

    # Per-AA accuracy
    per_aa_accuracy = {}
    for aa, results in per_aa_results.items():
        if results['total'] > 0:
            per_aa_accuracy[aa] = results['correct'] / results['total']

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'transition_params': trans_params,
        'per_aa_accuracy': per_aa_accuracy,
        'confusion': {k: dict(v) for k, v in confusion.items()},
        'param_combo': param_combo
    }


class TransitionGridSearch:
    """Grid search for optimal HMM transition probabilities using segment or profile variance."""

    def __init__(
            self,
            barycenters: Optional[Dict[str, List[np.ndarray]]],
            profile_stats: Optional[Dict[str, Dict[int, Tuple[float, float]]]],
            signal_data: List[Dict[str, Any]],
            segmenter: Segmenter,
            base_config: Dict[str, Any],
            output_dir: str = "./grid_search_results",
            seg_mode: str = 'dynp',
            use_pickle: bool = False,
            n_folds: int = 5,
            n_processes: int = None,
            max_per_aa: int = None,
            variance_scale_file: Optional[str] = None
    ):
        self.barycenters = barycenters
        self.profile_stats = profile_stats
        self.signal_data = signal_data
        self.segmenter = segmenter
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seg_mode = seg_mode
        self.use_pickle = use_pickle
        self.n_folds = n_folds
        self.n_processes = n_processes or max(1, 48)
        self.max_per_aa = max_per_aa

        # Determine amino acids
        if profile_stats:
            amino_acids = list(profile_stats.keys())
        elif barycenters:
            amino_acids = list(barycenters.keys())
        else:
            raise ValueError("Must provide either barycenters or profile_stats")

        # Determine variance mode and load appropriate data
        if profile_stats:
            # Profile CSV mode - must use profile variance
            self.variance_mode = 'profile'
            self.variance_scales = None
            self.variance_collectors = None
            
            if variance_scale_file:
                logger.info("=" * 60)
                logger.info("LOADING VARIANCE SCALES (Profile Mode)")
                logger.info("=" * 60)
                
                # Handle both JSON and CSV formats
                if variance_scale_file.endswith('.json'):
                    with open(variance_scale_file, 'r') as f:
                        self.variance_scales = json.load(f)
                else:
                    # CSV format: amino_acid,variance_scale
                    import pandas as pd
                    df = pd.read_csv(variance_scale_file)
                    self.variance_scales = dict(zip(df['amino_acid'], df['variance_scale']))
                
                logger.info(f"Loaded variance scales for {len(self.variance_scales)} amino acids")
                for aa in sorted(self.variance_scales.keys()):
                    logger.info(f"  {aa}: {self.variance_scales[aa]:.6f}")
            else:
                logger.info("No variance scales provided - using 1.0 for all amino acids")
        else:
            # Barycenter mode - can use either segment or profile variance
            self.variance_mode = 'profile' if variance_scale_file else 'segment'
            self.variance_scales = None
            self.variance_collectors = None

            if variance_scale_file:
                # Profile variance mode
                logger.info("=" * 60)
                logger.info("LOADING VARIANCE SCALES (Profile Mode)")
                logger.info("=" * 60)
                
                if variance_scale_file.endswith('.json'):
                    with open(variance_scale_file, 'r') as f:
                        self.variance_scales = json.load(f)
                else:
                    import pandas as pd
                    df = pd.read_csv(variance_scale_file)
                    self.variance_scales = dict(zip(df['amino_acid'], df['variance_scale']))
                
                logger.info(f"Loaded variance scales for {len(self.variance_scales)} amino acids")
                for aa in sorted(self.variance_scales.keys()):
                    logger.info(f"  {aa}: {self.variance_scales[aa]:.6f}")
            else:
                # Segment variance mode - collect variances ONCE
                logger.info("=" * 60)
                logger.info("COLLECTING SEGMENT VARIANCES")
                logger.info("=" * 60)
                self.variance_collectors = collect_segment_variances(
                    signal_data=signal_data,
                    amino_acids=amino_acids,
                    segmenter=segmenter,
                    use_pickle=use_pickle,
                    max_signals_per_aa=10
                )

        # Prepare test data
        logger.info("=" * 60)
        logger.info("PREPARING TEST DATA")
        logger.info("=" * 60)
        self.all_test_data = self._prepare_test_data(amino_acids)
        self.folds = self._create_stratified_folds()

        logger.info(f"Initialized with {len(self.all_test_data)} test samples, {n_folds} folds")
        logger.info(f"Using {self.n_processes} parallel processes")
        logger.info(f"Variance mode: {self.variance_mode}")

    def _prepare_test_data(self, amino_acids: List[str]) -> List[Dict[str, Any]]:
        """Prepare and normalize test data."""
        test_data = []
        aa_counts = defaultdict(int)

        for record in self.signal_data:
            aa = record.get('aa', 'unknown')
            if aa not in amino_acids:
                continue

            # Apply max_per_aa limit if specified
            if self.max_per_aa is not None and aa_counts[aa] >= self.max_per_aa:
                continue

            try:
                raw_data = record.get('cleaned_segment')
                if raw_data is None:
                    continue

                # Handle pre-segmented vs raw
                if self.use_pickle or _is_presegmented(raw_data):
                    segment_results = process_pre_segmented_data(raw_data)
                else:
                    signal = parse_signal_data(raw_data)
                    segment_results = self.segmenter.segment(signal, self.seg_mode)

                if len(segment_results['means']) == 0:
                    continue

                # Z-normalize the means
                means = np.array(segment_results['means'])
                std = np.std(means, ddof=1)
                if std > 0:
                    z_normalized = (means - np.mean(means)) / std
                else:
                    z_normalized = means - np.mean(means)

                test_data.append({
                    'observation': z_normalized,
                    'true_aa': aa,
                    'means': segment_results['means'],
                    'variances': segment_results['variances']
                })

                aa_counts[aa] += 1

            except Exception as e:
                logger.debug(f"Error processing record: {e}")
                continue

        # Print distribution
        logger.info("Test data distribution:")
        for aa in sorted(aa_counts.keys()):
            logger.info(f"  {aa}: {aa_counts[aa]} signals")
        logger.info(f"  Total: {len(test_data)} signals")

        return test_data

    def _create_stratified_folds(self) -> List[List[int]]:
        """Create stratified folds for cross-validation."""
        aa_indices = defaultdict(list)
        for i, sample in enumerate(self.all_test_data):
            aa_indices[sample['true_aa']].append(i)

        # Shuffle within each group
        rng = np.random.default_rng(42)
        for indices in aa_indices.values():
            rng.shuffle(indices)

        # Distribute to folds
        folds = [[] for _ in range(self.n_folds)]
        for aa, indices in aa_indices.items():
            for i, idx in enumerate(indices):
                folds[i % self.n_folds].append(idx)

        return folds

    def _evaluate_with_cv(
            self,
            param_combo: Tuple,
            param_names: List[str]
    ) -> Dict[str, Any]:
        """Evaluate configuration with cross-validation."""

        # Build transition params
        trans_params = self.base_config['hmm']['transitions'].copy()
        for name, value in zip(param_names, param_combo):
            trans_params[name] = value
        trans_params = normalize_transitions(trans_params)

        fold_accuracies = []
        all_confusion = defaultdict(lambda: defaultdict(int))

        for fold_idx in range(self.n_folds):
            # Get test split for this fold
            test_indices = set(self.folds[fold_idx])
            test_data = [self.all_test_data[i] for i in test_indices]

            # Build classifier
            classifier = build_classifier_with_transitions(
                self.barycenters,
                self.profile_stats,
                self.variance_mode,
                self.variance_collectors,
                self.variance_scales,
                trans_params,
                self.base_config
            )

            correct = 0
            total = 0

            for sample in test_data:
                try:
                    pred_aa, _, _ = classifier.predict(sample['observation'])
                    if pred_aa == sample['true_aa']:
                        correct += 1
                    total += 1
                    all_confusion[sample['true_aa']][pred_aa] += 1
                except:
                    continue

            if total > 0:
                fold_accuracies.append(correct / total)

        return {
            'cv_mean_accuracy': np.mean(fold_accuracies) if fold_accuracies else 0,
            'cv_std_accuracy': np.std(fold_accuracies) if fold_accuracies else 0,
            'fold_accuracies': fold_accuracies,
            'transition_params': trans_params,
            'param_combo': param_combo,
            'confusion': {k: dict(v) for k, v in all_confusion.items()}
        }

    def run_grid_search(
            self,
            transition_grids: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Run grid search over transition probabilities."""

        param_names = list(transition_grids.keys())
        param_values = list(transition_grids.values())
        all_combos = list(itertools.product(*param_values))

        logger.info("=" * 60)
        logger.info("TRANSITION PROBABILITY GRID SEARCH")
        logger.info("=" * 60)
        logger.info(f"Parameters being searched:")
        for name, values in transition_grids.items():
            logger.info(f"  {name}: {values}")
        logger.info(f"Total combinations: {len(all_combos)}")
        logger.info(f"Using {self.n_folds}-fold cross-validation")
        logger.info(f"Variance mode: {self.variance_mode}")

        # Create partial function for parallel evaluation
        eval_func = partial(
            self._evaluate_with_cv,
            param_names=param_names
        )

        # Run parallel evaluation
        logger.info("\nRunning parallel evaluation...")
        with mp.Pool(self.n_processes) as pool:
            results = list(tqdm(
                pool.imap(eval_func, all_combos, chunksize=max(1, len(all_combos) // 100)),
                total=len(all_combos),
                desc="Evaluating"
            ))

        # Find best result
        best_result = max(results, key=lambda x: x['cv_mean_accuracy'])

        # Save results
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df.to_csv(self.output_dir / f'transition_search_{timestamp}.csv', index=False)

        # Save best config
        best_config = {
            'cv_mean_accuracy': best_result['cv_mean_accuracy'],
            'cv_std_accuracy': best_result['cv_std_accuracy'],
            'transition_params': best_result['transition_params'],
            'variance_mode': self.variance_mode,
            'n_folds': self.n_folds,
            'n_test_samples': len(self.all_test_data),
            'timestamp': timestamp
        }

        with open(self.output_dir / f'best_config_{timestamp}.json', 'w') as f:
            json.dump(best_config, f, indent=2)

        # Also save as 'latest'
        with open(self.output_dir / 'best_config_latest.json', 'w') as f:
            json.dump(best_config, f, indent=2)

        self._print_summary(best_result, transition_grids)

        return df

    def _print_summary(
            self,
            best_result: Dict[str, Any],
            transition_grids: Dict[str, List[float]]
    ) -> None:
        """Print search summary."""
        print("\n" + "=" * 70)
        print("GRID SEARCH COMPLETE")
        print("=" * 70)

        print(f"\nVariance Mode: {self.variance_mode}")
        print(f"Cross-validation: {self.n_folds} folds")
        print(f"Test samples: {len(self.all_test_data)}")

        print(f"\nBest CV Accuracy: {best_result['cv_mean_accuracy']:.4f} ± {best_result['cv_std_accuracy']:.4f}")
        print(f"Fold accuracies: {[f'{a:.4f}' for a in best_result['fold_accuracies']]}")

        print("\nOptimal Transition Parameters:")
        print("-" * 40)
        for key, value in sorted(best_result['transition_params'].items()):
            # Mark if this was a searched parameter
            searched = "  <-- searched" if key in transition_grids else ""
            print(f"  {key}: {value:.6f}{searched}")

        print(f"\nResults saved to: {self.output_dir}")

        # Print config update suggestion
        print("\n" + "=" * 70)
        print("TO USE THESE PARAMETERS, UPDATE default_config.py:")
        print("=" * 70)
        print("\nHMM_TRANSITIONS: Dict[str, float] = {")
        for key, value in sorted(best_result['transition_params'].items()):
            print(f"    '{key}': {value:.6f},")
        print("}")


def create_transition_search_config() -> Dict[str, List[float]]:
    """Create default comprehensive search grid."""
    return {
        # Match state transitions (most important)
        'match_self_loop': [0.01, 0.02, 0.05, 0.10],
        'forward': [0.55, 0.60, 0.65, 0.70, 0.75],
        'to_skip': [0.10, 0.15, 0.20, 0.25],
        'to_slip': [0.03, 0.05, 0.07, 0.10],
        'to_insert': [0.02, 0.03, 0.05],
        'to_end': [0.02, 0.03, 0.04, 0.05],
    }


def create_quick_search_config() -> Dict[str, List[float]]:
    """Create reduced search grid for quick testing."""
    return {
        'match_self_loop': [0.01, 0.05],
        'forward': [0.60, 0.70],
        'to_skip': [0.15, 0.20],
        'to_slip': [0.05, 0.10],
    }


def create_fine_search_config(base_params: Dict[str, float]) -> Dict[str, List[float]]:
    """Create fine-tuning grid around existing parameters."""
    grid = {}

    for key, value in base_params.items():
        if key in ['match_self_loop', 'forward', 'to_skip', 'to_slip', 'to_insert', 'to_end']:
            # Search ±20% around current value
            delta = value * 0.2
            grid[key] = [
                max(0.001, value - delta),
                value,
                min(0.99, value + delta)
            ]

    return grid


def load_profile_csv(profile_file: str) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Load pre-computed profile CSV with columns: amino_acid, state, mean, std"""
    import pandas as pd
    
    df = pd.read_csv(profile_file)
    
    # Handle different possible column names
    if 'amino_acid' in df.columns:
        aa_col = 'amino_acid'
    elif 'aa' in df.columns:
        aa_col = 'aa'
    else:
        raise ValueError(f"Profile CSV must have 'amino_acid' or 'aa' column. Found: {df.columns.tolist()}")
    
    profiles = {}
    for aa in df[aa_col].unique():
        aa_data = df[df[aa_col] == aa].sort_values('state')
        profiles[aa] = {
            int(row['state']): (float(row['mean']), float(row['std']))
            for _, row in aa_data.iterrows()
        }
    
    return profiles


def main():
    parser = argparse.ArgumentParser(
        description='HMM Transition Probability Grid Search (Segment or Profile Variance Mode)'
    )
    
    # Profile input (mutually exclusive with barycenter-file)
    profile_group = parser.add_mutually_exclusive_group(required=True)
    profile_group.add_argument('--barycenter-file', type=str,
                               help='Path to barycenter JSON/pickle file')
    profile_group.add_argument('--profile-file', type=str,
                               help='Path to pre-computed profile CSV (amino_acid, state, mean, std)')
    
    parser.add_argument('--signal-file', type=str, required=True,
                        help='Path to signal data file (CSV or pickle)')
    parser.add_argument('--output-dir', type=str, default='./grid_search_results',
                        help='Output directory for results')
    parser.add_argument('--seg-mode', type=str, default='dynp',
                        choices=['dynp', 'set_window', 'pelt'],
                        help='Segmentation mode')
    parser.add_argument('--use-pickle', action='store_true',
                        help='Force treating data as pre-segmented')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n-processes', type=int, default=None,
                        help='Number of parallel processes')
    parser.add_argument('--max-per-aa', type=int, default=None,
                        help='Maximum signals per amino acid')
    parser.add_argument('--variance-scale-file', type=str, default=None,
                        help='Path to variance scale JSON/CSV file (for profile variance mode)')
    parser.add_argument('--metadata-file', type=str, default=None,
                        help='Path to metadata JSON file for filtering signals (format: {"traces": [...]})')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with reduced grid')
    parser.add_argument('--fine-tune', type=str, default=None,
                        help='Path to existing best_config.json for fine-tuning')

    args = parser.parse_args()

    # Load profiles
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    barycenters = None
    profile_stats = None
    
    if args.profile_file:
        logger.info("Loading pre-computed profiles from CSV...")
        profile_stats = load_profile_csv(args.profile_file)
        logger.info(f"Loaded {len(profile_stats)} amino acid profiles from CSV: {sorted(profile_stats.keys())}")
        # Profile CSV mode requires variance scales
        if not args.variance_scale_file:
            logger.warning("Using profile CSV without variance scales - will use variance_scale=1.0 for all")
    else:
        logger.info("Loading barycenters...")
        loader = DataLoader(args.barycenter_file, 'json' if args.barycenter_file.endswith('.json') else 'pickle')
        data = loader.load_data()
        barycenters = {k: v for k, v in data.items() if k in valid_aas}
        logger.info(f"Loaded {len(barycenters)} amino acid barycenters: {sorted(barycenters.keys())}")

    # Load signals
    logger.info("Loading signals...")
    signal_path = Path(args.signal_file)
    data_type = 'pickle' if signal_path.suffix in ['.pkl', '.pickle'] else 'csv'
    use_pickle = args.use_pickle or data_type == 'pickle'

    # Load metadata if provided
    metadata = None
    if args.metadata_file:
        logger.info(f"Loading metadata from {args.metadata_file}...")
        with open(args.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if 'traces' in metadata:
            logger.info(f"Metadata specifies {len(metadata['traces'])} traces to analyze")
        else:
            logger.warning("Metadata file missing 'traces' key - expected format: {'traces': [...]}")

    loader = DataLoader(str(signal_path), data_type, signal_dict=True, metadata=metadata)
    signal_data = loader.load_data()
    logger.info(f"Loaded {len(signal_data)} signals")

    # Initialize
    segmenter = Segmenter(CONFIG)

    search = TransitionGridSearch(
        barycenters=barycenters,
        profile_stats=profile_stats,
        signal_data=signal_data,
        segmenter=segmenter,
        base_config=CONFIG,
        output_dir=args.output_dir,
        seg_mode=args.seg_mode,
        use_pickle=use_pickle,
        n_folds=args.n_folds,
        n_processes=args.n_processes,
        max_per_aa=args.max_per_aa,
        variance_scale_file=args.variance_scale_file
    )

    # Select search configuration
    if args.fine_tune:
        logger.info(f"Fine-tuning from: {args.fine_tune}")
        with open(args.fine_tune) as f:
            existing = json.load(f)
        transition_grids = create_fine_search_config(existing['transition_params'])
    elif args.quick:
        logger.info("Using quick search configuration")
        transition_grids = create_quick_search_config()
    else:
        logger.info("Using full search configuration")
        transition_grids = create_transition_search_config()

    # Run search
    results_df = search.run_grid_search(transition_grids)

    # Print top 5 results
    print("\nTop 5 Configurations:")
    print("-" * 50)
    top5 = results_df.nlargest(5, 'cv_mean_accuracy')
    for i, row in top5.iterrows():
        print(f"  {row['cv_mean_accuracy']:.4f} ± {row['cv_std_accuracy']:.4f}")


if __name__ == "__main__":
    main()