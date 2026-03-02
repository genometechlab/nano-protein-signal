#!/usr/bin/env python
"""
Joint Bayesian optimization for HMM parameters with multi-objective training.

Optimizes:
  - 20 variance scales (one per amino acid)
  - 6 transition probabilities (global or per-AA)
  - 3 objective weights (alpha, beta, gamma)

Training objective combines:
  - Log-likelihood (model fit)
  - Match state coverage (observed-to-expected ratio)
  - Path smoothness (penalize skips/slips/self-loops)

Usage:
    python batch_joint_optimization.py \
        --profile-file data/amino_acid_profiles.csv \
        --signal-file data/signals.pkl \
        --metadata-file data/test_metadata.json \
        --n-calls 5056 \
        --batch-size 32 \
        --output-dir ./optimization_results
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed

# Handle imports whether run as script or as module
try:
    from vrhmm.optimization.objective import (
        MultiObjectiveTrainer,
        TrainingConfig,
        analyze_viterbi_path
    )
    from vrhmm.optimization.model_builder import OptimizationModelBuilder
    from vrhmm.optimization.data_loader import OptimizationDataLoader
except ImportError:
    from objective import (
        MultiObjectiveTrainer,
        TrainingConfig,
        analyze_viterbi_path
    )
    from model_builder import OptimizationModelBuilder
    from data_loader import OptimizationDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
N_AA = len(AMINO_ACIDS)

# Global transition parameters with their typical ranges
TRANSITION_PARAMS = [
    'match_self_loop',
    'forward',
    'to_skip',
    'to_slip',
    'to_insert',
    'to_end'
]

# Default bounds for each transition (min, max) - before normalization
TRANSITION_BOUNDS = {
    'match_self_loop': (0.001, 0.1),    # Default: 0.012
    'forward':         (0.5, 0.9),       # Default: 0.679
    'to_skip':         (0.01, 0.3),      # Default: 0.123
    'to_slip':         (0.01, 0.2),      # Default: 0.086
    'to_insert':       (0.01, 0.2),      # Default: 0.062
    'to_end':          (0.001, 0.15)     # Default: 0.037
}

N_TRANS = len(TRANSITION_PARAMS)

# Objective weight parameters
OBJECTIVE_WEIGHTS = ['alpha', 'beta', 'gamma']
N_WEIGHTS = len(OBJECTIVE_WEIGHTS)


class JointBayesianOptimizer:
    """
    Bayesian optimizer for HMM parameters with multi-objective training.
    """
    
    def __init__(
        self,
        profile_file: str,
        signal_file: str,
        metadata_file: Optional[str] = None,
        per_aa_transitions: bool = False,
        optimize_weights: bool = True,
        optimize_variance: bool = True,
        optimize_transitions: bool = True,
        n_jobs: int = -1
    ):
        self.profile_file = profile_file
        self.signal_file = signal_file
        self.metadata_file = metadata_file
        self.per_aa_transitions = per_aa_transitions
        self.optimize_weights = optimize_weights
        self.optimize_variance = optimize_variance
        self.optimize_transitions = optimize_transitions
        self.n_jobs = n_jobs
        
        # Calculate dimensions
        self.n_variance = N_AA
        self.n_transitions = N_TRANS * N_AA if per_aa_transitions else N_TRANS
        self.n_weights = N_WEIGHTS if optimize_weights else 0
        self.n_dims = self.n_variance + self.n_transitions + self.n_weights
        
        # Load data
        logger.info("Loading profile and signal data...")
        self.data_loader = OptimizationDataLoader(
            profile_file=profile_file,
            signal_file=signal_file,
            metadata_file=metadata_file
        )
        self.profile_stats = self.data_loader.load_profiles()
        self.test_traces = self.data_loader.load_traces()
        
        # Model builder
        self.model_builder = OptimizationModelBuilder(self.profile_stats)
        
        # Trainer
        self.trainer = MultiObjectiveTrainer(
            model_builder=self.model_builder,
            profile_stats=self.profile_stats
        )
        
        logger.info(f"Loaded profiles for {len(self.profile_stats)} amino acids")
        logger.info(f"Loaded test traces: {sum(len(t) for t in self.test_traces.values())} total")
        logger.info(f"Optimization dimensions: {self.n_dims}")
        logger.info(f"  - Variance scales: {self.n_variance}")
        logger.info(f"  - Transitions: {self.n_transitions} ({'per-AA' if per_aa_transitions else 'global'})")
        logger.info(f"  - Objective weights: {self.n_weights}")
    
    def get_bounds(
        self,
        scale_min: float = 10.0,
        scale_max: float = 200.0,
        transition_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        alpha_range: Tuple[float, float] = (0.5, 2.0),
        beta_range: Tuple[float, float] = (0.1, 2.0),
        gamma_range: Tuple[float, float] = (0.1, 2.0)
    ) -> np.ndarray:
        """Build parameter bounds in log space."""
        # Use default transition bounds if not provided
        trans_bounds = TRANSITION_BOUNDS.copy()
        if transition_bounds:
            trans_bounds.update(transition_bounds)
        
        bounds = np.zeros((self.n_dims, 2))
        idx = 0
        
        # Variance scales (log space)
        bounds[idx:idx + self.n_variance, 0] = np.log(scale_min)
        bounds[idx:idx + self.n_variance, 1] = np.log(scale_max)
        idx += self.n_variance
        
        # Transitions (log space) - each has its own bounds
        if self.per_aa_transitions:
            for aa_idx in range(N_AA):
                for t_idx, param in enumerate(TRANSITION_PARAMS):
                    t_min, t_max = trans_bounds[param]
                    bounds[idx + aa_idx * N_TRANS + t_idx, 0] = np.log(t_min)
                    bounds[idx + aa_idx * N_TRANS + t_idx, 1] = np.log(t_max)
        else:
            for t_idx, param in enumerate(TRANSITION_PARAMS):
                t_min, t_max = trans_bounds[param]
                bounds[idx + t_idx, 0] = np.log(t_min)
                bounds[idx + t_idx, 1] = np.log(t_max)
        idx += self.n_transitions
        
        # Objective weights (log space)
        if self.optimize_weights:
            bounds[idx, 0] = np.log(alpha_range[0])
            bounds[idx, 1] = np.log(alpha_range[1])
            bounds[idx + 1, 0] = np.log(beta_range[0])
            bounds[idx + 1, 1] = np.log(beta_range[1])
            bounds[idx + 2, 0] = np.log(gamma_range[0])
            bounds[idx + 2, 1] = np.log(gamma_range[1])
        
        self._transition_bounds = trans_bounds
        return bounds
    
    def vector_to_params(
        self,
        x: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], TrainingConfig]:
        """Convert optimization vector to parameter dictionaries."""
        idx = 0
        
        # Variance scales
        variance_scales = {
            aa: float(np.exp(x[idx + i]))
            for i, aa in enumerate(AMINO_ACIDS)
        }
        idx += self.n_variance
        
        # Transitions
        if self.per_aa_transitions:
            transitions = {}
            for i, aa in enumerate(AMINO_ACIDS):
                aa_trans = {}
                for j, param in enumerate(TRANSITION_PARAMS):
                    aa_trans[param] = float(np.exp(x[idx + i * N_TRANS + j]))
                transitions[aa] = self._normalize_transitions(aa_trans)
            idx += self.n_transitions
        else:
            raw_trans = {
                param: float(np.exp(x[idx + i]))
                for i, param in enumerate(TRANSITION_PARAMS)
            }
            global_trans = self._normalize_transitions(raw_trans)
            transitions = {aa: global_trans.copy() for aa in AMINO_ACIDS}
            idx += self.n_transitions
        
        # Objective weights
        if self.optimize_weights:
            config = TrainingConfig(
                alpha=float(np.exp(x[idx])),
                beta=float(np.exp(x[idx + 1])),
                gamma=float(np.exp(x[idx + 2]))
            )
        else:
            config = TrainingConfig()
        
        return variance_scales, transitions, config
    
    def params_to_vector(
        self,
        variance_scales: Dict[str, float],
        transitions: Dict[str, Dict[str, float]],
        config: Optional[TrainingConfig] = None
    ) -> np.ndarray:
        """Convert parameters back to optimization vector."""
        x = np.zeros(self.n_dims)
        idx = 0
        
        for i, aa in enumerate(AMINO_ACIDS):
            x[idx + i] = np.log(variance_scales.get(aa, 80.0))
        idx += self.n_variance
        
        if self.per_aa_transitions:
            for i, aa in enumerate(AMINO_ACIDS):
                aa_trans = transitions.get(aa, {})
                for j, param in enumerate(TRANSITION_PARAMS):
                    x[idx + i * N_TRANS + j] = np.log(aa_trans.get(param, 0.1))
        else:
            first_aa = AMINO_ACIDS[0]
            global_trans = transitions.get(first_aa, {})
            for j, param in enumerate(TRANSITION_PARAMS):
                x[idx + j] = np.log(global_trans.get(param, 0.1))
        idx += self.n_transitions
        
        if self.optimize_weights and config:
            x[idx] = np.log(config.alpha)
            x[idx + 1] = np.log(config.beta)
            x[idx + 2] = np.log(config.gamma)
        
        return x
    
    def _normalize_transitions(self, trans: Dict[str, float]) -> Dict[str, float]:
        """Normalize transition probabilities to sum to 1."""
        normalized = trans.copy()
        
        match_keys = ['match_self_loop', 'forward', 'to_skip', 'to_slip', 'to_insert', 'to_end']
        match_total = sum(trans.get(k, 0) for k in match_keys)
        
        if match_total > 0:
            for key in match_keys:
                if key in trans:
                    normalized[key] = trans[key] / match_total
        
        normalized.setdefault('insert_self_loop', 0.3)
        normalized.setdefault('insert_to_match', 0.7)
        normalized.setdefault('skip_to_match', 0.9)
        normalized.setdefault('skip_continue', 0.1)
        normalized.setdefault('slip_to_match', 0.92)
        normalized.setdefault('slip_continue', 0.08)
        
        return normalized
    
    def evaluate_single(self, x: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a single parameter configuration."""
        variance_scales, transitions, config = self.vector_to_params(x)
        
        score, metrics = self.trainer.compute_objective(
            variance_scales=variance_scales,
            transitions=transitions,
            test_traces=self.test_traces,
            config=config
        )
        
        return score, metrics
    
    def evaluate_batch(self, X_batch: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Evaluate a batch of configurations in parallel."""
        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self.evaluate_single)(x) for x in X_batch
        )
        
        scores = np.array([r[0] for r in results])
        metrics = [r[1] for r in results]
        
        return scores, metrics
    
    def propose_batch(
        self,
        gp: GaussianProcessRegressor,
        y_best: float,
        bounds: np.ndarray,
        batch_size: int,
        n_restarts: int = 5,
        xi: float = 0.01
    ) -> np.ndarray:
        """Propose a batch of diverse points using Expected Improvement."""
        candidates = []
        X_pending = []
        
        for _ in range(batch_size):
            best_ei = -np.inf
            best_x = None
            
            for _ in range(n_restarts):
                x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
                
                def neg_ei(x):
                    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
                    sigma = max(sigma[0], 1e-9)
                    
                    imp = mu[0] - y_best - xi
                    Z = imp / sigma
                    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                    
                    for xp in X_pending:
                        dist = np.linalg.norm(x - xp)
                        ei *= (1 - np.exp(-dist / 0.5))
                    
                    return -ei
                
                try:
                    result = minimize(
                        neg_ei, x0,
                        bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                        method='L-BFGS-B'
                    )
                    
                    if -result.fun > best_ei:
                        best_ei = -result.fun
                        best_x = result.x
                except Exception:
                    pass
            
            if best_x is not None:
                candidates.append(best_x)
                X_pending.append(best_x)
            else:
                rand_x = np.random.uniform(bounds[:, 0], bounds[:, 1])
                candidates.append(rand_x)
                X_pending.append(rand_x)
        
        return np.array(candidates)
    
    def optimize(
        self,
        n_calls: int = 5056,
        batch_size: int = 32,
        n_initial: int = 256,
        scale_min: float = 10.0,
        scale_max: float = 200.0,
        transition_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        warm_start_variance: Optional[Dict[str, float]] = None,
        warm_start_transitions: Optional[Dict[str, float]] = None,
        warm_start_weights: Optional[Dict[str, float]] = None,
        output_dir: str = './optimization_results',
        checkpoint_frequency: int = 1
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], TrainingConfig, float, pd.DataFrame]:
        """Run Bayesian optimization."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        bounds = self.get_bounds(
            scale_min=scale_min,
            scale_max=scale_max,
            transition_bounds=transition_bounds
        )
        
        # Initialize defaults
        default_variance = {aa: 80.0 for aa in AMINO_ACIDS}
        default_trans_single = {
            'match_self_loop': 0.012,
            'forward': 0.679,
            'to_skip': 0.123,
            'to_slip': 0.086,
            'to_insert': 0.062,
            'to_end': 0.037
        }
        default_trans = {aa: default_trans_single.copy() for aa in AMINO_ACIDS}
        default_config = TrainingConfig()
        
        if warm_start_variance:
            default_variance.update(warm_start_variance)
        if warm_start_transitions:
            for aa in AMINO_ACIDS:
                default_trans[aa].update(warm_start_transitions)
        if warm_start_weights:
            default_config = TrainingConfig(
                alpha=warm_start_weights.get('alpha', 1.0),
                beta=warm_start_weights.get('beta', 0.5),
                gamma=warm_start_weights.get('gamma', 0.3)
            )
        
        X_observed = []
        y_observed = []
        all_results = []
        
        best_score = -np.inf
        best_variance = default_variance.copy()
        best_transitions = {aa: t.copy() for aa, t in default_trans.items()}
        best_config = default_config
        
        self._log_header(n_calls, batch_size, bounds)
        
        start_time = time.time()
        total_evaluated = 0
        
        # Evaluate warm start point
        warm_x = self.params_to_vector(default_variance, default_trans, default_config)
        logger.info("Evaluating warm start configuration...")
        warm_score, warm_metrics = self.evaluate_single(warm_x)
        
        X_observed.append(warm_x)
        y_observed.append(warm_score)
        all_results.append(self._build_result_dict(warm_x, warm_score, warm_metrics, 0))
        
        if warm_score > best_score:
            best_score = warm_score
            best_variance, best_transitions, best_config = self.vector_to_params(warm_x)
        
        logger.info(f"Warm start score: {warm_score:.4f}")
        self._log_metrics(warm_metrics)
        
        try:
            accuracy, _ = self.trainer.compute_classification_accuracy(
                best_variance, best_transitions, self.test_traces, coverage_weight=0.0
            )
            logger.info(f"Initial classification accuracy: {accuracy*100:.1f}%")
        except Exception as e:
            logger.debug(f"Could not compute initial accuracy: {e}")
        
        self._save_checkpoint(
            best_variance, best_transitions, best_config,
            best_score, all_results, output_path, 1
        )
        
        total_evaluated = 1
        n_batches = (n_calls - 1 + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            batch_start = total_evaluated
            batch_end = min(batch_start + batch_size, n_calls)
            current_batch_size = batch_end - batch_start
            
            if current_batch_size <= 0:
                break
            
            batch_time_start = time.time()
            logger.info(f"\nBatch {batch_idx + 1}/{n_batches}: evaluating {current_batch_size} points...")
            
            if len(X_observed) < n_initial:
                n_random = min(current_batch_size, n_initial - len(X_observed))
                X_batch = np.random.uniform(
                    bounds[:, 0], bounds[:, 1],
                    size=(n_random, self.n_dims)
                )
                
                if n_random < current_batch_size and len(X_observed) >= 20:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                    gp = GaussianProcessRegressor(
                        kernel=kernel, alpha=1e-6,
                        normalize_y=True, n_restarts_optimizer=3
                    )
                    gp.fit(np.array(X_observed), np.array(y_observed))
                    
                    X_gp = self.propose_batch(
                        gp, np.max(y_observed), bounds,
                        current_batch_size - n_random
                    )
                    X_batch = np.vstack([X_batch, X_gp])
            else:
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel, alpha=1e-6,
                    normalize_y=True, n_restarts_optimizer=3
                )
                
                logger.info(f"  Fitting GP on {len(X_observed)} observations...")
                gp.fit(np.array(X_observed), np.array(y_observed))
                
                logger.info(f"  Proposing {current_batch_size} candidates...")
                X_batch = self.propose_batch(gp, np.max(y_observed), bounds, current_batch_size)
            
            logger.info(f"  Running {len(X_batch)} parallel evaluations...")
            y_batch, metrics_batch = self.evaluate_batch(X_batch)
            
            batch_time = time.time() - batch_time_start
            
            for i in range(len(X_batch)):
                X_observed.append(X_batch[i])
                y_observed.append(y_batch[i])
                all_results.append(self._build_result_dict(
                    X_batch[i], y_batch[i], metrics_batch[i], total_evaluated + i + 1
                ))
                
                if y_batch[i] > best_score:
                    best_score = y_batch[i]
                    best_variance, best_transitions, best_config = self.vector_to_params(X_batch[i])
                    logger.info(f"  *** New best: {best_score:.4f} ***")
                    self._log_metrics(metrics_batch[i])
                    global_trans = best_transitions[AMINO_ACIDS[0]]
                    logger.info(f"    Forward: {global_trans['forward']:.4f}, "
                               f"Skip: {global_trans['to_skip']:.4f}, "
                               f"Slip: {global_trans['to_slip']:.4f}")
                    logger.info(f"    Weights: α={best_config.alpha:.3f}, "
                               f"β={best_config.beta:.3f}, γ={best_config.gamma:.3f}")
            
            total_evaluated += len(X_batch)
            
            batch_best = np.max(y_batch)
            batch_mean = np.mean(y_batch)
            logger.info(f"  Batch: best={batch_best:.4f}, mean={batch_mean:.4f}, time={batch_time:.1f}s")
            logger.info(f"  Overall best: {best_score:.4f} | Total evaluated: {total_evaluated}")
            
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                try:
                    accuracy, _ = self.trainer.compute_classification_accuracy(
                        best_variance, best_transitions, self.test_traces, coverage_weight=0.0
                    )
                    logger.info(f"  Classification accuracy (best params): {accuracy*100:.1f}%")
                except Exception as e:
                    logger.debug(f"Could not compute accuracy: {e}")
            
            if (batch_idx + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(
                    best_variance, best_transitions, best_config,
                    best_score, all_results, output_path, total_evaluated
                )
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total evaluations: {len(all_results)}")
        logger.info(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info("=" * 70)
        
        history_df = pd.DataFrame(all_results)
        
        return best_variance, best_transitions, best_config, best_score, history_df
    
    def _log_header(self, n_calls: int, batch_size: int, bounds: np.ndarray):
        """Log optimization header."""
        logger.info("=" * 70)
        logger.info("JOINT BAYESIAN OPTIMIZATION - MULTI-OBJECTIVE")
        logger.info("=" * 70)
        logger.info(f"Profile file: {self.profile_file}")
        logger.info(f"Signal file: {self.signal_file}")
        logger.info(f"Total dimensions: {self.n_dims}")
        logger.info(f"  - Variance scales: {self.n_variance}")
        logger.info(f"  - Transitions: {self.n_transitions}")
        logger.info(f"  - Objective weights: {self.n_weights}")
        logger.info(f"Per-AA transitions: {self.per_aa_transitions}")
        logger.info(f"Total calls: {n_calls}, Batch size: {batch_size}")
        logger.info("-" * 70)
        logger.info("SEARCH SPACE:")
        logger.info(f"  Variance scales: [{np.exp(bounds[0, 0]):.1f}, {np.exp(bounds[0, 1]):.1f}]")
        logger.info("  Transitions (per-parameter bounds):")
        trans_bounds = getattr(self, '_transition_bounds', TRANSITION_BOUNDS)
        for param in TRANSITION_PARAMS:
            t_min, t_max = trans_bounds[param]
            logger.info(f"    {param}: [{t_min:.4f}, {t_max:.4f}]")
        if self.optimize_weights:
            weight_idx = self.n_variance + self.n_transitions
            logger.info(f"  Alpha (LL weight): [{np.exp(bounds[weight_idx, 0]):.2f}, {np.exp(bounds[weight_idx, 1]):.2f}]")
            logger.info(f"  Beta (coverage): [{np.exp(bounds[weight_idx+1, 0]):.2f}, {np.exp(bounds[weight_idx+1, 1]):.2f}]")
            logger.info(f"  Gamma (smoothness): [{np.exp(bounds[weight_idx+2, 0]):.2f}, {np.exp(bounds[weight_idx+2, 1]):.2f}]")
        logger.info("=" * 70)
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log detailed metrics."""
        logger.info(f"    Total LL: {metrics['total_ll']:.2f}")
        logger.info(f"    Mean coverage: {metrics['mean_coverage']:.3f}")
        logger.info(f"    Mean smoothness: {metrics['mean_smoothness']:.3f}")
        logger.info(f"    Mean efficiency: {metrics['mean_efficiency']:.3f}")
    
    def _build_result_dict(
        self,
        x: np.ndarray,
        score: float,
        metrics: Dict[str, Any],
        call_num: int
    ) -> Dict:
        """Build result dictionary."""
        variance_scales, transitions, config = self.vector_to_params(x)
        
        result = {
            'call': call_num,
            'score': score,
            'total_ll': metrics['total_ll'],
            'mean_coverage': metrics['mean_coverage'],
            'mean_smoothness': metrics['mean_smoothness'],
            'mean_efficiency': metrics['mean_efficiency'],
            'alpha': config.alpha,
            'beta': config.beta,
            'gamma': config.gamma
        }
        
        for aa, scale in variance_scales.items():
            result[f'scale_{aa}'] = scale
        
        first_trans = transitions[AMINO_ACIDS[0]]
        for param, value in first_trans.items():
            result[f'trans_{param}'] = value
        
        return result
    
    def _save_checkpoint(
        self,
        variance_scales: Dict[str, float],
        transitions: Dict[str, Dict[str, float]],
        config: TrainingConfig,
        score: float,
        all_results: List[Dict],
        output_path: Path,
        total_evaluated: int
    ):
        """Save optimization checkpoint for recovery."""
        global_trans = transitions[AMINO_ACIDS[0]]
        
        checkpoint = {
            'best_score': score,
            'best_variance_scales': variance_scales,
            'best_transitions': global_trans,
            'best_config': {
                'alpha': config.alpha,
                'beta': config.beta,
                'gamma': config.gamma
            },
            'n_evaluated': total_evaluated,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = output_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        scales_df = pd.DataFrame([
            {'amino_acid': aa, 'variance_scale': scale}
            for aa, scale in sorted(variance_scales.items())
        ])
        scales_path = output_path / "checkpoint_variance_scales.csv"
        scales_df.to_csv(scales_path, index=False)
        
        trans_path = output_path / "checkpoint_transitions.json"
        with open(trans_path, 'w') as f:
            json.dump(global_trans, f, indent=2)
        
        if all_results:
            history_df = pd.DataFrame(all_results)
            history_path = output_path / "checkpoint_history.csv"
            history_df.to_csv(history_path, index=False)
        
        logger.debug(f"Checkpoint saved: {total_evaluated} evaluations, best={score:.4f}")


def load_warm_start(
    variance_scale_file: Optional[str],
    transition_file: Optional[str],
    weights_file: Optional[str]
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Load warm start parameters."""
    variance_scales = None
    transitions = None
    weights = None
    
    if variance_scale_file:
        path = Path(variance_scale_file)
        if path.suffix == '.json':
            with open(path) as f:
                variance_scales = json.load(f)
        else:
            df = pd.read_csv(path)
            variance_scales = dict(zip(df['amino_acid'], df['variance_scale']))
        logger.info(f"Loaded variance scales from {variance_scale_file}")
    
    if transition_file:
        with open(transition_file) as f:
            transitions = json.load(f)
        logger.info(f"Loaded transitions from {transition_file}")
    
    if weights_file:
        with open(weights_file) as f:
            weights = json.load(f)
        logger.info(f"Loaded objective weights from {weights_file}")
    
    return variance_scales, transitions, weights


def load_transition_bounds(bounds_file: Optional[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    """Load custom transition bounds from JSON file."""
    if not bounds_file:
        return None
    
    path = Path(bounds_file)
    if not path.exists():
        logger.warning(f"Transition bounds file not found: {bounds_file}")
        return None
    
    with open(path) as f:
        raw_bounds = json.load(f)
    
    bounds = {}
    for param, (low, high) in raw_bounds.items():
        bounds[param] = (float(low), float(high))
    
    logger.info(f"Loaded custom transition bounds from {bounds_file}")
    return bounds


def save_results(
    variance_scales: Dict[str, float],
    transitions: Dict[str, Dict[str, float]],
    config: TrainingConfig,
    score: float,
    history_df: pd.DataFrame,
    output_dir: str,
    args: argparse.Namespace
):
    """Save all optimization results."""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    scales_df = pd.DataFrame([
        {'amino_acid': aa, 'variance_scale': scale}
        for aa, scale in sorted(variance_scales.items())
    ])
    scales_path = output_path / f"optimal_variance_scales_{timestamp}.csv"
    scales_df.to_csv(scales_path, index=False)
    logger.info(f"Saved variance scales: {scales_path}")
    
    global_trans = transitions[AMINO_ACIDS[0]]
    trans_path = output_path / f"optimal_transitions_{timestamp}.json"
    with open(trans_path, 'w') as f:
        json.dump(global_trans, f, indent=2)
    logger.info(f"Saved transitions: {trans_path}")
    
    weights = {'alpha': config.alpha, 'beta': config.beta, 'gamma': config.gamma}
    weights_path = output_path / f"optimal_weights_{timestamp}.json"
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    logger.info(f"Saved objective weights: {weights_path}")
    
    history_path = output_path / f"optimization_history_{timestamp}.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f"Saved history: {history_path}")
    
    summary = {
        'timestamp': timestamp,
        'best_score': score,
        'best_variance_scales': variance_scales,
        'best_transitions': global_trans,
        'best_weights': weights,
        'n_calls': args.n_calls,
        'batch_size': args.batch_size
    }
    summary_path = output_path / f"optimization_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("OPTIMAL VARIANCE SCALES")
    print("=" * 60)
    for aa, scale in sorted(variance_scales.items()):
        print(f"  {aa}: {scale:.4f}")
    
    print("\n" + "=" * 60)
    print("OPTIMAL TRANSITIONS")
    print("=" * 60)
    for param, value in sorted(global_trans.items()):
        print(f"  {param}: {value:.6f}")
    
    print("\n" + "=" * 60)
    print("OPTIMAL OBJECTIVE WEIGHTS")
    print("=" * 60)
    print(f"  alpha (log-likelihood): {config.alpha:.4f}")
    print(f"  beta (coverage):        {config.beta:.4f}")
    print(f"  gamma (smoothness):     {config.gamma:.4f}")
    
    print("\n" + "-" * 60)
    print(f"Best Score: {score:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Joint Bayesian optimization with multi-objective training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--profile-file', type=str, required=True)
    parser.add_argument('--signal-file', type=str, required=True)
    parser.add_argument('--metadata-file', type=str, default=None)
    
    parser.add_argument('--variance-scale-file', type=str, default=None)
    parser.add_argument('--transition-file', type=str, default=None)
    parser.add_argument('--weights-file', type=str, default=None)
    
    parser.add_argument('--optimize-variance', action='store_true', default=True)
    parser.add_argument('--no-optimize-variance', dest='optimize_variance', action='store_false')
    parser.add_argument('--optimize-transitions', action='store_true', default=True)
    parser.add_argument('--no-optimize-transitions', dest='optimize_transitions', action='store_false')
    parser.add_argument('--optimize-weights', action='store_true', default=True)
    parser.add_argument('--no-optimize-weights', dest='optimize_weights', action='store_false')
    parser.add_argument('--per-aa-transitions', action='store_true', default=False)
    
    parser.add_argument('--n-calls', type=int, default=5056)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-initial', type=int, default=256)
    parser.add_argument('--scale-min', type=float, default=0.01)
    parser.add_argument('--scale-max', type=float, default=50.0)
    parser.add_argument('--transition-bounds-file', type=str, default=None)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--output-dir', type=str, default='./optimization_results')
    
    args = parser.parse_args()
    
    warm_variance, warm_trans, warm_weights = load_warm_start(
        args.variance_scale_file,
        args.transition_file,
        args.weights_file
    )
    
    custom_trans_bounds = load_transition_bounds(
        getattr(args, 'transition_bounds_file', None)
    )
    
    optimizer = JointBayesianOptimizer(
        profile_file=args.profile_file,
        signal_file=args.signal_file,
        metadata_file=args.metadata_file,
        per_aa_transitions=args.per_aa_transitions,
        optimize_weights=args.optimize_weights,
        optimize_variance=args.optimize_variance,
        optimize_transitions=args.optimize_transitions,
        n_jobs=args.n_jobs
    )
    
    best_variance, best_trans, best_config, best_score, history = optimizer.optimize(
        n_calls=args.n_calls,
        batch_size=args.batch_size,
        n_initial=args.n_initial,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        transition_bounds=custom_trans_bounds,
        warm_start_variance=warm_variance,
        warm_start_transitions=warm_trans,
        warm_start_weights=warm_weights,
        output_dir=args.output_dir
    )
    
    save_results(
        best_variance, best_trans, best_config, best_score,
        history, args.output_dir, args
    )


if __name__ == '__main__':
    main()