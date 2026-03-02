#!/usr/bin/env python
"""
Batch Bayesian optimization for HMM parameters — accuracy maximization.

Evaluation modes:
  single:  Fixed test metadata (original behavior, fast, prone to overfit)
  shuffle: Each evaluation randomly holds out N traces per AA from the
           train+test pool. Prevents BO from memorizing specific traces.
           Profiles remain fixed (built from original train set).

Validation is NEVER used during optimization — evaluated once at the end.

Optimizes:
  - 20 variance scales (one per amino acid)  [optional]
  - 6 transition probabilities (global)       [optional]

Usage:
    # Original single-split (fast, but overfits over many iterations)
    python batch_bayesian_optimization.py \
        --profile-file data/amino_acid_profiles.csv \
        --signal-file data/signals.pkl \
        --metadata-file data/test_metadata.json \
        --eval-mode single \
        --optimize variance \
        --n-calls 2500 --batch-size 32

    # Randomized holdout (recommended)
    python batch_bayesian_optimization.py \
        --profile-file data/amino_acid_profiles.csv \
        --signal-file data/signals.pkl \
        --metadata-file data/test_metadata.json \
        --train-metadata-file data/train_metadata.json \
        --val-metadata-file data/val_metadata.json \
        --eval-mode shuffle \
        --holdout-per-aa 5 \
        --optimize variance \
        --n-calls 2500 --batch-size 32
"""

import argparse
import json
import subprocess
import shutil
import tempfile
import time
import random
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
N_AA = len(AMINO_ACIDS)

TRANSITION_PARAMS = [
    'match_self_loop',
    'forward',
    'to_skip',
    'to_slip',
    'to_insert',
    'to_end'
]
N_TRANS = len(TRANSITION_PARAMS)

DEFAULT_TRANSITIONS = {
    'match_self_loop': 0.012,
    'forward': 0.679,
    'to_skip': 0.123,
    'to_slip': 0.086,
    'to_insert': 0.062,
    'to_end': 0.037
}

DEFAULT_TRANSITION_BOUNDS = {
    'match_self_loop': (0.001, 0.15),
    'forward':         (0.3, 0.95),
    'to_skip':         (0.005, 0.35),
    'to_slip':         (0.005, 0.25),
    'to_insert':       (0.005, 0.25),
    'to_end':          (0.001, 0.15)
}


# ──────────────────────────────────────────────
# Parameter space
# ──────────────────────────────────────────────

class ParameterSpace:
    """Maps flat log-space vector ↔ named HMM parameters."""

    def __init__(
        self,
        optimize_variance: bool = True,
        optimize_transitions: bool = False,
        scale_min: float = 0.1,
        scale_max: float = 5.0,
        transition_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.optimize_variance = optimize_variance
        self.optimize_transitions = optimize_transitions
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.trans_bounds = DEFAULT_TRANSITION_BOUNDS.copy()
        if transition_bounds:
            self.trans_bounds.update(transition_bounds)

        self.n_variance = N_AA if optimize_variance else 0
        self.n_trans = N_TRANS if optimize_transitions else 0
        self.n_dims = self.n_variance + self.n_trans

    def get_bounds(self) -> np.ndarray:
        bounds = np.zeros((self.n_dims, 2))
        idx = 0
        if self.optimize_variance:
            bounds[idx:idx + N_AA, 0] = np.log(self.scale_min)
            bounds[idx:idx + N_AA, 1] = np.log(self.scale_max)
            idx += N_AA
        if self.optimize_transitions:
            for i, param in enumerate(TRANSITION_PARAMS):
                lo, hi = self.trans_bounds[param]
                bounds[idx + i, 0] = np.log(lo)
                bounds[idx + i, 1] = np.log(hi)
        return bounds

    def vector_to_params(
        self, x: np.ndarray,
        default_scales: Optional[Dict[str, float]] = None,
        default_trans: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        idx = 0
        if self.optimize_variance:
            scales = {aa: float(np.exp(x[idx + i])) for i, aa in enumerate(AMINO_ACIDS)}
            idx += N_AA
        else:
            scales = (default_scales or {aa: 1.0 for aa in AMINO_ACIDS}).copy()

        if self.optimize_transitions:
            raw = {p: float(np.exp(x[idx + i])) for i, p in enumerate(TRANSITION_PARAMS)}
            total = sum(raw.values())
            transitions = {k: v / total for k, v in raw.items()}
        else:
            transitions = (default_trans or DEFAULT_TRANSITIONS).copy()

        return scales, transitions

    def params_to_vector(
        self, scales: Dict[str, float], transitions: Dict[str, float],
    ) -> np.ndarray:
        x = np.zeros(self.n_dims)
        idx = 0
        if self.optimize_variance:
            for i, aa in enumerate(AMINO_ACIDS):
                x[idx + i] = np.log(scales.get(aa, 1.0))
            idx += N_AA
        if self.optimize_transitions:
            for i, p in enumerate(TRANSITION_PARAMS):
                x[idx + i] = np.log(transitions.get(p, DEFAULT_TRANSITIONS[p]))
        return x

    def describe(self) -> str:
        parts = []
        if self.optimize_variance:
            parts.append(f"{N_AA} variance scales [{self.scale_min}, {self.scale_max}]")
        if self.optimize_transitions:
            parts.append(f"{N_TRANS} transitions")
        return f"{self.n_dims} dims: " + ", ".join(parts)


# ──────────────────────────────────────────────
# Metadata loading and shuffled holdout
# ──────────────────────────────────────────────

def _load_meta(path: str) -> List[Dict]:
    """Load metadata, handling both flat list and {traces: [...]} format."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "traces" in data:
        return data["traces"]
    return data


class ShuffledHoldoutPool:
    """
    Manages a pool of train+test traces and generates random holdout
    splits on demand. Each call to sample_holdout() returns a different
    random subset of traces for evaluation.

    The pool is stratified by AA so each holdout has exactly
    holdout_per_aa traces per amino acid.
    """

    def __init__(
        self,
        train_meta_path: str,
        test_meta_path: str,
        holdout_per_aa: int = 5,
        seed: Optional[int] = None,
    ):
        train_meta = _load_meta(train_meta_path)
        test_meta = _load_meta(test_meta_path)
        self.pool = train_meta + test_meta
        self.holdout_per_aa = holdout_per_aa

        # Group by AA
        self.by_aa: Dict[str, List[Dict]] = defaultdict(list)
        for m in self.pool:
            self.by_aa[m["AA"]].append(m)

        # Validate
        self.aa_list = sorted(self.by_aa.keys())
        for aa in self.aa_list:
            n = len(self.by_aa[aa])
            if n < holdout_per_aa:
                logger.warning(
                    f"  AA '{aa}' has only {n} traces in pool, "
                    f"but holdout_per_aa={holdout_per_aa}. "
                    f"Will use {n} for this AA."
                )

        self.rng = random.Random(seed)

        total = len(self.pool)
        holdout_total = sum(
            min(holdout_per_aa, len(self.by_aa[aa])) for aa in self.aa_list
        )
        logger.info(f"  Shuffle pool: {total} traces across {len(self.aa_list)} AAs")
        logger.info(f"  Each holdout: ~{holdout_total} traces "
                     f"({holdout_per_aa}/AA)")

    def sample_holdout(self) -> List[Dict]:
        """Return a random stratified holdout subset."""
        holdout = []
        for aa in self.aa_list:
            traces = self.by_aa[aa]
            k = min(self.holdout_per_aa, len(traces))
            holdout.extend(self.rng.sample(traces, k))
        return holdout

    def write_holdout_file(self, output_path: str) -> str:
        """Sample a holdout and write to a temp metadata JSON file."""
        holdout = self.sample_holdout()
        with open(output_path, 'w') as f:
            json.dump({"traces": holdout}, f)
        return output_path


# ──────────────────────────────────────────────
# Subprocess evaluation
# ──────────────────────────────────────────────

def run_single_evaluation(
    scales: Dict[str, float],
    transitions: Dict[str, float],
    signal_file: str,
    profile_file: Optional[str] = None,
    barycenter_file: Optional[str] = None,
    metadata_file: Optional[str] = None,
    variance_mode: str = 'segment',
    timeout: int = 300,
) -> float:
    """Run one HMM classification, return accuracy 0–100."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('amino_acid,variance_scale\n')
        for aa in AMINO_ACIDS:
            f.write(f'{aa},{scales.get(aa, 1.0)}\n')
        scale_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(transitions, f)
        trans_file = f.name

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            'python', '-m', 'vrhmm.cli.main',
            '--signal-file', signal_file,
            '--variance-scale-file', scale_file,
            '--transition-file', trans_file,
            '--output-dir', tmp_dir,
            '--no-plots'
        ]
        if profile_file:
            cmd.extend(['--profile-file', profile_file])
        elif barycenter_file:
            cmd.extend(['--barycenter-file', barycenter_file])
            cmd.extend(['--variance-mode', variance_mode])
        else:
            return 0.0
        if metadata_file:
            cmd.extend(['--metadata-file', metadata_file])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            for output in [result.stderr, result.stdout]:
                for line in output.split('\n'):
                    if 'Accuracy:' in line:
                        try:
                            acc_str = line.split('Accuracy:')[1].strip()
                            if '%' in acc_str:
                                return float(acc_str.replace('%', ''))
                            else:
                                return float(acc_str) * 100
                        except (ValueError, IndexError):
                            continue
            return 0.0
        except subprocess.TimeoutExpired:
            logger.warning("Pipeline timed out")
            return 0.0
        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return 0.0
        finally:
            Path(scale_file).unlink(missing_ok=True)
            Path(trans_file).unlink(missing_ok=True)


def evaluate_single_point_shuffle(
    x: np.ndarray,
    param_space: ParameterSpace,
    signal_file: str,
    profile_file: Optional[str],
    barycenter_file: Optional[str],
    variance_mode: str,
    default_scales: Optional[Dict[str, float]],
    default_trans: Optional[Dict[str, float]],
    holdout_pool: ShuffledHoldoutPool,
) -> float:
    """
    Evaluate one parameter vector using a fresh random holdout.
    Each call samples a different holdout set from the pool.
    """
    scales, trans = param_space.vector_to_params(x, default_scales, default_trans)

    # Write a unique holdout file for this evaluation
    # Use PID + random suffix to avoid collisions in parallel
    tmp_meta = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', prefix=f'holdout_{os.getpid()}_',
        delete=False
    )
    tmp_meta_path = tmp_meta.name
    tmp_meta.close()

    try:
        holdout_pool.write_holdout_file(tmp_meta_path)
        acc = run_single_evaluation(
            scales, trans, signal_file,
            profile_file, barycenter_file,
            tmp_meta_path, variance_mode
        )
        return acc
    finally:
        Path(tmp_meta_path).unlink(missing_ok=True)


def evaluate_single_point_fixed(
    x: np.ndarray,
    param_space: ParameterSpace,
    signal_file: str,
    profile_file: Optional[str],
    barycenter_file: Optional[str],
    variance_mode: str,
    default_scales: Optional[Dict[str, float]],
    default_trans: Optional[Dict[str, float]],
    metadata_file: str,
) -> float:
    """Evaluate one parameter vector on fixed test metadata."""
    scales, trans = param_space.vector_to_params(x, default_scales, default_trans)
    return run_single_evaluation(
        scales, trans, signal_file,
        profile_file, barycenter_file,
        metadata_file, variance_mode
    )


def evaluate_batch(
    X_batch: np.ndarray,
    param_space: ParameterSpace,
    signal_file: str,
    profile_file: Optional[str],
    barycenter_file: Optional[str],
    variance_mode: str,
    default_scales: Optional[Dict[str, float]],
    default_trans: Optional[Dict[str, float]],
    n_jobs: int,
    # Single mode
    metadata_file: Optional[str] = None,
    # Shuffle mode
    holdout_pool: Optional[ShuffledHoldoutPool] = None,
) -> np.ndarray:
    """Evaluate a batch in parallel."""

    if holdout_pool is not None:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(evaluate_single_point_shuffle)(
                x, param_space, signal_file, profile_file,
                barycenter_file, variance_mode, default_scales, default_trans,
                holdout_pool,
            )
            for x in X_batch
        )
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(evaluate_single_point_fixed)(
                x, param_space, signal_file, profile_file,
                barycenter_file, variance_mode, default_scales, default_trans,
                metadata_file,
            )
            for x in X_batch
        )

    return np.array(results)


# ──────────────────────────────────────────────
# Acquisition function
# ──────────────────────────────────────────────

def propose_batch_points(
    gp: GaussianProcessRegressor,
    y_best: float,
    bounds: np.ndarray,
    batch_size: int,
    n_restarts: int = 5,
    xi: float = 0.01,
) -> np.ndarray:
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


def _fit_gp(X_observed, y_observed):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6,
        normalize_y=True, n_restarts_optimizer=3
    )
    gp.fit(np.array(X_observed), np.array(y_observed))
    return gp


# ──────────────────────────────────────────────
# Main optimization loop
# ──────────────────────────────────────────────

def batch_bayesian_optimization(
    signal_file: str,
    param_space: ParameterSpace,
    profile_file: Optional[str] = None,
    barycenter_file: Optional[str] = None,
    metadata_file: Optional[str] = None,
    train_metadata_file: Optional[str] = None,
    val_metadata_file: Optional[str] = None,
    variance_mode: str = 'segment',
    eval_mode: str = 'single',
    holdout_per_aa: int = 5,
    n_calls: int = 2500,
    batch_size: int = 32,
    n_initial: int = 256,
    n_jobs: int = -1,
    warm_start_scales: Optional[Dict[str, float]] = None,
    warm_start_trans: Optional[Dict[str, float]] = None,
    output_dir: str = './optimization_results',
    checkpoint_freq: int = 2,
) -> Tuple[Dict[str, float], Dict[str, float], float, pd.DataFrame]:
    """
    Batch BO maximizing classification accuracy.

    eval_mode='single': Fixed test metadata each evaluation (original).
    eval_mode='shuffle': Random holdout from train+test pool each evaluation.
                         Same cost per eval as single, but BO can't memorize
                         specific traces because the eval set changes every time.

    Validation is evaluated ONCE at the end — never during optimization.

    Returns (best_scales, best_transitions, best_acc, history_df)
    """
    if not profile_file and not barycenter_file:
        raise ValueError("Must provide either --profile-file or --barycenter-file")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bounds = param_space.get_bounds()
    default_scales = warm_start_scales or {aa: 1.0 for aa in AMINO_ACIDS}
    default_trans = warm_start_trans or DEFAULT_TRANSITIONS.copy()

    # ── Set up evaluation mode ──
    holdout_pool = None

    if eval_mode == 'shuffle':
        if not train_metadata_file or not metadata_file:
            raise ValueError("Shuffle mode requires --train-metadata-file and --metadata-file")
        logger.info("Setting up shuffled holdout pool...")
        holdout_pool = ShuffledHoldoutPool(
            train_metadata_file, metadata_file,
            holdout_per_aa=holdout_per_aa,
        )
        eval_label = f"shuffled holdout ({holdout_per_aa}/AA)"
    else:
        eval_label = "fixed test split"

    X_observed: List[np.ndarray] = []
    y_observed: List[float] = []
    all_results: List[Dict] = []

    best_acc = 0.0
    best_scales = default_scales.copy()
    best_trans = default_trans.copy()

    # ── Header ──
    logger.info("=" * 70)
    logger.info("BATCH BAYESIAN OPTIMIZATION — ACCURACY MAXIMIZATION")
    logger.info("=" * 70)
    logger.info(f"BO objective: {eval_label}")
    logger.info(f"Parameter space: {param_space.describe()}")
    if profile_file:
        logger.info(f"Profile file: {profile_file}")
    else:
        logger.info(f"Barycenter file: {barycenter_file}")
    logger.info(f"Signal file: {signal_file}")
    logger.info(f"Test metadata: {metadata_file}")
    if train_metadata_file:
        logger.info(f"Train metadata: {train_metadata_file}")
    logger.info(f"Val metadata: {val_metadata_file or 'NONE'}")
    logger.info(f"Eval mode: {eval_mode}")
    if eval_mode == 'shuffle':
        logger.info(f"Holdout per AA: {holdout_per_aa}")
    logger.info(f"Calls: {n_calls}, Batch: {batch_size}, Initial: {n_initial}")
    logger.info(f"Parallel jobs: {n_jobs}")
    logger.info("=" * 70)

    start_time = time.time()
    total_evaluated = 0

    # ── Warm start ──
    if warm_start_scales or warm_start_trans:
        logger.info("Evaluating warm start configuration...")
        warm_x = param_space.params_to_vector(default_scales, default_trans)

        # Warm start always evaluated on fixed test for a stable baseline number
        warm_acc = run_single_evaluation(
            default_scales, default_trans, signal_file,
            profile_file, barycenter_file, metadata_file, variance_mode
        )
        X_observed.append(warm_x)
        y_observed.append(warm_acc)
        all_results.append(_build_result(
            warm_x, warm_acc, param_space, default_scales, default_trans, 0
        ))

        if warm_acc > best_acc:
            best_acc = warm_acc
            best_scales = default_scales.copy()
            best_trans = default_trans.copy()

        logger.info(f"Warm start accuracy (fixed test): {warm_acc:.2f}%")
        total_evaluated = 1

    # ── Main loop ──
    n_batches = (n_calls - total_evaluated + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_end = min(total_evaluated + batch_size, n_calls)
        current_batch_size = batch_end - total_evaluated
        if current_batch_size <= 0:
            break

        batch_t0 = time.time()
        logger.info(f"\nBatch {batch_idx + 1}/{n_batches}: "
                     f"evaluating {current_batch_size} points...")

        # Generate candidates
        if len(X_observed) < n_initial:
            n_random = min(current_batch_size, n_initial - len(X_observed))
            X_batch = np.random.uniform(
                bounds[:, 0], bounds[:, 1],
                size=(n_random, param_space.n_dims)
            )
            if n_random < current_batch_size and len(X_observed) >= 20:
                gp = _fit_gp(X_observed, y_observed)
                X_gp = propose_batch_points(
                    gp, max(y_observed), bounds,
                    current_batch_size - n_random
                )
                X_batch = np.vstack([X_batch, X_gp])
        else:
            logger.info(f"  Fitting GP on {len(X_observed)} observations...")
            gp = _fit_gp(X_observed, y_observed)
            logger.info(f"  Proposing {current_batch_size} candidates...")
            X_batch = propose_batch_points(
                gp, max(y_observed), bounds, current_batch_size
            )

        # Evaluate
        logger.info(f"  Running {len(X_batch)} parallel evaluations "
                     f"({eval_label})...")
        y_batch = evaluate_batch(
            X_batch, param_space, signal_file,
            profile_file, barycenter_file, variance_mode,
            default_scales, default_trans, n_jobs,
            metadata_file=metadata_file,
            holdout_pool=holdout_pool,
        )

        batch_time = time.time() - batch_t0

        # Store
        for i in range(len(X_batch)):
            X_observed.append(X_batch[i])
            y_observed.append(y_batch[i])

            scales_i, trans_i = param_space.vector_to_params(
                X_batch[i], default_scales, default_trans
            )
            all_results.append(_build_result(
                X_batch[i], y_batch[i], param_space,
                scales_i, trans_i, total_evaluated + i + 1
            ))

            if y_batch[i] > best_acc:
                best_acc = y_batch[i]
                best_scales = scales_i
                best_trans = trans_i
                logger.info(f"  *** New best: {best_acc:.2f}% ***")
                if param_space.optimize_transitions:
                    logger.info(
                        f"    fwd={best_trans['forward']:.4f} "
                        f"skip={best_trans['to_skip']:.4f} "
                        f"slip={best_trans['to_slip']:.4f} "
                        f"ins={best_trans['to_insert']:.4f}"
                    )

        total_evaluated += len(X_batch)

        batch_best = np.max(y_batch)
        batch_mean = np.mean(y_batch)
        batch_std = np.std(y_batch)
        logger.info(f"  Batch: best={batch_best:.2f}%, "
                     f"mean={batch_mean:.2f}% ± {batch_std:.2f}%, "
                     f"time={batch_time:.1f}s")
        logger.info(f"  Overall best: {best_acc:.2f}% | "
                     f"Evaluated: {total_evaluated}")

        # Checkpoint
        if (batch_idx + 1) % checkpoint_freq == 0:
            _save_checkpoint(
                best_scales, best_trans, best_acc,
                all_results, output_path
            )

    elapsed = time.time() - start_time

    # ── Final: evaluate best params on fixed test and validation ──
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)

    # Always report fixed test accuracy for comparability
    if eval_mode == 'shuffle' and metadata_file:
        fixed_test_acc = run_single_evaluation(
            best_scales, best_trans, signal_file,
            profile_file, barycenter_file,
            metadata_file, variance_mode
        )
        logger.info(f"  Fixed test accuracy:  {fixed_test_acc:.2f}%")
    else:
        fixed_test_acc = best_acc
        logger.info(f"  Test accuracy:        {fixed_test_acc:.2f}%")

    val_acc = None
    if val_metadata_file:
        val_acc = run_single_evaluation(
            best_scales, best_trans, signal_file,
            profile_file, barycenter_file,
            val_metadata_file, variance_mode
        )
        logger.info(f"  Validation accuracy:  {val_acc:.2f}%")
        gap = fixed_test_acc - val_acc
        logger.info(f"  Test→Val gap:         {gap:+.2f}%")
        if gap > 10:
            logger.warning("  Large gap — parameters may be overfitting")
        elif gap < 5:
            logger.info("  Small gap — good generalisation")

    logger.info(f"\n  BO objective ({eval_label}): best={best_acc:.2f}%")
    logger.info(f"  Total evaluations: {total_evaluated}")
    logger.info(f"  Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("=" * 70)

    # Append final eval results to history
    final_entry = {
        'call': total_evaluated + 1,
        'accuracy': fixed_test_acc,
        'is_final_test': True,
    }
    if val_acc is not None:
        final_entry['val_accuracy'] = val_acc
    if param_space.optimize_variance:
        for aa in AMINO_ACIDS:
            final_entry[f'scale_{aa}'] = best_scales.get(aa, 1.0)
    if param_space.optimize_transitions:
        for p in TRANSITION_PARAMS:
            final_entry[f'trans_{p}'] = best_trans.get(p, 0.0)
    all_results.append(final_entry)

    history_df = pd.DataFrame(all_results)
    return best_scales, best_trans, best_acc, history_df


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _build_result(x, accuracy, param_space, scales, trans, call_num):
    result = {'call': call_num, 'accuracy': accuracy}
    if param_space.optimize_variance:
        for aa in AMINO_ACIDS:
            result[f'scale_{aa}'] = scales.get(aa, 1.0)
    if param_space.optimize_transitions:
        for p in TRANSITION_PARAMS:
            result[f'trans_{p}'] = trans.get(p, 0.0)
    return result


def _save_checkpoint(scales, trans, accuracy, results, output_path):
    checkpoint = {
        'best_accuracy': accuracy,
        'best_scales': scales,
        'best_transitions': trans,
        'n_evaluated': len(results),
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_path / "checkpoint.json", 'w') as f:
        json.dump(checkpoint, f, indent=2)

    scales_df = pd.DataFrame([
        {'amino_acid': aa, 'variance_scale': s}
        for aa, s in sorted(scales.items())
    ])
    scales_df.to_csv(output_path / "checkpoint_variance_scales.csv", index=False)

    with open(output_path / "checkpoint_transitions.json", 'w') as f:
        json.dump(trans, f, indent=2)

    if results:
        pd.DataFrame(results).to_csv(
            output_path / "checkpoint_history.csv", index=False
        )


# ──────────────────────────────────────────────
# Warm start loading
# ──────────────────────────────────────────────

def load_warm_start_scales(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    p = Path(path)
    if p.suffix == '.json':
        with open(p) as f:
            data = json.load(f)
        if 'best_scales' in data:
            return data['best_scales']
        return data
    else:
        df = pd.read_csv(p)
        return dict(zip(df['amino_acid'], df['variance_scale']))


def load_warm_start_trans(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    with open(path) as f:
        data = json.load(f)
    if 'best_transitions' in data:
        return data['best_transitions']
    return data


# ──────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────

def save_results(best_scales, best_trans, best_acc, history_df, output_dir, args):
    output_path = Path(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    scales_df = pd.DataFrame([
        {'amino_acid': aa, 'variance_scale': s}
        for aa, s in sorted(best_scales.items())
    ])
    scales_path = output_path / f"optimal_variance_scales_{ts}.csv"
    scales_df.to_csv(scales_path, index=False)
    logger.info(f"Saved variance scales: {scales_path}")

    trans_path = output_path / f"optimal_transitions_{ts}.json"
    with open(trans_path, 'w') as f:
        json.dump(best_trans, f, indent=2)
    logger.info(f"Saved transitions: {trans_path}")

    history_path = output_path / f"optimization_history_{ts}.csv"
    history_df.to_csv(history_path, index=False)

    # Extract final val accuracy if present
    final_rows = history_df[history_df.get('is_final_test', False) == True]
    val_acc = None
    if len(final_rows) > 0 and 'val_accuracy' in final_rows.columns:
        val_acc = final_rows['val_accuracy'].iloc[0]
        if pd.isna(val_acc):
            val_acc = None

    summary = {
        'timestamp': ts,
        'best_optimization_accuracy': best_acc,
        'final_val_accuracy': val_acc,
        'eval_mode': args.eval_mode,
        'holdout_per_aa': args.holdout_per_aa if args.eval_mode == 'shuffle' else None,
        'optimize_mode': args.optimize,
        'best_scales': best_scales,
        'best_transitions': best_trans,
        'n_calls': args.n_calls,
        'batch_size': args.batch_size,
    }
    with open(output_path / f"optimization_summary_{ts}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS")
    print("=" * 60)
    print("\nVariance scales:")
    for aa, s in sorted(best_scales.items()):
        print(f"  {aa}: {s:.4f}")
    print("\nTransitions:")
    for p, v in sorted(best_trans.items()):
        print(f"  {p}: {v:.6f}")
    print(f"\n  Optimization accuracy: {best_acc:.2f}%")
    if val_acc is not None:
        print(f"  Validation accuracy:   {val_acc:.2f}%")
    print("=" * 60)

    print(f"\nTo use these results:")
    print(f"  python -m vrhmm.cli.main \\")
    if args.profile_file:
        print(f"    --profile-file {args.profile_file} \\")
    else:
        print(f"    --barycenter-file {args.barycenter_file} \\")
    print(f"    --signal-file {args.signal_file} \\")
    print(f"    --variance-scale-file {scales_path} \\")
    print(f"    --transition-file {trans_path} \\")
    print(f"    --output-dir ./results_optimized")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Batch BO — maximize accuracy with optional shuffled holdout',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--profile-file', type=str, default=None)
    input_group.add_argument('--barycenter-file', type=str, default=None)

    parser.add_argument('--signal-file', type=str, required=True)
    parser.add_argument('--metadata-file', type=str, default=None,
                        help='Test metadata (BO objective in single mode, '
                             'pooled with train in shuffle mode)')
    parser.add_argument('--train-metadata-file', type=str, default=None,
                        help='Train metadata (required for shuffle mode)')
    parser.add_argument('--val-metadata-file', type=str, default=None,
                        help='Validation metadata — evaluated ONCE at the end, '
                             'never used during optimization')
    parser.add_argument('--variance-mode', type=str, default='segment',
                        choices=['barycenter', 'segment'])

    parser.add_argument('--optimize', type=str, default='variance',
                        choices=['variance', 'transitions', 'both'])

    # Evaluation mode
    parser.add_argument('--eval-mode', type=str, default='single',
                        choices=['single', 'shuffle'],
                        help='single: fixed test set each eval. '
                             'shuffle: random holdout from train+test pool '
                             'each eval (same speed, better generalisation)')
    parser.add_argument('--holdout-per-aa', type=int, default=5,
                        help='Traces to hold out per AA in shuffle mode')

    # Warm start
    parser.add_argument('--variance-scale-file', type=str, default=None)
    parser.add_argument('--transition-file', type=str, default=None)

    # Optimization
    parser.add_argument('--n-calls', type=int, default=2500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-initial', type=int, default=256)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--scale-min', type=float, default=0.1)
    parser.add_argument('--scale-max', type=float, default=5.0)
    parser.add_argument('--output-dir', type=str, default='./optimization_results')
    parser.add_argument('--checkpoint-freq', type=int, default=2)

    args = parser.parse_args()

    if not args.profile_file and not args.barycenter_file:
        parser.error("Must provide either --profile-file or --barycenter-file")
    if args.eval_mode == 'shuffle' and not args.train_metadata_file:
        parser.error("Shuffle mode requires --train-metadata-file")
    if args.eval_mode == 'single' and not args.metadata_file:
        parser.error("Single mode requires --metadata-file")

    param_space = ParameterSpace(
        optimize_variance=args.optimize in ('variance', 'both'),
        optimize_transitions=args.optimize in ('transitions', 'both'),
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )

    warm_scales = load_warm_start_scales(args.variance_scale_file)
    warm_trans = load_warm_start_trans(args.transition_file)

    best_scales, best_trans, best_acc, history = batch_bayesian_optimization(
        signal_file=args.signal_file,
        param_space=param_space,
        profile_file=args.profile_file,
        barycenter_file=args.barycenter_file,
        metadata_file=args.metadata_file,
        train_metadata_file=args.train_metadata_file,
        val_metadata_file=args.val_metadata_file,
        variance_mode=args.variance_mode,
        eval_mode=args.eval_mode,
        holdout_per_aa=args.holdout_per_aa,
        n_calls=args.n_calls,
        batch_size=args.batch_size,
        n_initial=args.n_initial,
        n_jobs=args.n_jobs,
        warm_start_scales=warm_scales,
        warm_start_trans=warm_trans,
        output_dir=args.output_dir,
        checkpoint_freq=args.checkpoint_freq,
    )

    save_results(best_scales, best_trans, best_acc, history, args.output_dir, args)


if __name__ == '__main__':
    main()