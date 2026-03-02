#!/usr/bin/env python
"""
Batch Bayesian optimization with parallel evaluations for per-AA variance scales.

Supports both barycenter files and pre-computed profile CSVs.

Usage:
    # With barycenter file
    python batch_bayesian_optimization.py \
        --barycenter-file data/dba_centroids.json \
        --signal-file data/signals.pkl \
        --n-calls 2500 \
        --batch-size 256 \
        --output-dir ./optimization_results

    # With profile CSV (from DBA pipeline)
    python batch_bayesian_optimization.py \
        --profile-file data/amino_acid_profiles.csv \
        --signal-file data/signals.pkl \
        --metadata-file data/test_metadata.json \
        --n-calls 2500 \
        --batch-size 256 \
        --output-dir ./optimization_results

Requirements:
    pip install scikit-learn scipy pandas numpy joblib
"""

import argparse
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
N_DIMS = len(AMINO_ACIDS)


def run_single_evaluation(
    scales: Dict[str, float],
    signal_file: str,
    barycenter_file: Optional[str] = None,
    profile_file: Optional[str] = None,
    metadata_file: Optional[str] = None,
    variance_mode: str = 'segment',
    timeout: int = 300
) -> float:
    """Run a single classification with given variance scales."""
    
    # Create temporary variance scale file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('amino_acid,variance_scale\n')
        for aa, scale in scales.items():
            f.write(f'{aa},{scale}\n')
        scale_file = f.name
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            'python', '-m', 'vrhmm.cli.main',
            '--signal-file', signal_file,
            '--variance-scale-file', scale_file,
            '--output-dir', tmp_dir,
            '--no-plots'
        ]
        
        # Add profile or barycenter file
        if profile_file:
            cmd.extend(['--profile-file', profile_file])
        elif barycenter_file:
            cmd.extend(['--barycenter-file', barycenter_file])
            cmd.extend(['--variance-mode', variance_mode])
        else:
            logger.error("Must provide either --profile-file or --barycenter-file")
            return 0.0
        
        # Add metadata file if provided
        if metadata_file:
            cmd.extend(['--metadata-file', metadata_file])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            # DEBUG: Print error if failed
            if result.returncode != 0:
                logger.debug(f"Pipeline failed! stderr: {result.stderr[:500]}")
            
            # Check both stderr and stdout for accuracy
            for output in [result.stderr, result.stdout]:
                for line in output.split('\n'):
                    if 'Accuracy:' in line:
                        try:
                            acc_str = line.split('Accuracy:')[1].strip()
                            # Handle both "85.5%" and "0.855" formats
                            if '%' in acc_str:
                                acc = float(acc_str.replace('%', ''))
                            else:
                                acc = float(acc_str) * 100
                            return acc
                        except (ValueError, IndexError):
                            continue
            
            logger.debug(f"No accuracy found. Return code: {result.returncode}")
            return 0.0
            
        except subprocess.TimeoutExpired:
            logger.warning("Pipeline timed out")
            return 0.0
        except Exception as e:
            logger.warning(f"Error: {e}")
            return 0.0
        finally:
            Path(scale_file).unlink(missing_ok=True)


def evaluate_batch_parallel(
    X_batch: np.ndarray,
    signal_file: str,
    barycenter_file: Optional[str],
    profile_file: Optional[str],
    metadata_file: Optional[str],
    variance_mode: str,
    n_jobs: int
) -> np.ndarray:
    """Evaluate a batch of points in parallel using joblib."""
    
    def eval_point(x):
        scales = {aa: float(x[i]) for i, aa in enumerate(AMINO_ACIDS)}
        return run_single_evaluation(
            scales, signal_file, barycenter_file, profile_file, 
            metadata_file, variance_mode
        )
    
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(eval_point)(x) for x in X_batch
    )
    
    return np.array(results)


def expected_improvement(
    X: np.ndarray, 
    gp: GaussianProcessRegressor, 
    y_best: float, 
    xi: float = 0.01
) -> np.ndarray:
    """Compute Expected Improvement acquisition function."""
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    
    imp = mu - y_best - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    return ei


def propose_batch_points(
    gp: GaussianProcessRegressor,
    y_best: float,
    bounds: np.ndarray,
    batch_size: int,
    n_restarts: int = 5
) -> np.ndarray:
    """Propose a batch of diverse points using Expected Improvement with penalization."""
    
    candidates = []
    X_pending = []
    
    for b in range(batch_size):
        best_ei = -np.inf
        best_x = None
        
        # Random restarts for optimization
        for _ in range(n_restarts):
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            
            def neg_ei(x):
                ei = expected_improvement(x.reshape(1, -1), gp, y_best)[0]
                
                # Penalize points close to already selected candidates
                if X_pending:
                    for xp in X_pending:
                        dist = np.linalg.norm(x - xp)
                        ei *= (1 - np.exp(-dist / 0.5))  # Diversity penalty
                
                return -ei
            
            try:
                result = minimize(
                    neg_ei,
                    x0,
                    bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                    method='L-BFGS-B'
                )
                
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_x = result.x
            except:
                pass
        
        if best_x is not None:
            candidates.append(best_x)
            X_pending.append(best_x)
        else:
            # Fallback to random
            rand_x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            candidates.append(rand_x)
            X_pending.append(rand_x)
    
    return np.array(candidates)


def batch_bayesian_optimization(
    signal_file: str,
    barycenter_file: Optional[str] = None,
    profile_file: Optional[str] = None,
    metadata_file: Optional[str] = None,
    variance_mode: str = 'segment',
    n_calls: int = 2500,
    batch_size: int = 256,
    n_initial: int = 64,
    scale_min: float = 0.1,
    scale_max: float = 5.0,
    output_dir: str = './optimization_results'
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    """
    Batch Bayesian optimization with parallel evaluations.
    
    Args:
        signal_file: Path to signal data
        barycenter_file: Path to barycenter JSON (optional)
        profile_file: Path to profile CSV (optional, preferred)
        metadata_file: Path to metadata JSON for filtering signals (optional)
        variance_mode: 'segment' or 'barycenter' (only used with barycenter_file)
        n_calls: Total evaluations
        batch_size: Evaluations per batch (set to number of cores)
        n_initial: Random samples before Bayesian modeling
        scale_min: Minimum variance scale
        scale_max: Maximum variance scale
        output_dir: Output directory
    
    Returns:
        Tuple of (best_scales, best_accuracy, history_df)
    """
    if not profile_file and not barycenter_file:
        raise ValueError("Must provide either --profile-file or --barycenter-file")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use log scale for bounds (variance scales work better in log space)
    log_bounds = np.array([[np.log(scale_min), np.log(scale_max)]] * N_DIMS)
    
    # Storage
    X_observed = []
    y_observed = []
    all_results = []
    
    best_accuracy = 0.0
    best_scales = {aa: 1.0 for aa in AMINO_ACIDS}
    
    logger.info("=" * 70)
    logger.info("BATCH BAYESIAN OPTIMIZATION")
    logger.info("=" * 70)
    if profile_file:
        logger.info(f"Profile file: {profile_file}")
    else:
        logger.info(f"Barycenter file: {barycenter_file}")
        logger.info(f"Variance mode: {variance_mode}")
    logger.info(f"Signal file: {signal_file}")
    if metadata_file:
        logger.info(f"Metadata file: {metadata_file}")
    logger.info(f"Total calls: {n_calls}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Initial random samples: {n_initial}")
    logger.info(f"Scale range: [{scale_min}, {scale_max}]")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)
    
    start_time = time.time()
    n_batches = (n_calls + batch_size - 1) // batch_size
    total_evaluated = 0
    
    for batch_idx in range(n_batches):
        batch_start = total_evaluated
        batch_end = min(batch_start + batch_size, n_calls)
        current_batch_size = batch_end - batch_start
        
        if current_batch_size <= 0:
            break
        
        batch_time_start = time.time()
        logger.info(f"\nBatch {batch_idx + 1}/{n_batches}: evaluating {current_batch_size} points...")
        
        # Generate candidate points
        if total_evaluated < n_initial:
            # Random sampling phase
            n_random = min(current_batch_size, n_initial - total_evaluated)
            X_batch = np.random.uniform(
                log_bounds[:, 0], 
                log_bounds[:, 1], 
                size=(n_random, N_DIMS)
            )
            
            # If we need more points beyond initial, use GP
            if n_random < current_batch_size and len(X_observed) >= 20:
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=3
                )
                gp.fit(np.array(X_observed), np.array(y_observed))
                
                X_gp = propose_batch_points(
                    gp, np.max(y_observed), log_bounds, 
                    current_batch_size - n_random
                )
                X_batch = np.vstack([X_batch, X_gp])
        else:
            # Bayesian optimization phase
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=3
            )
            
            X_train = np.array(X_observed)
            y_train = np.array(y_observed)
            
            logger.info(f"  Fitting GP on {len(X_train)} observations...")
            gp.fit(X_train, y_train)
            
            logger.info(f"  Proposing {current_batch_size} candidates...")
            X_batch = propose_batch_points(gp, np.max(y_train), log_bounds, current_batch_size)
        
        # Convert from log scale to real scale
        X_batch_real = np.exp(X_batch)
        
        # Evaluate batch in parallel
        logger.info(f"  Running {current_batch_size} parallel evaluations...")
        y_batch = evaluate_batch_parallel(
            X_batch_real, signal_file, barycenter_file, profile_file,
            metadata_file, variance_mode, n_jobs=current_batch_size
        )
        
        batch_time = time.time() - batch_time_start
        
        # Store results
        for i in range(len(X_batch)):
            X_observed.append(X_batch[i])
            y_observed.append(y_batch[i])
            
            scales = {aa: float(X_batch_real[i, j]) for j, aa in enumerate(AMINO_ACIDS)}
            all_results.append({
                'call': total_evaluated + i + 1,
                'accuracy': y_batch[i],
                **{f'scale_{aa}': scales[aa] for aa in AMINO_ACIDS}
            })
            
            if y_batch[i] > best_accuracy:
                best_accuracy = y_batch[i]
                best_scales = scales.copy()
                logger.info(f"  *** New best: {best_accuracy:.2f}% ***")
        
        total_evaluated += len(X_batch)
        
        # Batch summary
        batch_best = np.max(y_batch)
        batch_mean = np.mean(y_batch)
        logger.info(f"  Batch: best={batch_best:.2f}%, mean={batch_mean:.2f}%, time={batch_time:.1f}s")
        logger.info(f"  Overall best: {best_accuracy:.2f}% | Total evaluated: {total_evaluated}")
        
        # Save intermediate results
        if (batch_idx + 1) % 2 == 0:
            _save_intermediate(best_scales, best_accuracy, all_results, output_path)
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total evaluations: {len(all_results)}")
    logger.info(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Best accuracy: {best_accuracy:.2f}%")
    logger.info("=" * 70)
    
    history_df = pd.DataFrame(all_results)
    
    return best_scales, best_accuracy, history_df


def _save_intermediate(
    best_scales: Dict[str, float],
    best_accuracy: float,
    all_results: List[Dict],
    output_path: Path
) -> None:
    """Save intermediate checkpoint."""
    checkpoint = {
        'best_accuracy': best_accuracy,
        'best_scales': best_scales,
        'n_evaluated': len(all_results)
    }
    checkpoint_path = output_path / "checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def save_results(
    best_scales: Dict[str, float],
    best_accuracy: float,
    history_df: pd.DataFrame,
    output_dir: str,
    args: argparse.Namespace
) -> None:
    """Save all optimization results."""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Optimal scales CSV (use directly with --variance-scale-file)
    scales_df = pd.DataFrame([
        {'amino_acid': aa, 'variance_scale': scale}
        for aa, scale in sorted(best_scales.items())
    ])
    scales_path = output_path / f"optimal_variance_scales_{timestamp}.csv"
    scales_df.to_csv(scales_path, index=False)
    logger.info(f"Saved optimal scales: {scales_path}")
    
    # 2. Full history CSV
    history_path = output_path / f"optimization_history_{timestamp}.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f"Saved history: {history_path}")
    
    # 3. Summary JSON
    summary = {
        'timestamp': timestamp,
        'best_accuracy': best_accuracy,
        'best_scales': best_scales,
        'n_calls': args.n_calls,
        'batch_size': args.batch_size,
        'n_initial': args.n_initial,
        'scale_range': [args.scale_min, args.scale_max],
        'variance_mode': args.variance_mode,
        'profile_file': args.profile_file,
        'barycenter_file': args.barycenter_file,
        'signal_file': args.signal_file,
        'metadata_file': args.metadata_file
    }
    summary_path = output_path / f"optimization_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    # 4. Print final results
    print("\n" + "=" * 60)
    print("OPTIMAL VARIANCE SCALES")
    print("=" * 60)
    for aa, scale in sorted(best_scales.items()):
        print(f"  {aa}: {scale:.4f}")
    print("-" * 60)
    print(f"  Best Accuracy: {best_accuracy:.2f}%")
    print("=" * 60)
    
    # Print usage command
    print(f"\nTo use these optimized scales:")
    print(f"  python -m vrhmm.cli.main \\")
    if args.profile_file:
        print(f"    --profile-file {args.profile_file} \\")
    else:
        print(f"    --barycenter-file {args.barycenter_file} \\")
        print(f"    --variance-mode {args.variance_mode} \\")
    print(f"    --signal-file {args.signal_file} \\")
    if args.metadata_file:
        print(f"    --metadata-file {args.metadata_file} \\")
    print(f"    --variance-scale-file {scales_path} \\")
    print(f"    --output-dir ./results_optimized")


def main():
    parser = argparse.ArgumentParser(
        description='Batch Bayesian optimization for per-AA variance scales',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files (one of profile or barycenter required)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--profile-file', type=str, default=None,
                            help='Path to profile CSV file (from DBA pipeline)')
    input_group.add_argument('--barycenter-file', type=str, default=None,
                            help='Path to barycenter JSON file')
    
    parser.add_argument('--signal-file', type=str, required=True,
                        help='Path to signal data file')
    parser.add_argument('--metadata-file', type=str, default=None,
                        help='Path to metadata JSON file for filtering signals')
    parser.add_argument('--variance-mode', type=str, default='segment',
                        choices=['barycenter', 'segment'],
                        help='Variance calculation mode (only used with barycenter file)')
    
    # Optimization parameters
    parser.add_argument('--n-calls', type=int, default=2500,
                        help='Total optimization evaluations')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Evaluations per batch (set to number of cores)')
    parser.add_argument('--n-initial', type=int, default=256,
                        help='Random evaluations before Bayesian modeling')
    parser.add_argument('--scale-min', type=float, default=0.1,
                        help='Minimum variance scale')
    parser.add_argument('--scale-max', type=float, default=5.0,
                        help='Maximum variance scale')
    parser.add_argument('--output-dir', type=str, default='./optimization_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.profile_file and not args.barycenter_file:
        parser.error("Must provide either --profile-file or --barycenter-file")
    
    # Run optimization
    best_scales, best_accuracy, history_df = batch_bayesian_optimization(
        signal_file=args.signal_file,
        barycenter_file=args.barycenter_file,
        profile_file=args.profile_file,
        metadata_file=args.metadata_file,
        variance_mode=args.variance_mode,
        n_calls=args.n_calls,
        batch_size=args.batch_size,
        n_initial=args.n_initial,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        output_dir=args.output_dir
    )
    
    # Save results
    save_results(best_scales, best_accuracy, history_df, args.output_dir, args)


if __name__ == '__main__':
    main()