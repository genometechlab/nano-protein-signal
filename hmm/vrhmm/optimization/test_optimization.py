#!/usr/bin/env python
"""
Test script for the multi-objective optimization system.

Verifies that all components work correctly before running full optimization.

Usage:
    python test_optimization.py \
        --profile-file data/profiles.csv \
        --signal-file data/signals.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loading(profile_file: str, signal_file: str, metadata_file: str = None):
    """Test data loading functionality."""
    logger.info("=" * 50)
    logger.info("Testing Data Loading")
    logger.info("=" * 50)
    
    # Import with fallback
    try:
        from vrhmm.optimization.data_loader import OptimizationDataLoader
    except ImportError:
        from data_loader import OptimizationDataLoader
    
    loader = OptimizationDataLoader(
        profile_file=profile_file,
        signal_file=signal_file,
        metadata_file=metadata_file
    )
    
    # Test profile loading
    profiles = loader.load_profiles()
    logger.info(f"✓ Loaded profiles for {len(profiles)} amino acids")
    
    for aa, profile in list(profiles.items())[:3]:
        logger.info(f"  {aa}: {len(profile)} states, "
                   f"mean range [{min(p[0] for p in profile.values()):.2f}, "
                   f"{max(p[0] for p in profile.values()):.2f}]")
    
    # Test trace loading
    traces = loader.load_traces()
    total_traces = sum(len(t) for t in traces.values())
    logger.info(f"✓ Loaded {total_traces} traces for {len(traces)} amino acids")
    
    for aa, aa_traces in list(traces.items())[:3]:
        if aa_traces:
            logger.info(f"  {aa}: {len(aa_traces)} traces, "
                       f"mean length {np.mean([len(t) for t in aa_traces]):.1f}")
    
    return loader, profiles, traces


def test_model_building(profiles: dict):
    """Test model building functionality."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Model Building")
    logger.info("=" * 50)
    
    try:
        from vrhmm.optimization.model_builder import OptimizationModelBuilder
    except ImportError:
        from model_builder import OptimizationModelBuilder
    
    builder = OptimizationModelBuilder(profiles)
    
    # Test building a few models
    test_aas = list(profiles.keys())[:3]
    
    for aa in test_aas:
        model = builder.build_model(
            aa=aa,
            variance_scale=1.0,
            transitions={
                'match_self_loop': 0.05,
                'forward': 0.65,
                'to_skip': 0.15,
                'to_slip': 0.07,
                'to_insert': 0.03,
                'to_end': 0.05
            }
        )
        
        n_states = len([s for s in model.states if hasattr(s, 'name') and s.name and 'Match' in s.name])
        logger.info(f"✓ Built model for {aa}: {n_states} match states")
    
    return builder


def test_objective_computation(builder, profiles: dict, traces: dict):
    """Test objective function computation."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Objective Computation")
    logger.info("=" * 50)
    
    try:
        from vrhmm.optimization.objective import (
            MultiObjectiveTrainer,
            TrainingConfig,
            analyze_viterbi_path
        )
    except ImportError:
        from objective import (
            MultiObjectiveTrainer,
            TrainingConfig,
            analyze_viterbi_path
        )
    
    trainer = MultiObjectiveTrainer(
        model_builder=builder,
        profile_stats=profiles
    )
    
    # Test with default config
    config = TrainingConfig(alpha=1.0, beta=0.5, gamma=0.3)
    
    # Use a subset of AAs for testing
    test_aas = [aa for aa in profiles.keys() if aa in traces and traces[aa]][:5]
    
    variance_scales = {aa: 1.0 for aa in test_aas}
    transitions = {aa: {
        'match_self_loop': 0.05,
        'forward': 0.65,
        'to_skip': 0.15,
        'to_slip': 0.07,
        'to_insert': 0.03,
        'to_end': 0.05
    } for aa in test_aas}
    
    test_traces = {aa: traces[aa][:2] for aa in test_aas if traces.get(aa)}
    
    score, metrics = trainer.compute_objective(
        variance_scales=variance_scales,
        transitions=transitions,
        test_traces=test_traces,
        config=config
    )
    
    logger.info(f"✓ Computed objective score: {score:.4f}")
    logger.info(f"  Total LL: {metrics['total_ll']:.2f}")
    logger.info(f"  Mean coverage: {metrics['mean_coverage']:.3f}")
    logger.info(f"  Mean smoothness: {metrics['mean_smoothness']:.3f}")
    logger.info(f"  Mean efficiency: {metrics['mean_efficiency']:.3f}")
    
    # Test path analysis
    logger.info("\nTesting path analysis:")
    
    for aa in test_aas[:2]:
        if aa not in test_traces or not test_traces[aa]:
            continue
        
        model = builder.build_model(aa, 1.0, transitions[aa])
        trace = test_traces[aa][0]
        
        _, path_raw = model.viterbi(trace)
        
        path = []
        for item in path_raw:
            if hasattr(item, 'name'):
                path.append(item.name)
            elif isinstance(item, tuple) and len(item) >= 2:
                if hasattr(item[1], 'name'):
                    path.append(item[1].name)
        
        n_expected = len(profiles[aa])
        path_metrics = analyze_viterbi_path(path, n_expected)
        
        logger.info(f"  {aa}: coverage={path_metrics.coverage:.3f}, "
                   f"smoothness={path_metrics.smoothness:.3f}, "
                   f"skips={path_metrics.n_skips}, slips={path_metrics.n_slips}")
    
    return trainer


def test_coverage_utilities(builder, profiles: dict, traces: dict):
    """Test coverage utility functions."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Coverage Utilities")
    logger.info("=" * 50)
    
    try:
        from vrhmm.optimization.coverage_utils import (
            compute_match_state_distribution,
            compute_path_transition_stats,
            rank_models_by_fit
        )
    except ImportError:
        from coverage_utils import (
            compute_match_state_distribution,
            compute_path_transition_stats,
            rank_models_by_fit
        )
    
    # Build models for ALL amino acids that have both profiles and traces
    models = {}
    model_lengths = {}
    
    available_aas = [aa for aa in profiles.keys() if aa in traces and traces[aa]]
    
    for aa in available_aas:
        models[aa] = builder.build_model(aa, 1.0)
        model_lengths[aa] = len(profiles[aa])
    
    logger.info(f"Built models for {len(models)} amino acids")
    
    # Get a test trace from an AA we have a model for
    test_aa = None
    test_trace = None
    
    for aa in available_aas:
        if traces.get(aa):
            test_aa = aa
            test_trace = traces[aa][0]
            break
    
    if test_trace is None:
        logger.warning("No test traces available")
        return
    
    # Test ranking
    rankings = rank_models_by_fit(
        test_trace, models, model_lengths, coverage_weight=0.5
    )
    
    logger.info(f"✓ Ranked {len(rankings)} models for trace (true AA: {test_aa})")
    logger.info("  Top 5 predictions:")
    
    for i, (aa, score, details) in enumerate(rankings[:5], 1):
        marker = "✓" if aa == test_aa else " "
        logger.info(f"    {marker} {i}. {aa}: score={score:.2f}, "
                   f"coverage={details.get('coverage', 0):.3f}, "
                   f"log_prob={details.get('log_prob', float('-inf')):.2f}")
    
    # Find where true AA ranked
    true_rank = next((i for i, (aa, _, _) in enumerate(rankings, 1) if aa == test_aa), None)
    if true_rank:
        logger.info(f"  True AA '{test_aa}' ranked #{true_rank}")
    
    # Test distribution analysis using the true AA's model
    model = models[test_aa]
    _, path_raw = model.viterbi(test_trace)
    
    path = []
    for item in path_raw:
        if hasattr(item, 'name'):
            path.append(item.name)
        elif isinstance(item, tuple) and len(item) >= 2:
            if hasattr(item[1], 'name'):
                path.append(item[1].name)
    
    distribution = compute_match_state_distribution(path, model_lengths[test_aa])
    
    logger.info(f"\n✓ Match state distribution for {test_aa}:")
    logger.info(f"  Visited: {distribution['n_visited']}/{distribution['n_expected']}")
    logger.info(f"  Coverage: {distribution['coverage']:.3f}")
    if distribution['gaps']:
        logger.info(f"  Gaps: {distribution['gaps'][:5]}{'...' if len(distribution['gaps']) > 5 else ''}")
    else:
        logger.info(f"  Gaps: None (full coverage)")
    
    # Test transition stats
    trans_stats = compute_path_transition_stats(path)
    
    logger.info(f"\n✓ Transition statistics:")
    logger.info(f"  Forward (normal): {trans_stats['forward_1']}")
    logger.info(f"  Skips: {trans_stats['forward_skip']} (sizes: {trans_stats['skip_sizes'][:5] if trans_stats['skip_sizes'] else 'none'})")
    logger.info(f"  Slips: {trans_stats['backward']} (sizes: {trans_stats['slip_sizes'][:5] if trans_stats['slip_sizes'] else 'none'})")
    logger.info(f"  Self-loops: {trans_stats['self_loop']}")
    logger.info(f"  Inserts: {trans_stats['total_insert']}")


def test_full_optimization_step(profiles: dict, traces: dict, profile_file: str, signal_file: str, metadata_file: str = None):
    """Test a single optimization step."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Full Optimization Step")
    logger.info("=" * 50)
    
    # Import with fallback for different run contexts
    try:
        from vrhmm.optimization.batch_joint_optimization import JointBayesianOptimizer
    except ImportError:
        from batch_joint_optimization import JointBayesianOptimizer
    
    # Create optimizer with minimal settings
    optimizer = JointBayesianOptimizer(
        profile_file=profile_file,
        signal_file=signal_file,
        metadata_file=metadata_file,
        per_aa_transitions=False,
        optimize_weights=True,
        n_jobs=1
    )
    
    # Test single evaluation
    test_x = np.zeros(optimizer.n_dims)
    
    # Set reasonable values
    for i in range(optimizer.n_variance):
        test_x[i] = np.log(1.0)  # variance scale = 1.0
    
    for i in range(optimizer.n_transitions):
        test_x[optimizer.n_variance + i] = np.log(0.1)  # transitions
    
    if optimizer.optimize_weights:
        idx = optimizer.n_variance + optimizer.n_transitions
        test_x[idx] = np.log(1.0)      # alpha
        test_x[idx + 1] = np.log(0.5)  # beta
        test_x[idx + 2] = np.log(0.3)  # gamma
    
    logger.info("Testing single evaluation...")
    score, metrics = optimizer.evaluate_single(test_x)
    
    logger.info(f"✓ Single evaluation completed")
    logger.info(f"  Score: {score:.4f}")
    logger.info(f"  Coverage: {metrics['mean_coverage']:.3f}")
    logger.info(f"  Smoothness: {metrics['mean_smoothness']:.3f}")
    logger.info(f"  Traces evaluated: {metrics['n_traces']}")
    
    # Test vector conversion (note: transitions get normalized, so we check params not raw vectors)
    variance_scales, transitions, config = optimizer.vector_to_params(test_x)
    
    # Verify variance scales roundtrip (these should be exact)
    variance_ok = all(
        abs(variance_scales[aa] - 1.0) < 1e-6 
        for aa in variance_scales
    )
    
    # Verify config roundtrip
    config_ok = (
        abs(config.alpha - 1.0) < 1e-6 and
        abs(config.beta - 0.5) < 1e-6 and
        abs(config.gamma - 0.3) < 1e-6
    )
    
    # Verify transitions sum to ~1 after normalization
    first_aa = list(transitions.keys())[0]
    trans_sum = sum(
        transitions[first_aa].get(k, 0) 
        for k in ['match_self_loop', 'forward', 'to_skip', 'to_slip', 'to_insert', 'to_end']
    )
    trans_ok = abs(trans_sum - 1.0) < 1e-6
    
    if variance_ok and config_ok and trans_ok:
        logger.info("✓ Parameter conversion roundtrip passed")
        logger.info(f"  Variance scales: all = 1.0 ✓")
        logger.info(f"  Transitions sum: {trans_sum:.6f} ✓")
        logger.info(f"  Config: α={config.alpha}, β={config.beta}, γ={config.gamma} ✓")
    else:
        logger.error("✗ Parameter conversion roundtrip FAILED")
        if not variance_ok:
            logger.error(f"  Variance scales incorrect")
        if not trans_ok:
            logger.error(f"  Transitions sum = {trans_sum}, expected 1.0")
        if not config_ok:
            logger.error(f"  Config incorrect: α={config.alpha}, β={config.beta}, γ={config.gamma}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test optimization components')
    parser.add_argument('--profile-file', type=str, required=True)
    parser.add_argument('--signal-file', type=str, required=True)
    parser.add_argument('--metadata-file', type=str, default=None)
    
    args = parser.parse_args()
    
    logger.info("Starting optimization component tests\n")
    
    try:
        # Test 1: Data loading
        loader, profiles, traces = test_data_loading(
            args.profile_file, args.signal_file, args.metadata_file
        )
        
        # Test 2: Model building
        builder = test_model_building(profiles)
        
        # Test 3: Objective computation
        trainer = test_objective_computation(builder, profiles, traces)
        
        # Test 4: Coverage utilities
        test_coverage_utilities(builder, profiles, traces)
        
        # Test 5: Full optimization step
        success = test_full_optimization_step(
            profiles, traces, args.profile_file, args.signal_file, args.metadata_file
        )
        
        logger.info("\n" + "=" * 50)
        if success:
            logger.info("ALL TESTS PASSED ✓")
        else:
            logger.info("SOME TESTS FAILED ✗")
        logger.info("=" * 50)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())