"""
Optimization module for vrHMM.

Provides multi-objective Bayesian optimization for HMM parameters.
"""

from vrhmm.optimization.objective import (
    TrainingConfig,
    PathMetrics,
    MultiObjectiveTrainer,
    analyze_viterbi_path,
    compute_coverage_score
)

from vrhmm.optimization.model_builder import OptimizationModelBuilder

from vrhmm.optimization.data_loader import OptimizationDataLoader

__all__ = [
    'TrainingConfig',
    'PathMetrics',
    'MultiObjectiveTrainer',
    'analyze_viterbi_path',
    'compute_coverage_score',
    'OptimizationModelBuilder',
    'OptimizationDataLoader'
]