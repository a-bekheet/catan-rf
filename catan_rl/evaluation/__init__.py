"""Evaluation framework for Catan RL agents."""

from .evaluator import (
    EvaluationResult,
    ComparisonResult,
    AgentEvaluator,
    ConfigurationComparator,
    run_comprehensive_evaluation
)

__all__ = [
    'EvaluationResult',
    'ComparisonResult',
    'AgentEvaluator',
    'ConfigurationComparator',
    'run_comprehensive_evaluation'
]