"""
Shared utilities for CAAC-FL experiments.
"""

from .models import SimpleCNN, MLP, create_model, count_parameters
from .data_utils import (
    load_cifar10,
    partition_data_iid,
    partition_data_dirichlet,
    partition_data_power_law,
    analyze_data_distribution,
    create_dataloaders
)
from .metrics import (
    evaluate_model,
    train_model,
    compute_gradient_norm,
    compute_model_norm,
    compute_cosine_similarity,
    compute_detection_metrics,
    MetricsLogger
)

__all__ = [
    # Models
    'SimpleCNN',
    'MLP',
    'create_model',
    'count_parameters',
    # Data utilities
    'load_cifar10',
    'partition_data_iid',
    'partition_data_dirichlet',
    'partition_data_power_law',
    'analyze_data_distribution',
    'create_dataloaders',
    # Metrics
    'evaluate_model',
    'train_model',
    'compute_gradient_norm',
    'compute_model_norm',
    'compute_cosine_similarity',
    'compute_detection_metrics',
    'MetricsLogger',
]
