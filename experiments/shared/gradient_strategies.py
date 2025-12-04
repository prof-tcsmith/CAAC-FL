"""
Gradient-based aggregation strategies for Federated Learning.

This module provides gradient-based alternatives to weight-sharing approaches:

Weight Sharing (existing):
- FedAvg: Weighted average of model weights
- FedMean: Unweighted average of model weights
- FedMedian: Coordinate-wise median of weights

Gradient Sharing (this module):
- FedSGD: Single gradient step (local_epochs=1, full batch)
- FedAdam: Server-side Adam optimizer on pseudo-gradients
- FedTrimmedAvg: Trimmed mean of updates (Byzantine-robust)
- FedMedianGrad: Coordinate-wise median (re-export for naming consistency)

Note: In Flower, clients send "model updates" (trained_weights - initial_weights),
which are mathematically equivalent to pseudo-gradients accumulated over local training.
"""

from flwr.server.strategy import (
    FedAvg,
    FedAdam,
    FedAdagrad,
    FedTrimmedAvg,
    FedMedian,
    FedYogi,
)
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np


# Re-export Flower's built-in strategies with consistent naming
FedTrimmedMean = FedTrimmedAvg  # Alias for clarity


class FedSGD(FedAvg):
    """
    Federated Stochastic Gradient Descent.

    This is FedAvg with the constraint that clients perform exactly one
    gradient step (local_epochs=1, ideally with full batch). This makes
    client updates true gradients rather than accumulated pseudo-gradients.

    In practice, we enforce this by:
    1. Setting local_epochs=1 in the client
    2. Using larger batch sizes (ideally full batch)

    The aggregation is identical to FedAvg (weighted average).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: local_epochs is controlled client-side, not here
        # This class serves as documentation and for result organization


class FedAdamW(FedAdam):
    """
    FedAdam with decoupled weight decay (AdamW-style).

    Extends FedAdam to apply weight decay separately from the adaptive
    learning rate, which can improve generalization.

    Note: This is a simple extension - Flower's FedAdam doesn't have
    built-in weight decay, so we'd need to implement it if needed.
    """
    pass  # Placeholder for future implementation


def create_strategy(
    strategy_name: str,
    initial_parameters: Parameters,
    evaluate_fn: Callable,
    fit_metrics_aggregation_fn: Callable,
    num_clients: int,
    **kwargs
) -> Union[FedAvg, FedAdam, FedTrimmedAvg, FedMedian]:
    """
    Factory function to create aggregation strategies.

    Args:
        strategy_name: One of 'fedavg', 'fedmean', 'fedmedian', 'fedsgd',
                      'fedadam', 'fedtrimmed', 'fedyogi', 'fedadagrad'
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        fit_metrics_aggregation_fn: Function to aggregate client metrics
        num_clients: Number of clients (for min_clients settings)
        **kwargs: Additional strategy-specific arguments

    Returns:
        Configured Flower strategy instance
    """

    # Common kwargs for all strategies
    common_kwargs = {
        'fraction_fit': 1.0,
        'fraction_evaluate': 0.0,
        'min_fit_clients': num_clients,
        'min_evaluate_clients': 0,
        'min_available_clients': num_clients,
        'evaluate_fn': evaluate_fn,
        'fit_metrics_aggregation_fn': fit_metrics_aggregation_fn,
        'initial_parameters': initial_parameters,
    }

    strategy_name = strategy_name.lower()

    if strategy_name == 'fedavg':
        return FedAvg(**common_kwargs)

    elif strategy_name == 'fedmean':
        # Import from existing module
        from level1_fundamentals.fedmean_strategy import FedMean
        return FedMean(**common_kwargs)

    elif strategy_name == 'fedmedian':
        return FedMedian(**common_kwargs)

    elif strategy_name == 'fedsgd':
        # FedSGD is FedAvg - the difference is in client config (local_epochs=1)
        return FedSGD(**common_kwargs)

    elif strategy_name == 'fedadam':
        # Server-side Adam optimizer
        adam_kwargs = {
            'eta': kwargs.get('eta', 0.1),  # Server learning rate
            'eta_l': kwargs.get('eta_l', 0.01),  # Client learning rate
            'beta_1': kwargs.get('beta_1', 0.9),
            'beta_2': kwargs.get('beta_2', 0.99),
            'tau': kwargs.get('tau', 1e-9),
        }
        return FedAdam(**common_kwargs, **adam_kwargs)

    elif strategy_name in ['fedtrimmed', 'fedtrimmedavg', 'fedtrimmedmean']:
        # Trimmed mean - removes beta fraction from each tail
        beta = kwargs.get('beta', 0.2)  # Trim 20% from each tail by default
        return FedTrimmedAvg(**common_kwargs, beta=beta)

    elif strategy_name == 'fedyogi':
        # Yogi optimizer (variant of Adam)
        yogi_kwargs = {
            'eta': kwargs.get('eta', 0.1),
            'eta_l': kwargs.get('eta_l', 0.01),
            'beta_1': kwargs.get('beta_1', 0.9),
            'beta_2': kwargs.get('beta_2', 0.99),
            'tau': kwargs.get('tau', 1e-3),
        }
        return FedYogi(**common_kwargs, **yogi_kwargs)

    elif strategy_name == 'fedadagrad':
        # Adagrad optimizer
        adagrad_kwargs = {
            'eta': kwargs.get('eta', 0.1),
            'eta_l': kwargs.get('eta_l', 0.01),
            'tau': kwargs.get('tau', 1e-9),
        }
        return FedAdagrad(**common_kwargs, **adagrad_kwargs)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Supported: fedavg, fedmean, fedmedian, fedsgd, "
                        f"fedadam, fedtrimmed, fedyogi, fedadagrad")


# Strategy metadata for experiment organization
WEIGHT_SHARING_STRATEGIES = ['fedavg', 'fedmean', 'fedmedian']
GRADIENT_SHARING_STRATEGIES = ['fedsgd', 'fedadam', 'fedtrimmed', 'fedyogi', 'fedadagrad']
ALL_STRATEGIES = WEIGHT_SHARING_STRATEGIES + GRADIENT_SHARING_STRATEGIES

STRATEGY_INFO = {
    'fedavg': {
        'type': 'weight',
        'description': 'Weighted average of model weights',
        'byzantine_robust': False,
    },
    'fedmean': {
        'type': 'weight',
        'description': 'Unweighted average of model weights',
        'byzantine_robust': False,
    },
    'fedmedian': {
        'type': 'weight',
        'description': 'Coordinate-wise median of weights',
        'byzantine_robust': True,
    },
    'fedsgd': {
        'type': 'gradient',
        'description': 'Single gradient step per round',
        'byzantine_robust': False,
    },
    'fedadam': {
        'type': 'gradient',
        'description': 'Server-side Adam optimizer',
        'byzantine_robust': False,
    },
    'fedtrimmed': {
        'type': 'gradient',
        'description': 'Trimmed mean (removes extreme updates)',
        'byzantine_robust': True,
    },
    'fedyogi': {
        'type': 'gradient',
        'description': 'Server-side Yogi optimizer',
        'byzantine_robust': False,
    },
    'fedadagrad': {
        'type': 'gradient',
        'description': 'Server-side Adagrad optimizer',
        'byzantine_robust': False,
    },
}


if __name__ == "__main__":
    print("Gradient-based Aggregation Strategies")
    print("=" * 50)

    print("\nWeight Sharing:")
    for s in WEIGHT_SHARING_STRATEGIES:
        info = STRATEGY_INFO[s]
        robust = "✓" if info['byzantine_robust'] else "✗"
        print(f"  {s:12s}: {info['description']} [Byzantine: {robust}]")

    print("\nGradient Sharing:")
    for s in GRADIENT_SHARING_STRATEGIES:
        info = STRATEGY_INFO[s]
        robust = "✓" if info['byzantine_robust'] else "✗"
        print(f"  {s:12s}: {info['description']} [Byzantine: {robust}]")
