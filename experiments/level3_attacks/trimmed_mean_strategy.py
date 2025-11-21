"""
Trimmed Mean Aggregation Strategy for Federated Learning

Implements Byzantine-robust aggregation using trimmed mean (also known as
truncated mean). This method removes the top and bottom β% of values for
each parameter coordinate before averaging.

Reference:
Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates,"
ICML 2018
"""

from typing import List, Tuple, Optional, Dict
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate
import numpy as np


class TrimmedMean(Strategy):
    """
    Trimmed Mean Aggregation Strategy

    For each parameter coordinate, this strategy:
    1. Sorts the values from all clients
    2. Removes the top β% and bottom β% of values
    3. Computes the mean of the remaining values

    This provides Byzantine robustness up to β fraction of malicious clients.

    Args:
        trim_ratio: Fraction of largest/smallest values to trim (β)
                   Default: 0.2 (20%, suitable for 20% Byzantine clients)
        fraction_fit: Fraction of clients to sample for training
        fraction_evaluate: Fraction of clients to sample for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
        evaluate_fn: Optional function for centralized evaluation
    """

    def __init__(
        self,
        trim_ratio: float = 0.2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
    ):
        super().__init__()
        self.trim_ratio = trim_ratio
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn

        # Validate trim ratio
        if not 0 <= trim_ratio < 0.5:
            raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")

    def __repr__(self) -> str:
        return f"TrimmedMean(trim_ratio={self.trim_ratio})"

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training"""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create fit configuration
        config = {"server_round": server_round}

        # Return client/config pairs
        return [(client, config) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation"""
        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create evaluate configuration
        config = {"server_round": server_round}

        # Return client/config pairs
        return [(client, config) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using trimmed mean

        Args:
            server_round: Current round number
            results: Successful client updates
            failures: Failed client updates

        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}

        # Convert results to weights
        weights_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # Get client sample sizes for metrics
        num_examples_list = [fit_res.num_examples for _, fit_res in results]

        # Apply trimmed mean aggregation
        aggregated_weights = self._trimmed_mean_aggregate(weights_list)

        # Aggregate custom metrics if any
        metrics_aggregated = {}
        if results:
            # Example: aggregate loss
            losses = [fit_res.metrics.get("loss", 0) for _, fit_res in results]
            if losses:
                metrics_aggregated["train_loss"] = np.mean(losses)

        return ndarrays_to_parameters(aggregated_weights), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}

        # Aggregate accuracy
        accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        losses = [r.loss for _, r in results]

        metrics_aggregated = {
            "accuracy": np.mean(accuracies) if accuracies else 0,
            "loss": np.mean(losses) if losses else 0,
        }

        return metrics_aggregated.get("loss"), metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using the evaluate_fn"""
        if self.evaluate_fn is None:
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res
        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for training"""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_fit_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation"""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_evaluate_clients

    def _trimmed_mean_aggregate(
        self, weights_list: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Compute trimmed mean across client updates

        For each parameter coordinate:
        1. Collect values from all clients
        2. Sort the values
        3. Remove top and bottom β% (trim_ratio)
        4. Compute mean of remaining values

        Args:
            weights_list: List of weight arrays from each client

        Returns:
            Aggregated weights using trimmed mean
        """
        num_clients = len(weights_list)

        # Calculate how many values to trim from each end
        num_trim = int(np.floor(num_clients * self.trim_ratio))

        # Ensure we don't trim everything
        if num_trim * 2 >= num_clients:
            # Fall back to regular mean if trimming would remove too many
            num_trim = max(0, (num_clients - 1) // 2)

        # Initialize aggregated weights with same structure
        aggregated = []

        # Process each layer
        for layer_idx in range(len(weights_list[0])):
            # Stack weights from all clients for this layer: shape (num_clients, ...)
            layer_weights = np.stack([w[layer_idx] for w in weights_list], axis=0)

            if num_trim == 0:
                # No trimming needed, just compute mean
                trimmed_mean = np.mean(layer_weights, axis=0)
            else:
                # Sort along client axis (axis=0)
                sorted_weights = np.sort(layer_weights, axis=0)

                # Remove top and bottom num_trim values
                trimmed_weights = sorted_weights[num_trim:-num_trim, ...]

                # Compute mean of trimmed values
                trimmed_mean = np.mean(trimmed_weights, axis=0)

            aggregated.append(trimmed_mean)

        return aggregated


# Example usage and testing
if __name__ == "__main__":
    print("Trimmed Mean Aggregation Strategy")
    print("=" * 50)

    # Create sample weights to demonstrate trimming
    print("\nDemo: Trimming with 5 clients, β=0.2 (trim 1 from each end)")
    print("-" * 50)

    # Create sample weight matrices (simplified to 1D for clarity)
    num_clients = 5
    weights_list = [
        [np.array([1.0, 2.0, 3.0])],  # Client 0 (will be trimmed)
        [np.array([5.0, 5.0, 5.0])],  # Client 1
        [np.array([5.0, 5.0, 5.0])],  # Client 2
        [np.array([5.0, 5.0, 5.0])],  # Client 3
        [np.array([9.0, 8.0, 7.0])],  # Client 4 (will be trimmed)
    ]

    print("\nClient weights (first parameter only):")
    for i, w in enumerate(weights_list):
        print(f"  Client {i}: {w[0]}")

    # Create strategy
    strategy = TrimmedMean(trim_ratio=0.2)

    # Aggregate
    aggregated = strategy._trimmed_mean_aggregate(weights_list)

    print(f"\nRegular mean:  {np.mean([w[0] for w in weights_list], axis=0)}")
    print(f"Trimmed mean:  {aggregated[0]}")
    print(f"Expected:      [5. 5. 5.] (after removing clients 0 and 4)")

    print("\n" + "=" * 50)
    print("Strategy initialized successfully!")
    print(f"Trim ratio: {strategy.trim_ratio}")
    print(f"Fraction fit: {strategy.fraction_fit}")
