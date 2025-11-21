"""
Krum aggregation strategy implementation.

Based on: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent," NeurIPS 2017.

Krum selects the client whose update is closest to other clients' updates,
making it robust to outliers (though may struggle with high heterogeneity).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate


class Krum(Strategy):
    """
    Krum aggregation strategy.

    Selects the client update that minimizes the sum of distances to its
    k nearest neighbors, where k = n - f - 2 (n = total clients, f = Byzantine clients).
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
        evaluate_fn=None,
        initial_parameters: Optional[Parameters] = None,
        num_byzantine: int = 0,
    ):
        """
        Initialize Krum strategy.

        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            evaluate_fn: Centralized evaluation function
            initial_parameters: Initial global model parameters
            num_byzantine: Expected number of Byzantine clients (for k calculation)
        """
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self.num_byzantine = num_byzantine

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create config
        config = {"server_round": server_round}

        # Return client/config pairs
        return [(client, {"parameters": parameters, "config": config}) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using Krum.

        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failures

        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}

        # Convert results to list of parameters
        weights_list = [parameters_to_ndarrays(fit_res.parameters)
                       for _, fit_res in results]

        # Apply Krum selection
        selected_idx = self._krum_selection(weights_list)

        # Use the selected client's weights
        aggregated_weights = weights_list[selected_idx]
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Aggregate custom metrics if available
        metrics_aggregated = {}
        if results:
            # Aggregate training metrics from all clients
            train_losses = [fit_res.metrics.get("train_loss", 0)
                          for _, fit_res in results if fit_res.metrics]
            train_accuracies = [fit_res.metrics.get("train_accuracy", 0)
                              for _, fit_res in results if fit_res.metrics]

            if train_losses:
                metrics_aggregated["train_loss"] = float(np.mean(train_losses))
            if train_accuracies:
                metrics_aggregated["train_accuracy"] = float(np.mean(train_accuracies))

            metrics_aggregated["selected_client_idx"] = int(selected_idx)

        return parameters_aggregated, metrics_aggregated

    def _krum_selection(self, weights_list: List[List[np.ndarray]]) -> int:
        """
        Select the best client using Krum criterion.

        Args:
            weights_list: List of client weight updates

        Returns:
            Index of selected client
        """
        n = len(weights_list)
        f = self.num_byzantine

        # k is the number of nearest neighbors to consider
        # k = n - f - 2 as per Krum paper
        k = max(1, n - f - 2)

        # Flatten weights for distance computation
        flattened_weights = []
        for weights in weights_list:
            flat = np.concatenate([w.flatten() for w in weights])
            flattened_weights.append(flat)

        # Compute pairwise distances
        n_clients = len(flattened_weights)
        distances = np.zeros((n_clients, n_clients))

        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(flattened_weights[i] - flattened_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, compute sum of distances to k nearest neighbors
        scores = []
        for i in range(n_clients):
            # Get distances to all other clients
            dists_to_others = distances[i].copy()
            dists_to_others[i] = np.inf  # Exclude self

            # Get k nearest neighbors
            k_nearest_dists = np.partition(dists_to_others, min(k, n_clients - 1))[:k]

            # Score is sum of distances to k nearest neighbors
            score = np.sum(k_nearest_dists)
            scores.append(score)

        # Select client with minimum score
        selected_idx = int(np.argmin(scores))

        return selected_idx

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create config
        config = {"server_round": server_round}

        # Return client/config pairs
        return [(client, {"parameters": parameters, "config": config}) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Aggregate loss and metrics
        losses = [r.loss for _, r in results]
        metrics = {}

        if losses:
            metrics["eval_loss"] = float(np.mean(losses))

        return float(np.mean(losses)) if losses else None, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model using centralized test set."""
        if self.evaluate_fn is None:
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res
        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for training."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
