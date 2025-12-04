"""
Krum and Multi-Krum Aggregation Strategies for Flower

Implements Byzantine-robust aggregation using distance-based client selection.

Krum Algorithm:
1. For each client i, compute sum of distances to n-f-2 closest clients
2. Select client with minimum score (most "central" update)
3. Use only the selected client's update

Multi-Krum Algorithm:
1. Same scoring as Krum
2. Select top-k clients with lowest scores
3. Average the selected clients' updates

Detection Capability:
- Selected clients: Considered "trusted" (predicted good)
- Non-selected clients: Considered "suspicious" (predicted bad)
- This allows TP/FP/TN/FN computation when Byzantine IDs are known

References:
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
  Gradient Descent", NeurIPS 2017
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from logging import WARNING

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.logger import log


def _compute_distances(weights_list: List[List[np.ndarray]]) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between client updates.

    Args:
        weights_list: List of client weight updates (each is list of ndarrays)

    Returns:
        n x n distance matrix
    """
    n = len(weights_list)

    # Flatten each client's weights into a single vector
    flat_weights = []
    for weights in weights_list:
        flat = np.concatenate([w.flatten() for w in weights])
        flat_weights.append(flat)

    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flat_weights[i] - flat_weights[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def _krum_scores(distances: np.ndarray, num_to_select: int) -> np.ndarray:
    """
    Compute Krum scores for each client.

    Score(i) = sum of distances to (n - f - 2) closest clients
    where f is the number of Byzantine clients we're defending against.

    Args:
        distances: n x n pairwise distance matrix
        num_to_select: Number of closest neighbors to consider (n - f - 2)

    Returns:
        Array of scores for each client (lower is better)
    """
    n = distances.shape[0]
    scores = np.zeros(n)

    for i in range(n):
        # Get distances from client i to all others
        dists = distances[i].copy()
        dists[i] = np.inf  # Exclude self

        # Sum of distances to num_to_select closest clients
        closest_dists = np.sort(dists)[:num_to_select]
        scores[i] = np.sum(closest_dists)

    return scores


class KrumStrategy(FedAvg):
    """
    Krum aggregation strategy with detection tracking.

    Selects the single client whose update is closest to others,
    making it robust to Byzantine attacks.

    Args:
        num_byzantine: Expected number of Byzantine clients (f)
        byzantine_ids: Known Byzantine client IDs (for detection metrics)
        **kwargs: Arguments passed to FedAvg
    """

    def __init__(
        self,
        num_byzantine: int = 0,
        byzantine_ids: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine
        self.byzantine_ids = set(byzantine_ids) if byzantine_ids else set()
        self.current_round = 0

        # Detection statistics
        self.detection_stats = {
            'total_tp': 0,
            'total_fp': 0,
            'total_tn': 0,
            'total_fn': 0,
            'selected_ids_per_round': [],
            'scores_per_round': [],
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using Krum selection."""

        if not results:
            return None, {}

        self.current_round = server_round

        # Extract weights and client info
        weights_list = []
        client_ids = []
        num_examples = []

        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_list.append(weights)

            # Get client ID from metrics
            cid = fit_res.metrics.get('client_id', len(client_ids))
            client_ids.append(int(cid))
            num_examples.append(fit_res.num_examples)

        n = len(weights_list)

        # Krum requires n >= 2f + 3
        if n < 2 * self.num_byzantine + 3:
            log(WARNING, f"Krum: Not enough clients ({n}) for {self.num_byzantine} Byzantine. Using FedAvg fallback.")
            return super().aggregate_fit(server_round, results, failures)

        # Compute distances and Krum scores
        distances = _compute_distances(weights_list)
        num_to_select = n - self.num_byzantine - 2
        scores = _krum_scores(distances, num_to_select)

        # Select client with minimum score
        selected_idx = int(np.argmin(scores))
        selected_client_id = client_ids[selected_idx]

        # Track detection metrics
        self._update_detection_stats(client_ids, [selected_idx], scores)

        # Use only the selected client's weights
        aggregated_weights = weights_list[selected_idx]

        return ndarrays_to_parameters(aggregated_weights), {}

    def _update_detection_stats(
        self,
        client_ids: List[int],
        selected_indices: List[int],
        scores: np.ndarray
    ):
        """Update detection statistics based on selection."""

        selected_ids = set(client_ids[i] for i in selected_indices)
        all_ids = set(client_ids)
        rejected_ids = all_ids - selected_ids

        # For Krum: selected = predicted good, rejected = predicted bad
        # TP: Byzantine clients that were rejected (correctly identified)
        # FP: Honest clients that were rejected (false alarm)
        # TN: Honest clients that were selected (correctly trusted)
        # FN: Byzantine clients that were selected (missed detection)

        honest_ids = all_ids - self.byzantine_ids

        tp = len(rejected_ids & self.byzantine_ids)
        fp = len(rejected_ids & honest_ids)
        tn = len(selected_ids & honest_ids)
        fn = len(selected_ids & self.byzantine_ids)

        self.detection_stats['total_tp'] += tp
        self.detection_stats['total_fp'] += fp
        self.detection_stats['total_tn'] += tn
        self.detection_stats['total_fn'] += fn
        self.detection_stats['selected_ids_per_round'].append(list(selected_ids))
        self.detection_stats['scores_per_round'].append({
            'round': self.current_round,
            'scores': {cid: float(scores[i]) for i, cid in enumerate(client_ids)},
            'selected': list(selected_ids),
            'rejected': list(rejected_ids),
        })

    def get_detection_stats(self) -> Dict:
        """Return detection statistics."""
        stats = self.detection_stats.copy()

        tp = stats['total_tp']
        fp = stats['total_fp']
        fn = stats['total_fn']

        stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else None
        stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else None

        if stats['precision'] is not None and stats['recall'] is not None:
            p, r = stats['precision'], stats['recall']
            stats['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        else:
            stats['f1'] = None

        return stats

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configure fit with server_round in config."""
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        config = {"server_round": server_round}
        return [(client, FitIns(parameters, config)) for client in clients]


class MultiKrumStrategy(KrumStrategy):
    """
    Multi-Krum aggregation strategy with detection tracking.

    Selects top-k clients whose updates are closest to others,
    then averages their updates.

    Args:
        num_byzantine: Expected number of Byzantine clients (f)
        k: Number of clients to select (default: n - num_byzantine)
        byzantine_ids: Known Byzantine client IDs (for detection metrics)
        **kwargs: Arguments passed to FedAvg
    """

    def __init__(
        self,
        num_byzantine: int = 0,
        k: Optional[int] = None,
        byzantine_ids: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(num_byzantine=num_byzantine, byzantine_ids=byzantine_ids, **kwargs)
        self.k = k  # If None, will be set dynamically

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using Multi-Krum selection."""

        if not results:
            return None, {}

        self.current_round = server_round

        # Extract weights and client info
        weights_list = []
        client_ids = []
        num_examples = []

        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_list.append(weights)

            cid = fit_res.metrics.get('client_id', len(client_ids))
            client_ids.append(int(cid))
            num_examples.append(fit_res.num_examples)

        n = len(weights_list)

        # Determine k (number to select)
        if self.k is not None:
            k = min(self.k, n)
        else:
            # Default: select n - num_byzantine clients
            k = max(1, n - self.num_byzantine)

        # Krum requires n >= 2f + 3
        if n < 2 * self.num_byzantine + 3:
            log(WARNING, f"Multi-Krum: Not enough clients ({n}) for {self.num_byzantine} Byzantine. Using FedAvg fallback.")
            return FedAvg.aggregate_fit(self, server_round, results, failures)

        # Compute distances and Krum scores
        distances = _compute_distances(weights_list)
        num_to_select = n - self.num_byzantine - 2
        scores = _krum_scores(distances, num_to_select)

        # Select top-k clients with lowest scores
        selected_indices = list(np.argsort(scores)[:k])

        # Track detection metrics
        self._update_detection_stats(client_ids, selected_indices, scores)

        # Average the selected clients' weights (weighted by num_examples)
        total_examples = sum(num_examples[i] for i in selected_indices)

        # Initialize aggregated weights
        aggregated_weights = [np.zeros_like(w) for w in weights_list[0]]

        for idx in selected_indices:
            weight_factor = num_examples[idx] / total_examples
            for i, w in enumerate(weights_list[idx]):
                aggregated_weights[i] += w * weight_factor

        return ndarrays_to_parameters(aggregated_weights), {}


# Import FitIns for configure_fit
from flwr.common import FitIns


def create_krum_strategy(
    num_clients: int,
    num_byzantine: int = 0,
    byzantine_ids: Optional[List[int]] = None,
    evaluate_fn: Optional[Callable] = None,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
) -> KrumStrategy:
    """
    Factory function to create a Krum strategy.

    Args:
        num_clients: Total number of clients
        num_byzantine: Expected number of Byzantine clients
        byzantine_ids: Known Byzantine client IDs (for detection metrics)
        evaluate_fn: Server-side evaluation function
        fit_metrics_aggregation_fn: Function to aggregate fit metrics

    Returns:
        Configured KrumStrategy instance
    """
    return KrumStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        num_byzantine=num_byzantine,
        byzantine_ids=byzantine_ids,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )


def create_multi_krum_strategy(
    num_clients: int,
    num_byzantine: int = 0,
    k: Optional[int] = None,
    byzantine_ids: Optional[List[int]] = None,
    evaluate_fn: Optional[Callable] = None,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
) -> MultiKrumStrategy:
    """
    Factory function to create a Multi-Krum strategy.

    Args:
        num_clients: Total number of clients
        num_byzantine: Expected number of Byzantine clients
        k: Number of clients to select (default: num_clients - num_byzantine)
        byzantine_ids: Known Byzantine client IDs (for detection metrics)
        evaluate_fn: Server-side evaluation function
        fit_metrics_aggregation_fn: Function to aggregate fit metrics

    Returns:
        Configured MultiKrumStrategy instance
    """
    return MultiKrumStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        num_byzantine=num_byzantine,
        k=k,
        byzantine_ids=byzantine_ids,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
