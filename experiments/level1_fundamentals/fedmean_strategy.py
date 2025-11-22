"""
FedMean aggregation strategy.

Simple unweighted averaging of client models (as opposed to FedAvg's weighted averaging).
"""

from typing import List, Tuple, Optional, Dict, Union
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
from functools import reduce
import numpy as np


class FedMean(FedAvg):
    """
    FedMean: Unweighted averaging aggregation strategy.

    Unlike FedAvg which weights updates by number of samples,
    FedMean gives equal weight to all clients regardless of data size.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[FitRes, int]],
        failures: List[Union[Tuple[FitRes, int], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, any]]:
        """
        Aggregate fit results using simple unweighted averaging.

        Args:
            server_round: Current round number
            results: List of (FitRes, num_examples) tuples
            failures: List of failed clients

        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Extract parameters from results (ignore num_examples for unweighted mean)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client_proxy, fit_res in results
        ]

        # Perform unweighted averaging
        parameters_aggregated = self._aggregate_unweighted(
            [weights for weights, _ in weights_results]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(parameters_aggregated), metrics_aggregated

    def _aggregate_unweighted(self, results: List[NDArrays]) -> NDArrays:
        """
        Compute unweighted average (simple mean).

        Args:
            results: List of parameter arrays from clients

        Returns:
            Averaged parameters
        """
        # Create a list of weights, each being a list of layer weights
        num_clients = len(results)

        # Average each layer across all clients
        aggregated_weights = []
        num_layers = len(results[0])

        for layer_idx in range(num_layers):
            # Stack all client weights for this layer
            layer_weights = np.array([client_weights[layer_idx] for client_weights in results])
            # Compute simple mean (unweighted average)
            avg_layer = np.mean(layer_weights, axis=0)
            aggregated_weights.append(avg_layer)

        return aggregated_weights
