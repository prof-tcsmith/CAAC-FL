"""
CAAC-FL Strategy for Flower Framework

This integrates the full-featured CAAC-FL aggregation algorithm from caacfl.py
into Flower's Strategy interface for use with fl.simulation.start_simulation().

================================================================================
WHY WE USE PSEUDO-GRADIENTS (Server-Side Computation)
================================================================================

Flower clients return updated WEIGHTS after local training, not gradients.
This is Flower's standard protocol used by all built-in strategies (FedAvg,
FedMedian, FedTrimmedAvg, etc.).

We compute pseudo-gradients on the server as:
    gradient = client_weights - global_weights

ALTERNATIVE APPROACH (Not Used):
We could modify clients to send gradients directly by:
    1. Storing model weights before training
    2. Computing gradient = weights_after - weights_before
    3. Sending gradient instead of weights

REASONS FOR CURRENT APPROACH (Server-Side Pseudo-Gradients):
1. **Compatibility**: Maintains compatibility with Flower's standard strategies
   (FedAvg, FedMedian, FedTrimmedAvg) for fair comparison in experiments.
   All strategies use the same client code and communication protocol.

2. **Attack Consistency**: Byzantine attack implementations (sign_flipping,
   random_noise, ALIE) are designed to manipulate the difference between
   the original and updated model. Computing pseudo-gradients on the server
   captures this attack surface accurately.

3. **Reduced Complexity**: No need for separate client implementations for
   CAAC-FL vs other strategies. Single client works with all aggregation methods.

TRADE-OFFS AND POTENTIAL NEGATIVE IMPACTS:
1. **Memory Overhead**: Server must store global_weights to compute pseudo-gradients.
   For large models, this adds memory usage on the server side.

2. **Computation Location**: Gradient computation happens on the server instead
   of being distributed across clients. However, this is a simple subtraction
   operation (O(n) where n = number of parameters), so the overhead is minimal.

3. **Semantic Difference**: Pseudo-gradients represent the total weight change
   from local training, which may differ slightly from accumulated SGD gradients
   due to momentum, weight decay, etc. However, for Byzantine detection purposes,
   what matters is the direction and magnitude of weight updates, which pseudo-
   gradients capture accurately.

4. **First Round Limitation**: No pseudo-gradient can be computed on the first
   round (no prior global weights exist), so we fall back to FedAvg.

NOTE: If native gradient support is needed in the future, modify the FlowerClient
class in client.py to send gradients, and create a gradient-aware version of
standard strategies for fair comparison.

================================================================================

The CAACFLAggregator provides:
1. EWMA-based per-client behavioral profiles
2. Three-dimensional anomaly detection (magnitude, directional, temporal)
3. Cold-start mitigations (warmup, cross-client comparison, population init)
4. Adaptive thresholds based on client reliability
5. Soft gradient clipping
"""

from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

# Import the full-featured CAAC-FL aggregator
from caacfl import CAACFLAggregator


class CAACFLStrategy(Strategy):
    """
    Client-Adaptive Anomaly-Aware Clipping (CAAC-FL) Strategy for Flower

    This strategy wraps the full CAACFLAggregator and properly handles
    Flower's weight-based update model by computing pseudo-gradients.

    The key insight is that Flower clients return updated weights, not gradients.
    We compute: gradient = new_weights - old_global_weights

    Parameters:
        num_clients: Total number of FL clients

        CAAC-FL Algorithm Parameters (passed to CAACFLAggregator):
            alpha: EWMA smoothing factor (default 0.1)
            gamma: Reliability update rate (default 0.1)
            tau_base: Base anomaly threshold (default 2.0)
            beta: Threshold flexibility factor (default 0.5)
            window_size: History window for anomaly detection (default 5)
            weights: Anomaly dimension weights (w_mag, w_dir, w_temp)

        Cold-Start Mitigation Parameters:
            warmup_rounds: Rounds with conservative thresholds (default 5)
            warmup_factor: Threshold multiplier during warmup (default 0.5)
            min_rounds_for_trust: Rounds before reliability bonus (default 3)
            use_cross_comparison: Enable cross-client comparison (default True)
            use_population_init: Initialize from population stats (default True)
            new_client_weight: Weight multiplier for new clients (default 0.5)

        Flower Parameters:
            fraction_fit: Fraction of clients for training (default 1.0)
            fraction_evaluate: Fraction of clients for evaluation (default 1.0)
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients to start
            evaluate_fn: Centralized evaluation function
            fit_metrics_aggregation_fn: Metrics aggregation function
    """

    def __init__(
        self,
        num_clients: int,
        # CAAC-FL parameters
        alpha: float = 0.1,
        gamma: float = 0.1,
        tau_base: float = 2.0,
        beta: float = 0.5,
        window_size: int = 5,
        weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
        # Cold-start mitigation parameters
        warmup_rounds: int = 5,
        warmup_factor: float = 0.5,
        min_rounds_for_trust: int = 3,
        use_cross_comparison: bool = True,
        use_population_init: bool = True,
        new_client_weight: float = 0.5,
        # Byzantine client tracking for TP/FP/TN/FN metrics
        byzantine_ids: Optional[List[int]] = None,
        # Flower parameters
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        fit_metrics_aggregation_fn=None,
    ):
        super().__init__()

        # Initialize the full-featured CAAC-FL aggregator
        self.aggregator = CAACFLAggregator(
            num_clients=num_clients,
            alpha=alpha,
            gamma=gamma,
            tau_base=tau_base,
            beta=beta,
            window_size=window_size,
            weights=weights,
            warmup_rounds=warmup_rounds,
            warmup_factor=warmup_factor,
            min_rounds_for_trust=min_rounds_for_trust,
            use_cross_comparison=use_cross_comparison,
            use_population_init=use_population_init,
            new_client_weight=new_client_weight,
        )

        # Flower parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn

        # Store current global weights for computing gradients
        self.global_weights = None

        # Round tracking
        self.current_round = 0
        self.num_clients = num_clients

        # Byzantine client tracking for confusion matrix computation
        # Byzantine IDs should be in the range [0, num_clients-1]
        self.byzantine_ids = set(byzantine_ids) if byzantine_ids else set()
        self.honest_ids = set(range(num_clients)) - self.byzantine_ids

        # Detection statistics with confusion matrix tracking
        self.detection_stats = {
            'total_detections': 0,
            'detections_per_round': [],
            'mean_clip_factors': [],
            # Confusion matrix metrics (cumulative)
            'total_tp': 0,  # True Positive: Byzantine client correctly detected
            'total_fp': 0,  # False Positive: Honest client incorrectly detected
            'total_tn': 0,  # True Negative: Honest client correctly not detected
            'total_fn': 0,  # False Negative: Byzantine client not detected
            # Per-round confusion matrix
            'confusion_per_round': [],  # List of {'tp': x, 'fp': y, 'tn': z, 'fn': w}
            # Per-round detected client IDs
            'detected_ids_per_round': [],
        }

    def __repr__(self) -> str:
        return (f"CAACFLStrategy(tau_base={self.aggregator.tau_base}, "
                f"warmup={self.aggregator.warmup_rounds})")

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training"""
        from flwr.common import FitIns

        self.current_round = server_round

        # Store global weights for gradient computation
        self.global_weights = parameters_to_ndarrays(parameters)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        config = {"server_round": server_round}
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation"""
        from flwr.common import EvaluateIns

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using the full CAAC-FL algorithm.

        Key steps:
        1. Compute pseudo-gradients: gradient = client_weights - global_weights
        2. Pass gradients to CAACFLAggregator for anomaly detection and clipping
        3. Apply the aggregated gradient to get new global weights

        PSEUDO-GRADIENT COMPUTATION:
        Flower clients send updated weights, not gradients. The CAACFLAggregator
        requires gradients for its anomaly detection (magnitude, direction, temporal
        analysis). We compute pseudo-gradients as:

            gradient_i = weights_i^{t+1} - weights^t

        This represents the total weight change from client i's local training.
        While not identical to accumulated SGD gradients (due to momentum, weight
        decay, etc.), pseudo-gradients accurately capture the update direction
        and magnitude that matter for Byzantine detection.

        See module docstring for detailed rationale and trade-offs.
        """
        if not results:
            return None, {}

        self.current_round = server_round

        # FIRST ROUND LIMITATION: No prior global weights exist, so we cannot
        # compute pseudo-gradients. Fall back to standard FedAvg for the first round.
        # This is a known limitation of server-side gradient computation.
        if self.global_weights is None:
            return self._fedavg_aggregate(results), {'warmup': 1, 'caacfl_active': 0}

        # Extract client updates and compute pseudo-gradients
        # NOTE: This is where we convert Flower's weight-based protocol to gradients
        # for CAAC-FL. Memory overhead: O(model_size) to store global_weights.
        # Computation overhead: O(model_size * num_clients) for subtraction operations.
        client_gradients = {}
        client_samples = {}

        # FLOWER CLIENT ID MAPPING:
        # Flower assigns unique IDs (large integers) to clients, not 0-indexed integers.
        # The CAACFLAggregator expects integer IDs from 0 to num_clients-1.
        # We map Flower's client IDs to sequential indices for profile management.
        # This is a one-time mapping that persists across rounds.
        if not hasattr(self, '_client_id_map'):
            self._client_id_map = {}

        for client_proxy, fit_res in results:
            flower_cid = client_proxy.cid

            # Map Flower's client ID to sequential index
            if flower_cid not in self._client_id_map:
                self._client_id_map[flower_cid] = len(self._client_id_map)
            client_id = self._client_id_map[flower_cid]

            client_weights = parameters_to_ndarrays(fit_res.parameters)

            # PSEUDO-GRADIENT COMPUTATION:
            # gradient = new_client_weights - previous_global_weights
            # This captures what the client "wanted to change" in the model.
            # For honest clients: reflects true learning from local data
            # For Byzantine clients: reflects their attack vector (sign flip, noise, etc.)
            gradient_arrays = []
            for cw, gw in zip(client_weights, self.global_weights):
                gradient_arrays.append(cw - gw)

            # Flatten for CAAC-FL processing (aggregator expects 1D gradient vectors)
            flat_gradient = np.concatenate([g.flatten() for g in gradient_arrays])

            client_gradients[client_id] = flat_gradient
            client_samples[client_id] = fit_res.num_examples

        # Use CAACFLAggregator for anomaly detection, clipping, and aggregation
        aggregated_gradient, client_stats = self.aggregator.aggregate(
            client_gradients, client_samples
        )

        # Track detection statistics
        round_detections = sum(1 for s in client_stats if s['is_anomalous'])
        self.detection_stats['total_detections'] += round_detections
        self.detection_stats['detections_per_round'].append(round_detections)

        clip_factors = [s['scaling_factor'] for s in client_stats]
        self.detection_stats['mean_clip_factors'].append(np.mean(clip_factors))

        # Compute confusion matrix for this round
        detected_ids = set(s['client_id'] for s in client_stats if s['is_anomalous'])
        self.detection_stats['detected_ids_per_round'].append(list(detected_ids))

        # TP: Byzantine clients that were detected
        tp = len(detected_ids & self.byzantine_ids)
        # FP: Honest clients that were incorrectly detected
        fp = len(detected_ids & self.honest_ids)
        # TN: Honest clients that were correctly NOT detected
        tn = len(self.honest_ids - detected_ids)
        # FN: Byzantine clients that were NOT detected
        fn = len(self.byzantine_ids - detected_ids)

        # Update cumulative counts
        self.detection_stats['total_tp'] += tp
        self.detection_stats['total_fp'] += fp
        self.detection_stats['total_tn'] += tn
        self.detection_stats['total_fn'] += fn

        # Store per-round confusion matrix
        self.detection_stats['confusion_per_round'].append({
            'round': server_round,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        })

        # Unflatten the aggregated gradient back to weight shapes
        # and apply to global weights to get new weights
        new_weights = []
        idx = 0
        for gw in self.global_weights:
            num_elements = gw.size
            grad_slice = aggregated_gradient[idx:idx + num_elements].reshape(gw.shape)
            new_weights.append(gw + grad_slice)  # Apply gradient update
            idx += num_elements

        # Build metrics
        metrics = {
            'round': server_round,
            'num_detections': round_detections,
            'mean_clip_factor': np.mean(clip_factors),
            'mean_anomaly_score': np.mean([s['anomaly_score'] for s in client_stats]),
            'mean_reliability': np.mean([s['reliability'] for s in client_stats]),
            'caacfl_active': 1,
        }

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics.update(aggregated_metrics)

        return ndarrays_to_parameters(new_weights), metrics

    def _fedavg_aggregate(self, results: List[Tuple[ClientProxy, FitRes]]) -> Parameters:
        """Simple FedAvg aggregation (for first round before global weights exist)"""
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        aggregated = None
        for _, fit_res in results:
            weight = fit_res.num_examples / total_examples
            weights = parameters_to_ndarrays(fit_res.parameters)
            if aggregated is None:
                aggregated = [w * weight for w in weights]
            else:
                for i, w in enumerate(weights):
                    aggregated[i] += w * weight

        return ndarrays_to_parameters(aggregated)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}

        accuracies = [r.metrics.get("accuracy", 0) for _, r in results if r.metrics]
        losses = [r.loss for _, r in results if r.loss is not None]

        metrics = {
            "accuracy": np.mean(accuracies) if accuracies else 0,
            "loss": np.mean(losses) if losses else 0,
        }

        return metrics.get("loss"), metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model using evaluate_fn"""
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

    def get_detection_stats(self) -> dict:
        """Return detection statistics for analysis"""
        stats = self.detection_stats.copy()
        # Add aggregator's summary stats
        stats['aggregator_summary'] = self.aggregator.get_summary_stats()
        stats['round_stats'] = self.aggregator.round_stats
        return stats


# ============================================================================
# Factory function
# ============================================================================

def create_caacfl_strategy(
    num_clients: int,
    evaluate_fn=None,
    fit_metrics_aggregation_fn=None,
    # CAAC-FL parameters - tuned for both directional and magnitude attacks
    alpha: float = 0.05,  # Slower EWMA updates to resist profile poisoning
    gamma: float = 0.1,
    tau_base: float = 1.2,  # Lower threshold to catch more anomalies
    beta: float = 0.5,
    window_size: int = 5,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),  # Prioritize magnitude for random noise
    # Cold-start mitigation parameters
    warmup_rounds: int = 10,  # Longer conservative period
    warmup_factor: float = 0.3,  # Stricter during warmup
    min_rounds_for_trust: int = 5,  # Longer trust building period
    use_cross_comparison: bool = True,
    use_population_init: bool = True,
    new_client_weight: float = 0.3,  # Less influence for new clients
    # Byzantine client tracking for TP/FP/TN/FN metrics
    byzantine_ids: Optional[List[int]] = None,
) -> CAACFLStrategy:
    """
    Factory function to create CAAC-FL strategy with full algorithm.

    Args:
        num_clients: Total number of clients
        evaluate_fn: Centralized evaluation function
        fit_metrics_aggregation_fn: Metrics aggregation function

        CAAC-FL Parameters:
            alpha: EWMA smoothing factor for profile updates
            gamma: Reliability update rate
            tau_base: Base anomaly threshold (2.0 = 2 std deviations)
            beta: Threshold flexibility based on reliability
            window_size: History window for temporal analysis
            weights: (w_mag, w_dir, w_temp) anomaly dimension weights

        Cold-Start Parameters:
            warmup_rounds: Conservative threshold period
            warmup_factor: Threshold reduction during warmup
            min_rounds_for_trust: Rounds before reliability bonus applies
            use_cross_comparison: Cross-client gradient comparison
            use_population_init: Initialize profiles from population stats
            new_client_weight: Weight reduction for new clients

    Returns:
        Configured CAACFLStrategy instance
    """
    return CAACFLStrategy(
        num_clients=num_clients,
        alpha=alpha,
        gamma=gamma,
        tau_base=tau_base,
        beta=beta,
        window_size=window_size,
        weights=weights,
        warmup_rounds=warmup_rounds,
        warmup_factor=warmup_factor,
        min_rounds_for_trust=min_rounds_for_trust,
        use_cross_comparison=use_cross_comparison,
        use_population_init=use_population_init,
        new_client_weight=new_client_weight,
        byzantine_ids=byzantine_ids,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
