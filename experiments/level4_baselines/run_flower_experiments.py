#!/usr/bin/env python3
"""
Level 4: Baseline Byzantine-Robust FL Experiments with Flower/Ray

Runs comprehensive baseline experiments for comparison with CAAC-FL.
Uses identical protocol: 25 clients, 50 rounds, CIFAR-10, Dirichlet α=0.5

Based on Li et al. 2024, "An Experimental Study of Byzantine-Robust
Aggregation Schemes in Federated Learning", IEEE TBDATA.

Aggregation Strategies:
- FedAvg: Weighted average (non-robust baseline)
- FedMedian: Coordinate-wise median
- TrimmedMean: Remove extreme values before averaging
- Krum: Distance-based selection (single client)
- MultiKrum: Distance-based selection (top-k clients)
- GeoMed: Geometric median (Weiszfeld algorithm)
- CC: Centered Clipping (momentum-based clipping)
- Clustering: Cosine similarity-based clustering
- ClippedClustering: Clustering + norm clipping (Li et al.'s proposed method)

Attacks (from Li et al. 2024):
- none: No attack (baseline)
- sign_flipping: Negate gradients
- random_noise: Gaussian noise attack
- alie: A Little Is Enough attack
- ipm_small: Inner Product Manipulation (ε=0.5)
- ipm_large: Inner Product Manipulation (ε=100)
- label_flipping: Train on flipped labels

Scenarios:
- no_attack: Baseline with no Byzantine clients
- immediate: Attack from round 0 (traditional threat model)
- delayed: Attack from round 15 (delayed compromise model)

Detection metrics tracked for strategies with client selection.
"""

import sys
import os
import argparse
import json
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Callable, Union
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    FitIns,
    EvaluateIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.client import NumPyClient

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'level3_attacks')))

from shared.models import SimpleCNN
from shared.data_utils import load_cifar10, partition_data_dirichlet
from level3_attacks.attacks import create_attack, SignFlippingAttack, RandomNoiseAttack, ALIEAttack, IPMAttack, LabelFlippingAttack, NoAttack

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 0.01
DEFAULT_NUM_ROUNDS = 50
DEFAULT_NUM_CLIENTS = 25
DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_ALPHA = 0.5
DEFAULT_COMPROMISE_ROUND = 15

# ============================================================================
# Novel Experimental Dimensions (Beyond Li et al. 2024)
# ============================================================================
# Li et al. 2024 only tested immediate attacks (round 0). We extend this with:
#
# 1. COMPROMISE TIMING: When attacks start (delayed compromise threat model)
#    - Tests whether partial model convergence provides inherent robustness
#
# 2. ATTACK WINDOWS: When attacks start AND end (transient attack model)
#    - Tests recovery after attack cessation
#
# 3. NON-IID SEVERITY: Varying Dirichlet alpha across the heterogeneity spectrum
#    - Li et al. only tested α=0.1 (severe) and IID
#
# These dimensions allow studying realistic threat models where devices are
# compromised at various points during training, not just from the beginning.
# ============================================================================

# Compromise timing scenarios (when attacks START)
# Li et al. only used round 0. We add delayed compromise scenarios.
COMPROMISE_TIMINGS = [0, 10, 20, 30, 40]

# Attack window scenarios: (start_round, end_round)
# None for end_round means attack continues until training ends
ATTACK_WINDOWS = [
    (0, None),    # Full attack throughout training (Li et al. baseline)
    (0, 25),      # Early attack window, then honest behavior
    (25, None),   # Late compromise only
    (10, 40),     # Mid-training attack window
    (0, 10),      # Brief initial attack
]

# Non-IID severity levels (Dirichlet alpha)
# Lower alpha = more severe heterogeneity
ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0]  # severe -> moderate -> mild

# Byzantine ratios to test
BYZANTINE_RATIOS = [0.1, 0.2, 0.3, 0.4]

# Attack configs following Li et al. 2024 paper
# Format: (attack_name, byzantine_ratio, extra_params)
ATTACK_CONFIGS = [
    ("none", 0.0, {}),
    # Sign Flipping (SF)
    ("sign_flipping", 0.1, {}),
    ("sign_flipping", 0.2, {}),
    ("sign_flipping", 0.3, {}),
    # Random Noise (Gaussian with noise_scale matching Li et al.)
    ("random_noise", 0.1, {"noise_scale": 5.0}),
    ("random_noise", 0.2, {"noise_scale": 5.0}),
    ("random_noise", 0.3, {"noise_scale": 5.0}),
    # ALIE (A Little Is Enough)
    ("alie", 0.1, {}),
    ("alie", 0.2, {}),
    ("alie", 0.3, {}),
    # IPM small epsilon (ε=0.5) - reduces magnitude
    ("ipm_small", 0.1, {"epsilon": 0.5}),
    ("ipm_small", 0.2, {"epsilon": 0.5}),
    ("ipm_small", 0.3, {"epsilon": 0.5}),
    # IPM large epsilon (ε=100) - reverses direction
    ("ipm_large", 0.1, {"epsilon": 100.0}),
    ("ipm_large", 0.2, {"epsilon": 100.0}),
    ("ipm_large", 0.3, {"epsilon": 100.0}),
    # Label Flipping (LF) - data poisoning
    ("label_flipping", 0.1, {}),
    ("label_flipping", 0.2, {}),
    ("label_flipping", 0.3, {}),
]

# Focused attack configs for comprehensive timing/window studies
# Uses 20% Byzantine ratio as the canonical test case
FOCUSED_ATTACK_CONFIGS = [
    ("sign_flipping", 0.2, {}),
    ("alie", 0.2, {}),
    ("ipm_large", 0.2, {"epsilon": 100.0}),
    ("label_flipping", 0.2, {}),
]

# All available strategies
ALL_STRATEGIES = ["fedavg", "fedmedian", "trimmed", "krum", "multikrum",
                  "geomed", "cc", "clustering", "clippedclustering"]


# ============================================================================
# Detection Stats Tracker
# ============================================================================

class DetectionTracker:
    """Track detection metrics for strategies that identify bad clients."""

    def __init__(self, byzantine_ids: Set[int], enabled: bool = True):
        self.byzantine_ids = byzantine_ids
        self.enabled = enabled
        self.stats = {
            'total_tp': 0,
            'total_fp': 0,
            'total_tn': 0,
            'total_fn': 0,
            'per_round': [],
        }

    def update(self, server_round: int, all_client_ids: Set[int],
               selected_ids: Set[int], scores: Optional[Dict[int, float]] = None):
        """
        Update detection stats based on client selection.

        For Krum/MultiKrum:
        - Selected = predicted good (not Byzantine)
        - Rejected = predicted bad (Byzantine)

        TP: Byzantine clients that were rejected (correctly identified)
        FP: Honest clients that were rejected (false alarm)
        TN: Honest clients that were selected (correctly trusted)
        FN: Byzantine clients that were selected (missed detection)
        """
        if not self.enabled:
            return

        rejected_ids = all_client_ids - selected_ids
        honest_ids = all_client_ids - self.byzantine_ids

        tp = len(rejected_ids & self.byzantine_ids)
        fp = len(rejected_ids & honest_ids)
        tn = len(selected_ids & honest_ids)
        fn = len(selected_ids & self.byzantine_ids)

        self.stats['total_tp'] += tp
        self.stats['total_fp'] += fp
        self.stats['total_tn'] += tn
        self.stats['total_fn'] += fn

        round_stats = {
            'round': server_round,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'selected': list(selected_ids),
            'rejected': list(rejected_ids),
        }
        if scores:
            round_stats['scores'] = scores
        self.stats['per_round'].append(round_stats)

    def get_stats(self) -> Dict:
        """Return detection statistics with precision/recall/F1."""
        if not self.enabled:
            return {'detection': 'N/A'}

        stats = self.stats.copy()
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


# ============================================================================
# Strategy Implementations
# ============================================================================

class BaselineStrategy(Strategy):
    """Base class for baseline aggregation strategies."""

    def __init__(
        self,
        initial_parameters: Parameters,
        num_clients: int,
        byzantine_ids: Set[int],
        evaluate_fn: Callable,
        strategy_name: str = "base",
        compromise_round: int = 0,
        detection_enabled: bool = False,
    ):
        self.initial_parameters = initial_parameters
        self.num_clients = num_clients
        self.byzantine_ids = byzantine_ids
        self.evaluate_fn = evaluate_fn
        self.strategy_name = strategy_name
        self.compromise_round = compromise_round
        self.current_round = 0

        self.global_weights = parameters_to_ndarrays(initial_parameters)
        self.detection = DetectionTracker(byzantine_ids, enabled=detection_enabled)

        self.metrics_history = {
            "rounds": [],
            "test_accuracy": [],
            "test_loss": [],
        }
        self.start_time = None

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.current_round = server_round
        if self.start_time is None:
            self.start_time = time.time()

        config = {
            "local_epochs": DEFAULT_LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "server_round": server_round,
        }

        sample_size = min(self.num_clients, client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)

        return [(client, FitIns(parameters, config)) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        config = {"server_round": server_round}
        sample_size = min(self.num_clients, client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_examples = sum([r.num_examples for _, r in results])
        weighted_loss = sum([r.num_examples * r.loss for _, r in results]) / total_examples
        weighted_acc = sum([r.num_examples * r.metrics.get("accuracy", 0) for _, r in results]) / total_examples

        return weighted_loss, {"accuracy": weighted_acc}

    def evaluate(self, server_round: int, parameters: Parameters):
        if self.evaluate_fn is None:
            return None

        loss, accuracy = self.evaluate_fn(parameters)
        self._log_round(server_round, loss, accuracy)

        return loss, {"accuracy": accuracy}

    def _log_round(self, server_round: int, loss: float, accuracy: float):
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Show attack marker at compromise round
        attack_marker = ""
        if server_round == self.compromise_round and self.byzantine_ids:
            attack_marker = " [ATTACK BEGINS]"

        if server_round > 0:
            time_per_round = elapsed / server_round
            remaining = DEFAULT_NUM_ROUNDS - server_round
            eta = str(timedelta(seconds=int(time_per_round * remaining)))
        else:
            eta = "calculating..."

        print(f"Round {server_round:2d}/{DEFAULT_NUM_ROUNDS}: Acc={accuracy:.2f}%, Loss={loss:.4f} | ETA: {eta}{attack_marker}")

        self.metrics_history["rounds"].append(server_round)
        self.metrics_history["test_accuracy"].append(accuracy)
        self.metrics_history["test_loss"].append(loss)

    def get_metrics(self) -> Dict:
        return self.metrics_history

    def get_detection_stats(self) -> Dict:
        return self.detection.get_stats()


class FedAvgStrategy(BaselineStrategy):
    """Standard FedAvg - weighted average (non-robust baseline)."""

    def __init__(self, **kwargs):
        super().__init__(strategy_name="fedavg", detection_enabled=False, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Weighted average of all client updates
        total_examples = sum([fit_res.num_examples for _, fit_res in results])

        aggregated_weights = None
        for client, fit_res in results:
            weight = fit_res.num_examples / total_examples
            client_weights = parameters_to_ndarrays(fit_res.parameters)

            if aggregated_weights is None:
                aggregated_weights = [w * weight for w in client_weights]
            else:
                for i, w in enumerate(client_weights):
                    aggregated_weights[i] += w * weight

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class FedMedianStrategy(BaselineStrategy):
    """FedMedian - coordinate-wise median (Byzantine-robust)."""

    def __init__(self, **kwargs):
        super().__init__(strategy_name="fedmedian", detection_enabled=False, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        all_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Coordinate-wise median
        aggregated_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in all_weights], axis=0)
            layer_median = np.median(layer_stack, axis=0)
            aggregated_weights.append(layer_median)

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class TrimmedMeanStrategy(BaselineStrategy):
    """Trimmed Mean - remove extreme values before averaging."""

    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        super().__init__(strategy_name="trimmed", detection_enabled=False, **kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        n = len(results)
        trim_count = int(n * self.trim_ratio)

        all_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        aggregated_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in all_weights], axis=0)

            if trim_count > 0 and n > 2 * trim_count:
                sorted_stack = np.sort(layer_stack, axis=0)
                trimmed = sorted_stack[trim_count:n-trim_count]
                layer_mean = np.mean(trimmed, axis=0)
            else:
                layer_mean = np.mean(layer_stack, axis=0)

            aggregated_weights.append(layer_mean)

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class KrumStrategy(BaselineStrategy):
    """Krum - select single client closest to others (Byzantine-robust)."""

    def __init__(self, num_byzantine: int = 0, **kwargs):
        super().__init__(strategy_name="krum", detection_enabled=True, **kwargs)
        self.num_byzantine = num_byzantine

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        n = len(results)
        f = self.num_byzantine

        # Get client IDs and weights
        client_ids = []
        all_weights_flat = []
        all_weights_structured = []

        for client, fit_res in results:
            cid = int(client.cid)
            client_ids.append(cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            all_weights_flat.append(flat)
            all_weights_structured.append(weights)

        all_weights_flat = np.array(all_weights_flat)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(all_weights_flat[i] - all_weights_flat[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores: sum of distances to (n - f - 2) closest neighbors
        k = max(1, n - f - 2)
        scores = {}
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:k+1])  # Exclude self
            scores[client_ids[i]] = float(score)

        # Select client with lowest score
        best_idx = min(range(n), key=lambda i: scores[client_ids[i]])
        selected_id = client_ids[best_idx]

        # Update detection stats
        all_ids = set(client_ids)
        selected_ids = {selected_id}
        self.detection.update(server_round, all_ids, selected_ids, scores)

        # Use the selected client's weights
        self.global_weights = all_weights_structured[best_idx]
        return ndarrays_to_parameters(self.global_weights), {}


class MultiKrumStrategy(BaselineStrategy):
    """Multi-Krum - average top-k clients closest to others."""

    def __init__(self, num_byzantine: int = 0, k: Optional[int] = None, **kwargs):
        super().__init__(strategy_name="multikrum", detection_enabled=True, **kwargs)
        self.num_byzantine = num_byzantine
        self.k = k  # If None, defaults to n - num_byzantine

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        n = len(results)
        f = self.num_byzantine
        m = self.k if self.k else max(1, n - f)

        # Get client IDs and weights
        client_ids = []
        all_weights_flat = []
        all_weights_structured = []
        num_examples = []

        for client, fit_res in results:
            cid = int(client.cid)
            client_ids.append(cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            all_weights_flat.append(flat)
            all_weights_structured.append(weights)
            num_examples.append(fit_res.num_examples)

        all_weights_flat = np.array(all_weights_flat)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(all_weights_flat[i] - all_weights_flat[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores
        k_neighbors = max(1, n - f - 2)
        scores = {}
        score_list = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:k_neighbors+1])
            scores[client_ids[i]] = float(score)
            score_list.append(score)

        # Select top-m clients with lowest scores
        selected_indices = np.argsort(score_list)[:m]
        selected_ids = set(client_ids[i] for i in selected_indices)

        # Update detection stats
        all_ids = set(client_ids)
        self.detection.update(server_round, all_ids, selected_ids, scores)

        # Average selected clients (weighted by num_examples)
        total_examples = sum(num_examples[i] for i in selected_indices)
        aggregated_weights = [np.zeros_like(w) for w in all_weights_structured[0]]

        for idx in selected_indices:
            weight_factor = num_examples[idx] / total_examples
            for i, w in enumerate(all_weights_structured[idx]):
                aggregated_weights[i] += w * weight_factor

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class GeoMedStrategy(BaselineStrategy):
    """
    Geometric Median (GeoMed) - Weiszfeld algorithm.

    From Chen et al., "Distributed Statistical Machine Learning in Adversarial Settings"
    and Pillutla et al., "Robust Aggregation for Federated Learning", 2022.

    Finds the point that minimizes sum of Euclidean distances to all client updates.
    More robust than coordinate-wise median as it considers vectors holistically.
    """

    def __init__(self, max_iter: int = 100, eps: float = 1e-6, **kwargs):
        super().__init__(strategy_name="geomed", detection_enabled=False, **kwargs)
        self.max_iter = max_iter
        self.eps = eps

    def _weiszfeld_step(self, points: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Single Weiszfeld iteration for geometric median."""
        distances = np.linalg.norm(points - current, axis=1)
        # Avoid division by zero
        distances = np.maximum(distances, self.eps)
        weights = 1.0 / distances
        weights /= np.sum(weights)
        return np.sum(points * weights[:, np.newaxis], axis=0)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        all_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Compute geometric median layer by layer
        aggregated_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_stack = np.stack([w[layer_idx].flatten() for w in all_weights], axis=0)

            # Initialize with coordinate-wise mean
            current = np.mean(layer_stack, axis=0)

            # Weiszfeld algorithm iterations
            for _ in range(self.max_iter):
                new = self._weiszfeld_step(layer_stack, current)
                if np.linalg.norm(new - current) < self.eps:
                    break
                current = new

            aggregated_weights.append(current.reshape(all_weights[0][layer_idx].shape))

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class CenteredClippingStrategy(BaselineStrategy):
    """
    Centered Clipping (CC) - iterative clipping Byzantine-robust aggregation.

    From Karimireddy et al., "Learning from History for Byzantine Robust Optimization", ICML 2021.
    Also described in Li et al. 2024, Section IV-F.

    Algorithm (Eq. 8 from Li et al.):
        Δₗ₊₁ ← Δₗ + (1/K) Σₖ (Δₖ - Δₗ) min(1, τ/||Δₖ - Δₗ||)

    Where Δ₀ is the aggregated update from the previous round.
    The clipping ensures no single client can have too much influence.
    """

    def __init__(self, tau: float = 1.0, num_iters: int = 3, **kwargs):
        super().__init__(strategy_name="cc", detection_enabled=False, **kwargs)
        self.tau = tau  # Clipping threshold τ₁
        self.num_iters = num_iters  # Number of clipping iterations
        self.previous_aggregate = None  # Δ₀ from previous round

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        n = len(results)
        all_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Initialize Δ₀ with previous round's aggregate (or mean for first round)
        if self.previous_aggregate is None:
            current = [np.mean(np.stack([w[i] for w in all_weights], axis=0), axis=0)
                      for i in range(len(all_weights[0]))]
        else:
            current = self.previous_aggregate

        # Iterative centered clipping (Li et al. Eq. 8)
        for _ in range(self.num_iters):
            aggregated_weights = []
            for layer_idx in range(len(all_weights[0])):
                center = current[layer_idx]

                # Compute clipped contribution from each client
                clipped_sum = np.zeros_like(center)
                for weights in all_weights:
                    diff = weights[layer_idx] - center
                    norm = np.linalg.norm(diff)
                    if norm > 0:
                        clip_factor = min(1.0, self.tau / norm)
                        clipped_sum += diff * clip_factor
                    # If norm is 0, diff is 0, nothing to add

                # Update: Δₗ₊₁ = Δₗ + (1/K) * Σ clipped_diffs
                new_layer = center + clipped_sum / n
                aggregated_weights.append(new_layer)

            current = aggregated_weights

        # Store for next round
        self.previous_aggregate = aggregated_weights
        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class ClusteringStrategy(BaselineStrategy):
    """
    Cosine similarity-based clustering aggregation with agglomerative clustering.

    From Li et al. 2024, Section IV-G:
    "separates the client population into two groups based on the cosine similarities
    using agglomerative clustering with average linkage"

    Algorithm:
    1. Compute pairwise cosine distances (1 - similarity)
    2. Apply agglomerative clustering with average linkage to split into 2 groups
    3. Select the larger group for aggregation
    """

    def __init__(self, **kwargs):
        super().__init__(strategy_name="clustering", detection_enabled=True, **kwargs)

    def _compute_cosine_distance_matrix(self, flat_weights: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine distance matrix (1 - cosine_similarity)."""
        n = len(flat_weights)
        distances = np.zeros((n, n))

        # Normalize vectors
        norms = np.linalg.norm(flat_weights, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        normalized = flat_weights / norms

        # Compute cosine similarities and convert to distances
        for i in range(n):
            for j in range(i + 1, n):
                similarity = np.dot(normalized[i], normalized[j])
                distance = 1.0 - similarity
                distances[i, j] = distance
                distances[j, i] = distance

        return distances

    def _agglomerative_cluster(self, distance_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Agglomerative clustering with average linkage into exactly 2 clusters.

        Uses scipy's hierarchy if available, otherwise manual implementation.
        """
        n = distance_matrix.shape[0]

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            # Convert to condensed form for scipy
            condensed = squareform(distance_matrix)
            # Average linkage clustering
            Z = linkage(condensed, method='average')
            # Cut into 2 clusters
            labels = fcluster(Z, t=2, criterion='maxclust')

            # Group by cluster label
            cluster0 = [i for i in range(n) if labels[i] == 1]
            cluster1 = [i for i in range(n) if labels[i] == 2]

        except ImportError:
            # Manual agglomerative clustering fallback
            # Start with each point as its own cluster
            clusters = [[i] for i in range(n)]
            cluster_distances = distance_matrix.copy()

            while len(clusters) > 2:
                # Find closest pair of clusters (average linkage)
                min_dist = float('inf')
                merge_i, merge_j = 0, 1

                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        # Average linkage: mean distance between all pairs
                        total_dist = 0.0
                        count = 0
                        for idx_i in clusters[i]:
                            for idx_j in clusters[j]:
                                total_dist += distance_matrix[idx_i, idx_j]
                                count += 1
                        avg_dist = total_dist / count if count > 0 else float('inf')

                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_i, merge_j = i, j

                # Merge clusters
                new_cluster = clusters[merge_i] + clusters[merge_j]
                clusters = [c for idx, c in enumerate(clusters) if idx not in (merge_i, merge_j)]
                clusters.append(new_cluster)

            cluster0, cluster1 = clusters[0], clusters[1]

        return cluster0, cluster1

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Get client IDs and weights
        client_ids = []
        all_weights_flat = []
        all_weights_structured = []
        num_examples = []

        for client, fit_res in results:
            cid = int(client.cid)
            client_ids.append(cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            all_weights_flat.append(flat)
            all_weights_structured.append(weights)
            num_examples.append(fit_res.num_examples)

        all_weights_flat = np.array(all_weights_flat)

        # Compute cosine distance matrix
        distance_matrix = self._compute_cosine_distance_matrix(all_weights_flat)

        # Agglomerative clustering into 2 groups
        cluster0, cluster1 = self._agglomerative_cluster(distance_matrix)

        # Select larger cluster
        largest_cluster = cluster0 if len(cluster0) >= len(cluster1) else cluster1
        selected_ids = set(client_ids[i] for i in largest_cluster)

        # Update detection stats
        all_ids = set(client_ids)
        scores = {cid: 0.0 for cid in client_ids}  # No scores for clustering
        self.detection.update(server_round, all_ids, selected_ids, scores)

        # Average the largest cluster (weighted by num_examples)
        total_examples = sum(num_examples[i] for i in largest_cluster)
        aggregated_weights = [np.zeros_like(w) for w in all_weights_structured[0]]

        for idx in largest_cluster:
            weight_factor = num_examples[idx] / total_examples
            for i, w in enumerate(all_weights_structured[idx]):
                aggregated_weights[i] += w * weight_factor

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class ClippedClusteringStrategy(BaselineStrategy):
    """
    Clipped Clustering - Li et al.'s proposed method.

    From Li et al. 2024, Section IV-H:
    "We enhance the robustness of the aforementioned Clustering aggregation scheme
    by performing a clipping on all the updates BEFORE clustering"

    "we set the clipping value hyper-parameter based on the statistics of the
    historical norms of the updates uploaded during training, i.e., we save the
    update norms up to current iteration and automatically set τ using the
    50-th percentile value (i.e., the median) of the history"

    Algorithm:
    1. Compute norms of all current updates, add to historical norm list
    2. Set τ = median of all historical norms
    3. Clip ALL updates to norm τ BEFORE clustering
    4. Apply agglomerative clustering with average linkage on clipped updates
    5. Select larger cluster and average
    """

    def __init__(self, **kwargs):
        super().__init__(strategy_name="clippedclustering", detection_enabled=True, **kwargs)
        self.historical_norms = []  # Track norms across all rounds

    def _compute_cosine_distance_matrix(self, flat_weights: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine distance matrix (1 - cosine_similarity)."""
        n = len(flat_weights)
        distances = np.zeros((n, n))

        # Normalize vectors
        norms = np.linalg.norm(flat_weights, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = flat_weights / norms

        for i in range(n):
            for j in range(i + 1, n):
                similarity = np.dot(normalized[i], normalized[j])
                distance = 1.0 - similarity
                distances[i, j] = distance
                distances[j, i] = distance

        return distances

    def _agglomerative_cluster(self, distance_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """Agglomerative clustering with average linkage into exactly 2 clusters."""
        n = distance_matrix.shape[0]

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            condensed = squareform(distance_matrix)
            Z = linkage(condensed, method='average')
            labels = fcluster(Z, t=2, criterion='maxclust')

            cluster0 = [i for i in range(n) if labels[i] == 1]
            cluster1 = [i for i in range(n) if labels[i] == 2]

        except ImportError:
            # Manual fallback
            clusters = [[i] for i in range(n)]

            while len(clusters) > 2:
                min_dist = float('inf')
                merge_i, merge_j = 0, 1

                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        total_dist = 0.0
                        count = 0
                        for idx_i in clusters[i]:
                            for idx_j in clusters[j]:
                                total_dist += distance_matrix[idx_i, idx_j]
                                count += 1
                        avg_dist = total_dist / count if count > 0 else float('inf')

                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_i, merge_j = i, j

                new_cluster = clusters[merge_i] + clusters[merge_j]
                clusters = [c for idx, c in enumerate(clusters) if idx not in (merge_i, merge_j)]
                clusters.append(new_cluster)

            cluster0, cluster1 = clusters[0], clusters[1]

        return cluster0, cluster1

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Get client IDs and weights
        client_ids = []
        all_weights_flat = []
        all_weights_structured = []
        num_examples = []

        for client, fit_res in results:
            cid = int(client.cid)
            client_ids.append(cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            all_weights_flat.append(flat)
            all_weights_structured.append(weights)
            num_examples.append(fit_res.num_examples)

        all_weights_flat = np.array(all_weights_flat)

        # Step 1: Compute current norms and add to historical list
        current_norms = [np.linalg.norm(w) for w in all_weights_flat]
        self.historical_norms.extend(current_norms)

        # Step 2: Set τ = median of ALL historical norms
        tau = np.median(self.historical_norms)

        # Step 3: Clip ALL updates to norm τ BEFORE clustering
        clipped_weights_flat = []
        clipped_weights_structured = []

        for i in range(len(all_weights_flat)):
            norm = current_norms[i]
            if norm > tau and norm > 0:
                scale = tau / norm
                # Clip flat weights
                clipped_flat = all_weights_flat[i] * scale
                # Clip structured weights
                clipped_structured = [w * scale for w in all_weights_structured[i]]
            else:
                clipped_flat = all_weights_flat[i]
                clipped_structured = all_weights_structured[i]

            clipped_weights_flat.append(clipped_flat)
            clipped_weights_structured.append(clipped_structured)

        clipped_weights_flat = np.array(clipped_weights_flat)

        # Step 4: Apply agglomerative clustering on CLIPPED updates
        distance_matrix = self._compute_cosine_distance_matrix(clipped_weights_flat)
        cluster0, cluster1 = self._agglomerative_cluster(distance_matrix)

        # Step 5: Select larger cluster
        largest_cluster = cluster0 if len(cluster0) >= len(cluster1) else cluster1
        selected_ids = set(client_ids[i] for i in largest_cluster)

        # Update detection stats
        all_ids = set(client_ids)
        scores = {cid: 0.0 for cid in client_ids}
        self.detection.update(server_round, all_ids, selected_ids, scores)

        # Step 6: Average the clipped weights from largest cluster
        total_examples = sum(num_examples[i] for i in largest_cluster)
        aggregated_weights = [np.zeros_like(w) for w in clipped_weights_structured[0]]

        for idx in largest_cluster:
            weight_factor = num_examples[idx] / total_examples
            for i, w in enumerate(clipped_weights_structured[idx]):
                aggregated_weights[i] += w * weight_factor

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


# ============================================================================
# Flower Client with Attack Window Support
# ============================================================================

class BaselineFlowerClient(NumPyClient):
    """
    Flower client with Byzantine attack support and attack window control.

    Novel contribution beyond Li et al. 2024:
    - Supports delayed compromise (attack starts at compromise_round)
    - Supports attack windows (attack ends at attack_end_round)
    - Enables study of transient attacks and recovery dynamics
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        client_id: int,
        is_byzantine: bool = False,
        attack = None,
        compromise_round: int = 0,
        attack_end_round: Optional[int] = None,  # None = attack until end
    ):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.device = device
        self.client_id = client_id
        self.is_byzantine = is_byzantine
        self.attack = attack if attack is not None else NoAttack()
        self.compromise_round = compromise_round
        self.attack_end_round = attack_end_round  # None means attack continues forever
        self.original_model = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Store original for attacks
        if self.is_byzantine:
            self.original_model = type(self.model)(num_classes=10).to(self.device)
            self.original_model.load_state_dict(self.model.state_dict())

    def _should_attack(self, config) -> bool:
        """
        Determine if attack should be applied this round.

        Attack is active when:
        - Client is byzantine AND
        - Current round >= compromise_round AND
        - Current round < attack_end_round (if attack_end_round is set)
        """
        if not self.is_byzantine:
            return False
        server_round = config.get('server_round', 0)

        # Check if within attack window
        if server_round < self.compromise_round:
            return False
        if self.attack_end_round is not None and server_round >= self.attack_end_round:
            return False
        return True

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Train locally
        self._train(config)

        # Apply attack if conditions met
        if self._should_attack(config):
            self.model = self.attack.apply(self.model, self.original_model)

        return self.get_parameters(config), len(self.trainloader.dataset), {
            "client_id": self.client_id,
            "is_byzantine": int(self.is_byzantine),
            "attack_applied": int(self._should_attack(config)),
        }

    def _train(self, config):
        local_epochs = config.get("local_epochs", DEFAULT_LOCAL_EPOCHS)
        lr = config.get("learning_rate", LEARNING_RATE)

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

        for _ in range(local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, total, {"accuracy": 100.0 * correct / total}


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(
    strategy_name: str,
    attack_name: str,
    byzantine_ratio: float,
    compromise_round: int,
    attack_end_round: Optional[int] = None,  # None = attack until training ends
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
    output_dir: str = "./results/flower",
    device: torch.device = None,
    attack_params: Dict = None,
):
    """
    Run a single baseline experiment.

    Novel parameters beyond Li et al. 2024:
    - compromise_round: When attack starts (delayed compromise)
    - attack_end_round: When attack ends (transient attack window)
    """
    if attack_params is None:
        attack_params = {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Determine scenario name for output files
    byz_pct = int(byzantine_ratio * 100)
    if attack_name == "none":
        scenario = "baseline"
        scenario_suffix = ""
    elif compromise_round == 0 and attack_end_round is None:
        scenario = "immediate"
        scenario_suffix = "_imm"
    elif attack_end_round is not None:
        scenario = "window"
        scenario_suffix = f"_w{compromise_round}-{attack_end_round}"
    else:
        scenario = "delayed"
        scenario_suffix = f"_d{compromise_round}"

    # Include alpha in experiment name for non-default values
    alpha_suffix = f"_a{alpha}" if alpha != DEFAULT_ALPHA else ""
    exp_name = f"{strategy_name}_{attack_name}_byz{byz_pct}_seed{seed}{scenario_suffix}{alpha_suffix}"

    # Attack window description
    if attack_end_round is None:
        window_desc = f"rounds [{compromise_round}, end)"
    else:
        window_desc = f"rounds [{compromise_round}, {attack_end_round})"

    print("\n" + "=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Attack: {attack_name}, Byzantine ratio: {byz_pct}%")
    print(f"  Attack window: {window_desc}")
    print(f"  Rounds: {num_rounds}, Local epochs: {local_epochs}")
    print(f"  Dataset: cifar10, Alpha: {alpha}")
    print("=" * 70)

    # Load data
    train_dataset, test_dataset = load_cifar10()

    print(f"Partitioning data (Non-IID, Dirichlet α={alpha})...")
    client_indices = partition_data_dirichlet(train_dataset, num_clients, alpha=alpha)

    # Determine Byzantine clients
    num_byzantine = int(num_clients * byzantine_ratio)
    byzantine_ids = set(range(num_byzantine)) if num_byzantine > 0 else set()
    print(f"Byzantine clients: {num_byzantine}/{num_clients} ({byz_pct}%)")
    if byzantine_ids:
        print(f"Byzantine client IDs: {sorted(byzantine_ids)}")

    # Create attack instance
    if num_byzantine == 0 or attack_name == "none":
        attack = NoAttack()
    elif attack_name == "sign_flipping":
        attack = SignFlippingAttack()
    elif attack_name == "random_noise":
        attack = RandomNoiseAttack(noise_scale=attack_params.get("noise_scale", 5.0))
    elif attack_name == "alie":
        attack = ALIEAttack(num_byzantine=num_byzantine, num_clients=num_clients)
    elif attack_name in ["ipm_small", "ipm_large", "ipm"]:
        epsilon = attack_params.get("epsilon", 1.0)
        attack = IPMAttack(epsilon=epsilon)
    elif attack_name == "label_flipping":
        attack = LabelFlippingAttack()
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    # Test loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initial model and parameters
    model = SimpleCNN().to(device)
    initial_params = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])

    # Evaluation function
    def evaluate_fn(parameters):
        params = parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, 100.0 * correct / total

    # Create strategy
    strategy_kwargs = {
        "initial_parameters": initial_params,
        "num_clients": num_clients,
        "byzantine_ids": byzantine_ids,
        "evaluate_fn": evaluate_fn,
        "compromise_round": compromise_round,
    }

    if strategy_name == "fedavg":
        fl_strategy = FedAvgStrategy(**strategy_kwargs)
    elif strategy_name == "fedmedian":
        fl_strategy = FedMedianStrategy(**strategy_kwargs)
    elif strategy_name == "trimmed":
        fl_strategy = TrimmedMeanStrategy(trim_ratio=0.1, **strategy_kwargs)
    elif strategy_name == "krum":
        fl_strategy = KrumStrategy(num_byzantine=num_byzantine, **strategy_kwargs)
    elif strategy_name == "multikrum":
        fl_strategy = MultiKrumStrategy(num_byzantine=num_byzantine, **strategy_kwargs)
    elif strategy_name == "geomed":
        fl_strategy = GeoMedStrategy(**strategy_kwargs)
    elif strategy_name == "cc":
        fl_strategy = CenteredClippingStrategy(tau=0.1, **strategy_kwargs)
    elif strategy_name == "clustering":
        fl_strategy = ClusteringStrategy(**strategy_kwargs)
    elif strategy_name == "clippedclustering":
        fl_strategy = ClippedClusteringStrategy(**strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from: {ALL_STRATEGIES}")

    # Client function
    def client_fn(cid: str):
        cid_int = int(cid)
        is_byzantine = cid_int in byzantine_ids

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(client_indices[cid_int]),
            num_workers=2,
        )

        client_model = SimpleCNN(num_classes=10)

        return BaselineFlowerClient(
            model=client_model,
            trainloader=train_loader,
            device=device,
            client_id=cid_int,
            is_byzantine=is_byzantine,
            attack=attack if is_byzantine else NoAttack(),
            compromise_round=compromise_round,
            attack_end_round=attack_end_round,  # Novel: attack window end
        ).to_client()

    # Run simulation
    print(f"\nStarting {strategy_name} simulation with Flower...")

    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": False,
    }

    gpu_fraction = 0.04 if device.type == "cuda" else 0.0

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl_strategy,
        client_resources={"num_cpus": 1, "num_gpus": gpu_fraction},
        ray_init_args=ray_init_args,
    )

    # Collect results
    metrics = fl_strategy.get_metrics()
    detection_stats = fl_strategy.get_detection_stats()

    results = {
        "experiment": exp_name,
        "strategy": strategy_name,
        "attack": attack_name,
        "byzantine_ratio": byzantine_ratio,
        "byzantine_ids": list(byzantine_ids),
        "scenario": scenario,
        "compromise_round": compromise_round,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "alpha": alpha,
        "seed": seed,
        "final_accuracy": metrics["test_accuracy"][-1] if metrics["test_accuracy"] else 0,
        "final_loss": metrics["test_loss"][-1] if metrics["test_loss"] else 0,
        "metrics": metrics,
        "detection_stats": detection_stats,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{exp_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(output_dir, f"{exp_name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "test_accuracy", "test_loss"])
        for i, r in enumerate(metrics["rounds"]):
            writer.writerow([
                r,
                metrics["test_accuracy"][i] if i < len(metrics["test_accuracy"]) else "",
                metrics["test_loss"][i] if i < len(metrics["test_loss"]) else "",
            ])

    print(f"\nResults saved to {json_path}")
    print(f"Final accuracy: {results['final_accuracy']:.2f}%")

    if detection_stats.get('detection') != 'N/A':
        print(f"Detection - P: {detection_stats.get('precision', 'N/A')}, R: {detection_stats.get('recall', 'N/A')}, F1: {detection_stats.get('f1', 'N/A')}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Level 4: Baseline Byzantine-Robust FL Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Novel Experimental Modes (Beyond Li et al. 2024):
  --timing_study     Run compromise timing experiments (when attacks start)
  --window_study     Run attack window experiments (transient attacks)
  --alpha_study      Run Non-IID severity experiments (varying alpha)

Examples:
  # Single experiment with attack window
  python run_flower_experiments.py --strategy krum --attack sign_flipping \\
      --byzantine_ratio 0.2 --compromise_round 10 --attack_end_round 30

  # Comprehensive timing study
  python run_flower_experiments.py --timing_study --all_strategies

  # Attack window study with specific strategy
  python run_flower_experiments.py --window_study --strategy clippedclustering
        """
    )

    # Basic parameters
    parser.add_argument("--strategy", type=str, default="fedavg",
                        choices=ALL_STRATEGIES,
                        help="Aggregation strategy")
    parser.add_argument("--attack", type=str, default="none",
                        choices=["none", "sign_flipping", "random_noise", "alie",
                                 "ipm_small", "ipm_large", "label_flipping"],
                        help="Attack type")
    parser.add_argument("--byzantine_ratio", type=float, default=0.0,
                        help="Fraction of Byzantine clients")

    # Attack timing parameters (NOVEL)
    parser.add_argument("--compromise_round", type=int, default=0,
                        help="Round at which attacks begin (0=immediate)")
    parser.add_argument("--attack_end_round", type=int, default=None,
                        help="Round at which attacks end (None=until training ends)")

    # Batch experiment modes
    parser.add_argument("--all_attacks", action="store_true",
                        help="Run all attack configurations")
    parser.add_argument("--all_strategies", action="store_true",
                        help="Run all aggregation strategies")
    parser.add_argument("--all_scenarios", action="store_true",
                        help="Run immediate and delayed scenarios")

    # Novel study modes (BEYOND Li et al. 2024)
    parser.add_argument("--timing_study", action="store_true",
                        help="Run compromise TIMING study (when attacks start)")
    parser.add_argument("--window_study", action="store_true",
                        help="Run attack WINDOW study (transient attacks)")
    parser.add_argument("--alpha_study", action="store_true",
                        help="Run Non-IID SEVERITY study (varying alpha)")

    # Training parameters
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--num_clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--local_epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results/flower")
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU to use")

    args = parser.parse_args()

    # Setup device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine strategies to run
    if args.all_strategies:
        strategies = ALL_STRATEGIES
    else:
        strategies = [args.strategy]

    # =========================================================================
    # NOVEL STUDY MODE: Compromise Timing Study
    # =========================================================================
    if args.timing_study:
        print("=" * 70)
        print("COMPROMISE TIMING STUDY (Novel contribution)")
        print("Testing when attacks start: rounds", COMPROMISE_TIMINGS)
        print("=" * 70)

        for strategy_name in strategies:
            for attack_name, byz_ratio, attack_params in FOCUSED_ATTACK_CONFIGS:
                for timing in COMPROMISE_TIMINGS:
                    run_experiment(
                        strategy_name=strategy_name,
                        attack_name=attack_name,
                        byzantine_ratio=byz_ratio,
                        compromise_round=timing,
                        attack_end_round=None,
                        attack_params=attack_params,
                        num_rounds=args.num_rounds,
                        num_clients=args.num_clients,
                        local_epochs=args.local_epochs,
                        alpha=args.alpha,
                        seed=args.seed,
                        output_dir=args.output_dir + "/timing_study",
                        device=device,
                    )
        return

    # =========================================================================
    # NOVEL STUDY MODE: Attack Window Study
    # =========================================================================
    if args.window_study:
        print("=" * 70)
        print("ATTACK WINDOW STUDY (Novel contribution)")
        print("Testing transient attack windows:", ATTACK_WINDOWS)
        print("=" * 70)

        for strategy_name in strategies:
            for attack_name, byz_ratio, attack_params in FOCUSED_ATTACK_CONFIGS:
                for start_round, end_round in ATTACK_WINDOWS:
                    run_experiment(
                        strategy_name=strategy_name,
                        attack_name=attack_name,
                        byzantine_ratio=byz_ratio,
                        compromise_round=start_round,
                        attack_end_round=end_round,
                        attack_params=attack_params,
                        num_rounds=args.num_rounds,
                        num_clients=args.num_clients,
                        local_epochs=args.local_epochs,
                        alpha=args.alpha,
                        seed=args.seed,
                        output_dir=args.output_dir + "/window_study",
                        device=device,
                    )
        return

    # =========================================================================
    # NOVEL STUDY MODE: Non-IID Severity Study
    # =========================================================================
    if args.alpha_study:
        print("=" * 70)
        print("NON-IID SEVERITY STUDY (Novel contribution)")
        print("Testing Dirichlet alpha values:", ALPHA_VALUES)
        print("=" * 70)

        for strategy_name in strategies:
            for attack_name, byz_ratio, attack_params in FOCUSED_ATTACK_CONFIGS:
                for alpha_val in ALPHA_VALUES:
                    run_experiment(
                        strategy_name=strategy_name,
                        attack_name=attack_name,
                        byzantine_ratio=byz_ratio,
                        compromise_round=0,  # Immediate for alpha study
                        attack_end_round=None,
                        attack_params=attack_params,
                        num_rounds=args.num_rounds,
                        num_clients=args.num_clients,
                        local_epochs=args.local_epochs,
                        alpha=alpha_val,
                        seed=args.seed,
                        output_dir=args.output_dir + "/alpha_study",
                        device=device,
                    )
        return

    # =========================================================================
    # Standard experiment mode (compatible with Li et al. baseline)
    # =========================================================================

    # Determine attack configurations
    if args.all_attacks:
        attacks = ATTACK_CONFIGS
    else:
        attacks = [(args.attack, args.byzantine_ratio, {})]

    # Determine scenarios
    if args.all_scenarios:
        scenarios = [(0, None), (DEFAULT_COMPROMISE_ROUND, None)]
    else:
        scenarios = [(args.compromise_round, args.attack_end_round)]

    # Run experiments
    total = len(strategies) * len(attacks) * len(scenarios)
    print(f"Running up to {total} experiments")
    print(f"Strategies: {strategies}")
    print(f"Attack configs: {len(attacks)}")
    print(f"Scenarios: {scenarios}")

    for strategy_name in strategies:
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'='*70}")

        for attack_name, byz_ratio, attack_params in attacks:
            for compromise_round, attack_end_round in scenarios:
                # Skip delayed scenario for no-attack baseline
                if attack_name == "none" and compromise_round > 0:
                    continue

                run_experiment(
                    strategy_name=strategy_name,
                    attack_name=attack_name,
                    byzantine_ratio=byz_ratio,
                    compromise_round=compromise_round,
                    attack_end_round=attack_end_round,
                    attack_params=attack_params,
                    num_rounds=args.num_rounds,
                    num_clients=args.num_clients,
                    local_epochs=args.local_epochs,
                    alpha=args.alpha,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    device=device,
                )

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
