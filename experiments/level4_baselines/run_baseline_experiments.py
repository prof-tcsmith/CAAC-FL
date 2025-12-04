#!/usr/bin/env python3
"""
Level 4: Baseline Byzantine-Robust Aggregation Strategies

This module implements baseline Byzantine-robust FL strategies for comparison:
- FedAvg: Weighted average (non-robust baseline)
- FedMedian: Coordinate-wise median
- Krum: Distance-based selection (single best client)
- Multi-Krum: Distance-based selection (top-k clients)
- Trimmed Mean: Remove extreme values before averaging

Uses identical configuration to level5a_caacfl for fair comparison.
"""

import sys
import os
import argparse
import json
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directories for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'level3_attacks')))

from shared.models import SimpleCNN
from shared.data_utils import load_cifar10, load_mnist, load_fashion_mnist, partition_data_dirichlet
from attacks import create_attack, get_available_attacks

# ============================================================================
# Configuration (matches level5a_caacfl)
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.01


# ============================================================================
# Byzantine-Robust Strategy Implementations
# ============================================================================

class BaseRobustStrategy(Strategy):
    """Base class for robust aggregation strategies"""

    def __init__(
        self,
        initial_parameters: Parameters,
        num_clients: int,
        byzantine_ids: set,
        attack_fn,
        evaluate_fn,
        strategy_name: str = "base",
    ):
        self.initial_parameters = initial_parameters
        self.num_clients = num_clients
        self.byzantine_ids = byzantine_ids
        self.attack_fn = attack_fn
        self.evaluate_fn = evaluate_fn
        self.strategy_name = strategy_name
        self.current_round = 0

        # Store global model weights
        self.global_weights = parameters_to_ndarrays(initial_parameters)

        # Metrics tracking
        self.metrics_history = {
            "rounds": [],
            "test_accuracy": [],
            "test_loss": [],
            "train_accuracy": [],
            "train_loss": [],
        }
        self.start_time = None

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_round = server_round
        if self.start_time is None:
            self.start_time = time.time()

        config = {"local_epochs": 5, "learning_rate": LEARNING_RATE}

        # Sample all clients
        sample_size = min(self.num_clients, client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)

        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {}
        sample_size = min(self.num_clients, client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_examples = sum([r.num_examples for _, r in results])
        weighted_loss = sum([r.num_examples * r.loss for _, r in results]) / total_examples
        weighted_acc = sum([r.num_examples * r.metrics.get("accuracy", 0) for _, r in results]) / total_examples

        return weighted_loss, {"accuracy": weighted_acc}

    def _apply_attacks(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[Tuple[ClientProxy, FitRes]]:
        """Apply Byzantine attacks to malicious client updates"""
        if not self.attack_fn or not self.byzantine_ids:
            return results

        attacked_results = []
        for client, fit_res in results:
            cid = int(client.cid)

            if cid in self.byzantine_ids:
                # Get client weights
                client_weights = parameters_to_ndarrays(fit_res.parameters)

                # Compute pseudo-gradient: gradient = global - client (since client = global - lr*grad)
                pseudo_gradient = [g - c for g, c in zip(self.global_weights, client_weights)]

                # Apply attack to gradient
                attacked_gradient = self.attack_fn(pseudo_gradient)

                # Reconstruct attacked weights
                attacked_weights = [g - ag for g, ag in zip(self.global_weights, attacked_gradient)]

                # Create new FitRes with attacked parameters
                attacked_params = ndarrays_to_parameters(attacked_weights)
                new_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=attacked_params,
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                attacked_results.append((client, new_fit_res))
            else:
                attacked_results.append((client, fit_res))

        return attacked_results

    def _log_round(self, server_round: int, loss: float, accuracy: float):
        """Log round progress"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        if server_round > 0:
            time_per_round = elapsed / server_round
            remaining_rounds = 50 - server_round  # Assuming 50 rounds
            eta = timedelta(seconds=int(time_per_round * remaining_rounds))
        else:
            eta = "calculating..."

        print(f"Round {server_round:2d}/50: Acc={accuracy:.2f}%, Loss={loss:.4f} | ETA: {eta}")

        # Store metrics
        self.metrics_history["rounds"].append(server_round)
        self.metrics_history["test_accuracy"].append(accuracy)
        self.metrics_history["test_loss"].append(loss)

    def evaluate(self, server_round, parameters):
        """Server-side evaluation"""
        if self.evaluate_fn is None:
            return None

        loss, accuracy = self.evaluate_fn(parameters)
        self._log_round(server_round, loss, accuracy)

        return loss, {"accuracy": accuracy}

    def get_metrics(self) -> Dict:
        """Return collected metrics"""
        return self.metrics_history


class FedAvgStrategy(BaseRobustStrategy):
    """Standard FedAvg - weighted average (non-robust baseline)"""

    def __init__(self, **kwargs):
        super().__init__(strategy_name="fedavg", **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Apply Byzantine attacks
        results = self._apply_attacks(results)

        # Weighted average
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


class FedMedianStrategy(BaseRobustStrategy):
    """FedMedian - coordinate-wise median (Byzantine-robust)"""

    def __init__(self, **kwargs):
        super().__init__(strategy_name="fedmedian", **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Apply Byzantine attacks
        results = self._apply_attacks(results)

        # Stack all client weights
        all_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Coordinate-wise median
        aggregated_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in all_weights], axis=0)
            layer_median = np.median(layer_stack, axis=0)
            aggregated_weights.append(layer_median)

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class KrumStrategy(BaseRobustStrategy):
    """Krum - select single client closest to others (Byzantine-robust)"""

    def __init__(self, num_byzantine: int = 0, **kwargs):
        super().__init__(strategy_name="krum", **kwargs)
        self.num_byzantine = num_byzantine

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Apply Byzantine attacks
        results = self._apply_attacks(results)

        n = len(results)
        f = self.num_byzantine  # Known/estimated number of Byzantine clients

        # Get all client weights as flat vectors
        all_weights = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            all_weights.append(flat)

        all_weights = np.array(all_weights)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(all_weights[i] - all_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, sum distances to n - f - 2 closest clients
        scores = []
        k = max(1, n - f - 2)  # Number of closest neighbors to consider
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:k+1])  # Exclude self (distance 0)
            scores.append(score)

        # Select client with lowest score
        best_idx = np.argmin(scores)

        # Use the selected client's weights
        selected_weights = parameters_to_ndarrays(results[best_idx][1].parameters)

        self.global_weights = selected_weights
        return ndarrays_to_parameters(selected_weights), {}


class MultiKrumStrategy(BaseRobustStrategy):
    """Multi-Krum - average top-k clients closest to others"""

    def __init__(self, num_byzantine: int = 0, num_select: int = None, **kwargs):
        super().__init__(strategy_name="multikrum", **kwargs)
        self.num_byzantine = num_byzantine
        self.num_select = num_select  # How many clients to select for averaging

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Apply Byzantine attacks
        results = self._apply_attacks(results)

        n = len(results)
        f = self.num_byzantine
        m = self.num_select if self.num_select else max(1, n - f)  # Default: select n-f clients

        # Get all client weights as flat vectors
        all_weights = []
        weights_structured = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            all_weights.append(flat)
            weights_structured.append(weights)

        all_weights = np.array(all_weights)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(all_weights[i] - all_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores
        scores = []
        k = max(1, n - f - 2)
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:k+1])
            scores.append(score)

        # Select top-m clients with lowest scores
        selected_indices = np.argsort(scores)[:m]

        # Average selected clients
        aggregated_weights = None
        for idx in selected_indices:
            if aggregated_weights is None:
                aggregated_weights = [w.copy() for w in weights_structured[idx]]
            else:
                for i, w in enumerate(weights_structured[idx]):
                    aggregated_weights[i] += w

        for i in range(len(aggregated_weights)):
            aggregated_weights[i] /= len(selected_indices)

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


class TrimmedMeanStrategy(BaseRobustStrategy):
    """Trimmed Mean - remove extreme values before averaging"""

    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        super().__init__(strategy_name="trimmed", **kwargs)
        self.trim_ratio = trim_ratio  # Fraction to trim from each end

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Apply Byzantine attacks
        results = self._apply_attacks(results)

        n = len(results)
        trim_count = int(n * self.trim_ratio)

        # Stack all client weights
        all_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Trimmed mean for each layer
        aggregated_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_stack = np.stack([w[layer_idx] for w in all_weights], axis=0)

            if trim_count > 0 and n > 2 * trim_count:
                # Sort along client axis and trim
                sorted_stack = np.sort(layer_stack, axis=0)
                trimmed = sorted_stack[trim_count:n-trim_count]
                layer_mean = np.mean(trimmed, axis=0)
            else:
                # Not enough clients to trim, use regular mean
                layer_mean = np.mean(layer_stack, axis=0)

            aggregated_weights.append(layer_mean)

        self.global_weights = aggregated_weights
        return ndarrays_to_parameters(aggregated_weights), {}


# ============================================================================
# Flower Client
# ============================================================================

class FlowerClient(fl.client.NumPyClient):
    """Standard FL client for baseline experiments"""

    def __init__(self, cid: int, train_loader: DataLoader, device: torch.device):
        self.cid = cid
        self.train_loader = train_loader
        self.device = device
        self.model = SimpleCNN().to(device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        local_epochs = config.get("local_epochs", 5)
        lr = config.get("learning_rate", LEARNING_RATE)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, total, {"accuracy": 100.0 * correct / total}


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(
    strategy_name: str,
    attack_name: str,
    byzantine_ratio: float,
    num_rounds: int = 50,
    num_clients: int = 25,
    local_epochs: int = 5,
    dataset: str = "cifar10",
    alpha: float = 0.5,
    seed: int = 42,
    output_dir: str = "./results/flower",
):
    """Run a single baseline experiment"""

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Experiment name
    byz_pct = int(byzantine_ratio * 100)
    exp_name = f"{strategy_name}_{attack_name}_byz{byz_pct}_seed{seed}"

    print("\n" + "=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Attack: {attack_name}, Byzantine ratio: {byz_pct}%")
    print(f"  Rounds: {num_rounds}, Local epochs: {local_epochs}")
    print(f"  Dataset: {dataset}, Alpha: {alpha}")
    print("=" * 70)

    # Load dataset
    if dataset == "cifar10":
        train_dataset, test_dataset = load_cifar10()
    elif dataset == "mnist":
        train_dataset, test_dataset = load_mnist()
    elif dataset == "fashion_mnist":
        train_dataset, test_dataset = load_fashion_mnist()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Partition data
    print("Partitioning data (Non-IID, Dirichlet α={})...".format(alpha))
    client_indices = partition_data_dirichlet(train_dataset, num_clients, alpha=alpha)

    # Determine Byzantine clients
    num_byzantine = int(num_clients * byzantine_ratio)
    byzantine_ids = set(range(num_byzantine)) if num_byzantine > 0 else set()
    print(f"Byzantine clients: {num_byzantine}/{num_clients} ({byz_pct}%)")

    # Create attack function
    attack_fn = None
    if attack_name != "none" and num_byzantine > 0:
        attack_fn = create_attack(attack_name, num_byzantine=num_byzantine, num_clients=num_clients)

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create initial model and parameters
    model = SimpleCNN().to(DEVICE)
    initial_params = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])

    # Create evaluation function
    def evaluate_fn(parameters):
        params = parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
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
        "attack_fn": attack_fn,
        "evaluate_fn": evaluate_fn,
    }

    if strategy_name == "fedavg":
        fl_strategy = FedAvgStrategy(**strategy_kwargs)
    elif strategy_name == "fedmedian":
        fl_strategy = FedMedianStrategy(**strategy_kwargs)
    elif strategy_name == "krum":
        fl_strategy = KrumStrategy(num_byzantine=num_byzantine, **strategy_kwargs)
    elif strategy_name == "multikrum":
        fl_strategy = MultiKrumStrategy(num_byzantine=num_byzantine, **strategy_kwargs)
    elif strategy_name == "trimmed":
        fl_strategy = TrimmedMeanStrategy(trim_ratio=0.1, **strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Create client function
    def client_fn(cid: str):
        cid_int = int(cid)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(client_indices[cid_int]),
            num_workers=4,
        )
        return FlowerClient(cid_int, train_loader, DEVICE).to_client()

    # Ray initialization
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": False,
    }

    # Run simulation
    print(f"\nStarting {strategy_name} simulation with Flower...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.04 if DEVICE.type == "cuda" else 0.0},
        ray_init_args=ray_init_args,
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    metrics = fl_strategy.get_metrics()

    # JSON output
    results = {
        "experiment": exp_name,
        "strategy": strategy_name,
        "attack": attack_name,
        "byzantine_ratio": byzantine_ratio,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "dataset": dataset,
        "alpha": alpha,
        "seed": seed,
        "final_accuracy": metrics["test_accuracy"][-1] if metrics["test_accuracy"] else 0,
        "final_loss": metrics["test_loss"][-1] if metrics["test_loss"] else 0,
        "metrics": metrics,
    }

    json_path = os.path.join(output_dir, f"{exp_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV metrics output
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

    return results


def main():
    parser = argparse.ArgumentParser(description="Level 4: Baseline Byzantine-Robust Strategies")

    parser.add_argument("--strategy", type=str, default="fedavg",
                        choices=["fedavg", "fedmedian", "krum", "multikrum", "trimmed"],
                        help="Aggregation strategy")
    parser.add_argument("--attack", type=str, default="none",
                        choices=["none", "sign_flipping", "random_noise", "alie"],
                        help="Attack type")
    parser.add_argument("--byzantine_ratio", type=float, default=0.0,
                        help="Fraction of Byzantine clients")
    parser.add_argument("--all_attacks", action="store_true",
                        help="Run all attack configurations")
    parser.add_argument("--all_strategies", action="store_true",
                        help="Run all strategies")
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=25)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results/flower")

    args = parser.parse_args()

    strategies = ["fedavg", "fedmedian", "krum", "multikrum", "trimmed"] if args.all_strategies else [args.strategy]

    if args.all_attacks:
        # Same configuration as level5a_caacfl
        attacks = [
            ("none", 0.0),
            ("sign_flipping", 0.1),
            ("sign_flipping", 0.2),
            ("sign_flipping", 0.3),
            ("random_noise", 0.1),
            ("random_noise", 0.2),
            ("random_noise", 0.3),
            ("alie", 0.1),
            ("alie", 0.2),
            ("alie", 0.3),
        ]
    else:
        attacks = [(args.attack, args.byzantine_ratio)]

    total_experiments = len(strategies) * len(attacks)
    print(f"Running {total_experiments} experiments...")

    for strategy in strategies:
        print(f"\nRunning all attacks for strategy: {strategy}")
        for attack_name, byz_ratio in attacks:
            run_experiment(
                strategy_name=strategy,
                attack_name=attack_name,
                byzantine_ratio=byz_ratio,
                num_rounds=args.num_rounds,
                num_clients=args.num_clients,
                local_epochs=args.local_epochs,
                dataset=args.dataset,
                alpha=args.alpha,
                seed=args.seed,
                output_dir=args.output_dir,
            )

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
