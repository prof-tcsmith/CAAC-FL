#!/usr/bin/env python3
"""
Level 5a: CAAC-FL Experiments using Flower Framework

This script runs CAAC-FL experiments using the same infrastructure as level3,
enabling proper parallel execution via Ray and direct comparison with other
aggregation strategies.

Uses:
- Flower (flwr) for FL simulation
- Ray for parallel client execution on multiple GPUs
- Same experimental setup as level3 for fair comparison

================================================================================
DELAYED COMPROMISE THREAT MODEL
================================================================================

This experiment framework supports two threat models:

1. **Immediate Attack (Traditional)**: compromise_round=0
   - Byzantine clients attack from round 0
   - Profiles are built on malicious behavior
   - Profile-based detection may not work well (malicious = baseline)

2. **Delayed Compromise (Recommended)**: compromise_round > 0
   - All clients behave honestly during warmup (rounds 0 to compromise_round-1)
   - CAAC-FL builds profiles based on legitimate behavior
   - At compromise_round, Byzantine clients begin attacking
   - Profile-based detection should catch the behavioral deviation

The delayed compromise model is more realistic because:
- Real compromises happen after initial deployment
- Defenders have time to establish behavioral baselines
- Profile-based detection has historical honest data to compare against

Recommended configuration:
- compromise_round >= warmup_rounds (default: 15 with warmup_rounds=10)
- This ensures profiles are well-established before attacks begin

================================================================================
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
import numpy as np

import flwr as fl
from flwr.common import Context

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.models import SimpleCNN, create_model_for_dataset
from shared.data_utils import load_fashion_mnist, load_cifar10, partition_data_dirichlet, analyze_data_distribution
from shared.metrics import evaluate_model, MetricsLogger

# Import level3 client and attacks
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../level3_attacks')))
from client import create_client
from attacks import create_attack, get_available_attacks

# Import CAAC-FL strategy
from caacfl_strategy import CAACFLStrategy, create_caacfl_strategy


# ============================================================================
# Configuration
# ============================================================================

STRATEGIES = ["caacfl", "fedavg", "fedmedian", "fedtrimmed"]
ATTACKS = ["none", "sign_flipping", "random_noise", "alie"]
BYZANTINE_RATIOS = [0.0, 0.1, 0.2, 0.3]

# Default parameters matching level3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default compromise round for delayed compromise threat model
# Should be >= warmup_rounds to allow profile building
DEFAULT_COMPROMISE_ROUND = 15


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Level 5a: CAAC-FL Experiments with Flower',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Threat Model Options:
  --compromise_round 0    Immediate attack (traditional, attacks from round 0)
  --compromise_round 15   Delayed compromise (recommended, profiles built first)

Examples:
  # Run with delayed compromise (recommended for CAAC-FL)
  python run_flower_experiments.py --strategy caacfl --attack random_noise \\
      --byzantine_ratio 0.3 --compromise_round 15

  # Run with immediate attack (traditional threat model)
  python run_flower_experiments.py --strategy caacfl --attack random_noise \\
      --byzantine_ratio 0.3 --compromise_round 0

  # Run all attacks with delayed compromise
  python run_flower_experiments.py --strategy caacfl --all_attacks \\
      --compromise_round 15 --dataset cifar10
        """
    )

    # Experiment selection
    parser.add_argument('--strategy', type=str, choices=STRATEGIES, default='caacfl',
                        help='Aggregation strategy to test')
    parser.add_argument('--attack', type=str, choices=ATTACKS, default='none',
                        help='Attack type')
    parser.add_argument('--byzantine_ratio', type=float, default=0.2,
                        help='Fraction of Byzantine clients (default: 0.2)')
    parser.add_argument('--all_attacks', action='store_true',
                        help='Run all attacks for the specified strategy')
    parser.add_argument('--full_study', action='store_true',
                        help='Run complete study (all strategies x attacks x ratios)')

    # Threat model configuration
    parser.add_argument('--compromise_round', type=int, default=DEFAULT_COMPROMISE_ROUND,
                        help=f'Round at which Byzantine attacks begin (default: {DEFAULT_COMPROMISE_ROUND}). '
                             'Set to 0 for immediate attack (traditional model). '
                             'Set to >0 for delayed compromise (recommended >= warmup_rounds).')

    # Experiment parameters
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=25,
                        help='Total number of clients')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Local training epochs per round')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID data')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['fashion_mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/flower',
                        help='Output directory for results')

    return parser.parse_args()


def run_experiment(
    strategy: str,
    attack: str,
    byzantine_ratio: float,
    num_rounds: int,
    num_clients: int,
    local_epochs: int,
    alpha: float,
    dataset: str,
    seed: int,
    output_dir: str,
    compromise_round: int = DEFAULT_COMPROMISE_ROUND,
) -> dict:
    """
    Run a single FL experiment with specified parameters.

    Supports the delayed compromise threat model where Byzantine clients
    behave honestly until compromise_round, then begin attacking.

    Args:
        strategy: Aggregation strategy ('caacfl', 'fedavg', etc.)
        attack: Attack type ('none', 'sign_flipping', 'random_noise', 'alie')
        byzantine_ratio: Fraction of Byzantine clients
        num_rounds: Number of FL rounds
        num_clients: Total number of clients
        local_epochs: Local training epochs per round
        alpha: Dirichlet alpha for non-IID partitioning
        dataset: Dataset name
        seed: Random seed
        output_dir: Output directory
        compromise_round: Round at which Byzantine attacks begin (default: 15)
            - 0: Immediate attack (traditional threat model)
            - >0: Delayed compromise (Byzantine clients honest until this round)

    Returns:
        Dictionary containing experiment results
    """

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Experiment naming (include threat model indicator)
    threat_model = "delayed" if compromise_round > 0 else "immediate"
    experiment_name = f"{strategy}_{attack}_byz{int(byzantine_ratio*100)}_seed{seed}"
    if compromise_round > 0:
        experiment_name += f"_comp{compromise_round}"

    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"  Strategy: {strategy}")
    print(f"  Attack: {attack}, Byzantine ratio: {byzantine_ratio*100:.0f}%")
    print(f"  Threat Model: {threat_model} (compromise_round={compromise_round})")
    print(f"  Rounds: {num_rounds}, Local epochs: {local_epochs}")
    print(f"  Dataset: {dataset}, Alpha: {alpha}")
    print(f"{'='*70}")

    # Load data
    if dataset == 'fashion_mnist':
        train_dataset, test_dataset = load_fashion_mnist(data_dir='./data')
        num_classes = 10
    else:
        train_dataset, test_dataset = load_cifar10(data_dir='./data')
        num_classes = 10

    # Partition data (Non-IID with Dirichlet)
    print(f"Partitioning data (Non-IID, Dirichlet α={alpha})...")
    client_dict = partition_data_dirichlet(
        train_dataset, num_clients, alpha=alpha, seed=seed
    )

    # Analyze distribution
    stats = analyze_data_distribution(train_dataset, client_dict)
    print(f"Data heterogeneity (KL): {stats['heterogeneity_metrics']['mean_kl_divergence']:.4f}")

    # Test loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Identify Byzantine clients
    NUM_BYZANTINE = int(num_clients * byzantine_ratio)
    byzantine_clients = set(range(NUM_BYZANTINE))
    print(f"Byzantine clients: {NUM_BYZANTINE}/{num_clients} ({byzantine_ratio*100:.0f}%)")
    if compromise_round > 0 and NUM_BYZANTINE > 0:
        print(f"  → Honest until round {compromise_round}, then attacking")

    # Client factory function with delayed compromise support
    def client_fn(context: Context):
        """Create a Flower client for simulation with delayed compromise support"""
        client_id = int(context.node_config.get("partition-id", context.node_id % num_clients))

        is_byzantine = client_id in byzantine_clients

        # Create attack instance if Byzantine
        attack_instance = None
        if is_byzantine and attack != "none":
            attack_kwargs = {"seed": seed + client_id}
            if attack == "alie":
                attack_kwargs["num_byzantine"] = NUM_BYZANTINE
                attack_kwargs["num_clients"] = num_clients
            elif attack == "random_noise":
                attack_kwargs["noise_scale"] = 1.0

            attack_instance = create_attack(attack, **attack_kwargs)

        # Get client data
        client_indices = client_dict[client_id]
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(client_indices),
            num_workers=4
        )

        # Create client with delayed compromise support
        return create_client(
            client_id=client_id,
            trainloader=train_loader,
            testloader=test_loader,
            device=DEVICE,
            is_byzantine=is_byzantine,
            attack=attack_instance,
            local_epochs=local_epochs,
            learning_rate=LEARNING_RATE,
            compromise_round=compromise_round,  # Delayed compromise parameter
        ).to_client()

    # Metrics logger
    os.makedirs(output_dir, exist_ok=True)
    logger = MetricsLogger(log_dir=output_dir, experiment_name=experiment_name)

    # Track timing and results
    start_time = time.time()
    round_accuracies = []
    detection_stats = {'total_detections': 0, 'detections_per_round': []}

    # Evaluation function
    def evaluate_fn(server_round, parameters, config):
        model = create_model_for_dataset(dataset).to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        result = evaluate_model(model, test_loader, DEVICE, criterion=torch.nn.CrossEntropyLoss())
        accuracy = result['accuracy']
        loss = result['loss']

        round_accuracies.append(accuracy)
        logger.log_round(round_num=server_round, test_accuracy=accuracy, test_loss=loss)

        elapsed = time.time() - start_time
        if server_round > 0:
            eta = (elapsed / server_round) * (num_rounds - server_round)
            # Add marker when compromise begins
            marker = " [ATTACK BEGINS]" if server_round == compromise_round and NUM_BYZANTINE > 0 else ""
            print(f"Round {server_round:2d}/{num_rounds}: Acc={accuracy:.2f}%, "
                  f"Loss={loss:.4f} | ETA: {timedelta(seconds=int(eta))}{marker}")
        else:
            print(f"Round {server_round:2d}: Acc={accuracy:.2f}%, Loss={loss:.4f}")

        return loss, {"accuracy": accuracy}

    # Metrics aggregation
    def fit_metrics_aggregation_fn(metrics):
        if not metrics:
            return {}
        total = sum([n for n, _ in metrics])
        return {k: sum([m.get(k, 0) * n for n, m in metrics]) / total
                for k in ['loss', 'client_id', 'is_byzantine', 'attack_applied']}

    # Create strategy
    if strategy == "caacfl":
        # Pass byzantine_ids for proper TP/FP/TN/FN tracking
        fl_strategy = create_caacfl_strategy(
            num_clients=num_clients,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            byzantine_ids=list(byzantine_clients),  # For confusion matrix tracking
            # CAAC-FL algorithm parameters (v2 tuned values)
            alpha=0.05,             # Slower EWMA updates to resist profile poisoning
            gamma=0.1,              # Reliability update rate
            tau_base=1.2,           # Lower threshold to catch more anomalies
            beta=0.5,               # Threshold flexibility factor
            window_size=5,          # History window for analysis
            weights=(0.5, 0.3, 0.2),  # Prioritize magnitude for random noise detection
            # Cold-start mitigation parameters
            warmup_rounds=10,       # Longer conservative period
            warmup_factor=0.3,      # Stricter during warmup
            min_rounds_for_trust=5, # Longer trust building period
            use_cross_comparison=True,   # Enable cross-client comparison
            use_population_init=True,    # Initialize profiles from population stats
            new_client_weight=0.3,       # Less influence for new clients
        )
    elif strategy == "fedavg":
        fl_strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=num_clients, min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
    elif strategy == "fedmedian":
        fl_strategy = fl.server.strategy.FedMedian(
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=num_clients, min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
    elif strategy == "fedtrimmed":
        fl_strategy = fl.server.strategy.FedTrimmedAvg(
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=num_clients, min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            beta=0.2,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Ray configuration for parallel execution
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": 64,
        "num_gpus": 2 if DEVICE.type == "cuda" else 0,
        "_memory": 50 * 1024 * 1024 * 1024,
        "object_store_memory": 100 * 1024 * 1024 * 1024,
    }

    # Run simulation
    # Use 0.04 GPU per client (matches level3 configuration)
    print(f"\nStarting {strategy} simulation with Flower...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.04 if DEVICE.type == "cuda" else 0.0},
        ray_init_args=ray_init_args
    )

    # Get detection stats if CAAC-FL
    if strategy == "caacfl" and hasattr(fl_strategy, 'get_detection_stats'):
        raw_stats = fl_strategy.get_detection_stats()
        # Convert to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj
        detection_stats = make_serializable(raw_stats)

    # Save results
    logger.save_csv()
    logger.save_json()

    total_time = time.time() - start_time
    final_accuracy = round_accuracies[-1] if round_accuracies else 0.0
    best_accuracy = max(round_accuracies) if round_accuracies else 0.0

    # Calculate accuracy at compromise point (for delayed compromise analysis)
    accuracy_at_compromise = None
    if compromise_round > 0 and compromise_round < len(round_accuracies):
        accuracy_at_compromise = round_accuracies[compromise_round]

    # Create result summary
    result = {
        "experiment": experiment_name,
        "strategy": strategy,
        "attack": attack,
        "byzantine_ratio": byzantine_ratio,
        "num_clients": num_clients,
        "num_byzantine": NUM_BYZANTINE,
        "byzantine_client_ids": list(byzantine_clients),
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "dataset": dataset,
        "alpha": alpha,
        "seed": seed,
        # Threat model configuration
        "threat_model": threat_model,
        "compromise_round": compromise_round,
        # Results
        "final_accuracy": final_accuracy,
        "best_accuracy": best_accuracy,
        "accuracy_at_compromise": accuracy_at_compromise,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "round_accuracies": round_accuracies,
        "detection_stats": detection_stats if strategy == "caacfl" else None,
        "data_heterogeneity_kl": stats['heterogeneity_metrics']['mean_kl_divergence'],
    }

    # Save individual result
    result_file = os.path.join(output_dir, f"{experiment_name}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"  Threat Model: {threat_model} (compromise at round {compromise_round})")
    print(f"  Final Accuracy: {final_accuracy:.2f}%")
    print(f"  Best Accuracy: {best_accuracy:.2f}%")
    if accuracy_at_compromise is not None:
        print(f"  Accuracy at Compromise: {accuracy_at_compromise:.2f}%")
    print(f"  Total Time: {timedelta(seconds=int(total_time))}")
    if strategy == "caacfl":
        print(f"  Total Detections: {detection_stats.get('total_detections', 0)}")
        if 'total_tp' in detection_stats:
            tp = detection_stats.get('total_tp', 0)
            fp = detection_stats.get('total_fp', 0)
            fn = detection_stats.get('total_fn', 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  Detection: TP={tp}, FP={fp}, FN={fn} | Precision={precision:.2%}, Recall={recall:.2%}")
    print(f"{'='*70}")

    return result


def main():
    args = parse_args()

    if args.full_study:
        # Run complete study
        print("Running FULL STUDY: All strategies x attacks x ratios")
        print(f"Threat model: {'delayed' if args.compromise_round > 0 else 'immediate'} "
              f"(compromise_round={args.compromise_round})")
        all_results = []

        for strategy in STRATEGIES:
            for attack in ATTACKS:
                for byz_ratio in BYZANTINE_RATIOS:
                    # Skip invalid combinations
                    if attack == 'none' and byz_ratio > 0:
                        continue
                    if attack != 'none' and byz_ratio == 0:
                        continue

                    result = run_experiment(
                        strategy=strategy,
                        attack=attack,
                        byzantine_ratio=byz_ratio,
                        num_rounds=args.num_rounds,
                        num_clients=args.num_clients,
                        local_epochs=args.local_epochs,
                        alpha=args.alpha,
                        dataset=args.dataset,
                        seed=args.seed,
                        output_dir=args.output_dir,
                        compromise_round=args.compromise_round,
                    )
                    all_results.append(result)

        # Save combined results
        combined_file = os.path.join(args.output_dir, "full_study_results.json")
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull study results saved to: {combined_file}")

    elif args.all_attacks:
        # Run all attacks for specified strategy
        print(f"Running all attacks for strategy: {args.strategy}")
        print(f"Threat model: {'delayed' if args.compromise_round > 0 else 'immediate'} "
              f"(compromise_round={args.compromise_round})")
        all_results = []

        for attack in ATTACKS:
            for byz_ratio in BYZANTINE_RATIOS:
                if attack == 'none' and byz_ratio > 0:
                    continue
                if attack != 'none' and byz_ratio == 0:
                    continue

                result = run_experiment(
                    strategy=args.strategy,
                    attack=attack,
                    byzantine_ratio=byz_ratio,
                    num_rounds=args.num_rounds,
                    num_clients=args.num_clients,
                    local_epochs=args.local_epochs,
                    alpha=args.alpha,
                    dataset=args.dataset,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    compromise_round=args.compromise_round,
                )
                all_results.append(result)

        # Save combined results
        combined_file = os.path.join(args.output_dir, f"{args.strategy}_all_attacks.json")
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    else:
        # Run single experiment
        run_experiment(
            strategy=args.strategy,
            attack=args.attack,
            byzantine_ratio=args.byzantine_ratio,
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            local_epochs=args.local_epochs,
            alpha=args.alpha,
            dataset=args.dataset,
            seed=args.seed,
            output_dir=args.output_dir,
            compromise_round=args.compromise_round,
        )


if __name__ == "__main__":
    main()
