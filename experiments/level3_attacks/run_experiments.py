#!/usr/bin/env python3
"""
Level 3: Unified Experiment Runner for Byzantine Attack Study

This script runs comprehensive experiments testing aggregation strategies
against various Byzantine attacks. It supports:

Aggregation Strategies:
- FedAvg: Weighted average (baseline, non-robust)
- FedMedian: Coordinate-wise median (Byzantine-robust)
- FedTrimmedAvg: Trimmed mean (Byzantine-robust)

Attacks:
- none: No attack (baseline)
- random_noise: Gaussian noise injection
- sign_flipping: Gradient sign reversal
- alie: A Little Is Enough (detection evasion)
- ipm: Inner Product Manipulation (steering attack)
- label_flipping: Simulated label flip attack

Based on findings from Level 1/2:
- FedAvg and FedSGD are equivalent, so we only test FedAvg
- FedAdam fails catastrophically, excluded from robustness study
- FedMedian showed best robustness in preliminary tests
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
import flwr as fl
from flwr.common import Context

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.models import SimpleCNN
from shared.data_utils import load_cifar10, partition_data_dirichlet, analyze_data_distribution
from shared.metrics import evaluate_model, MetricsLogger
from client import create_client
from attacks import create_attack, get_available_attacks, ATTACK_INFO


# Experiment configuration
STRATEGIES = ["fedavg", "fedmedian", "fedtrimmed"]
ATTACKS = ["none", "random_noise", "sign_flipping", "alie", "ipm"]
BYZANTINE_RATIOS = [0.1, 0.2, 0.3]  # 10%, 20%, 30% Byzantine
DATA_DISTRIBUTIONS = ["iid", "noniid"]  # IID and Non-IID


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Level 3: Byzantine Attack Robustness Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python run_experiments.py --strategy fedavg --attack sign_flipping --byzantine_ratio 0.2

  # Run all experiments for a strategy
  python run_experiments.py --strategy fedmedian --all_attacks

  # Run full study (all strategies x all attacks x all ratios)
  python run_experiments.py --full_study

  # Quick test with fewer rounds
  python run_experiments.py --strategy fedavg --attack none --num_rounds 10
        """
    )

    # Experiment selection
    parser.add_argument('--strategy', type=str, choices=STRATEGIES,
                        help='Aggregation strategy to test')
    parser.add_argument('--attack', type=str, choices=get_available_attacks(),
                        help='Attack type')
    parser.add_argument('--byzantine_ratio', type=float, default=0.2,
                        help='Fraction of Byzantine clients (default: 0.2)')
    parser.add_argument('--all_attacks', action='store_true',
                        help='Run all attacks for the specified strategy')
    parser.add_argument('--full_study', action='store_true',
                        help='Run complete study (all strategies x attacks x ratios)')

    # Experiment parameters
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=25,
                        help='Total number of clients')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Local epochs per round')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID data')
    parser.add_argument('--iid', action='store_true',
                        help='Use IID data distribution (default: non-IID)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/paper',
                        help='Output directory for results')

    return parser.parse_args()


def run_single_experiment(
    strategy: str,
    attack: str,
    byzantine_ratio: float,
    num_rounds: int,
    num_clients: int,
    local_epochs: int,
    alpha: float,
    seed: int,
    output_dir: str,
    iid: bool = False,
) -> Dict:
    """
    Run a single FL experiment with specified parameters.

    Returns:
        Dictionary with experiment results and metrics
    """
    # Configuration
    NUM_BYZANTINE = int(num_clients * byzantine_ratio)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dist = "iid" if iid else "noniid"
    experiment_name = f"{strategy}_{attack}_byz{int(byzantine_ratio*100)}_{data_dist}"

    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print("=" * 70)
    print(f"  Strategy: {strategy}")
    print(f"  Attack: {attack} ({ATTACK_INFO.get(attack, {}).get('description', 'N/A')})")
    print(f"  Byzantine: {NUM_BYZANTINE}/{num_clients} ({byzantine_ratio*100:.0f}%)")
    print(f"  Rounds: {num_rounds}, Local Epochs: {local_epochs}")
    print(f"  Data: {'IID' if iid else f'Non-IID (α={alpha})'}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(seed)

    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    train_dataset, test_dataset = load_cifar10()

    # Partition data
    if iid:
        print("Partitioning data (IID - uniform random)...")
        # IID partitioning: randomly shuffle and split evenly
        import numpy as np
        np.random.seed(seed)
        indices = np.random.permutation(len(train_dataset))
        split_size = len(train_dataset) // num_clients
        client_dict = {
            i: indices[i * split_size:(i + 1) * split_size].tolist()
            for i in range(num_clients)
        }
        # Handle remainder
        remainder = len(train_dataset) % num_clients
        if remainder > 0:
            for i, idx in enumerate(indices[-remainder:]):
                client_dict[i].append(idx)
    else:
        print(f"Partitioning data (Non-IID, Dirichlet α={alpha})...")
        client_dict = partition_data_dirichlet(
            train_dataset, num_clients, alpha=alpha, seed=seed
        )

    # Analyze distribution
    stats = analyze_data_distribution(train_dataset, client_dict)
    print(f"Heterogeneity KL: {stats['heterogeneity_metrics']['mean_kl_divergence']:.4f}")

    # Byzantine clients (first NUM_BYZANTINE)
    byzantine_clients = list(range(NUM_BYZANTINE))

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Client factory
    def client_fn(context: Context):
        client_id = int(context.node_config.get("partition-id", context.node_id))
        is_byzantine = client_id in byzantine_clients

        # Create attack for Byzantine clients
        attack_instance = None
        if is_byzantine and attack != "none":
            attack_kwargs = {"seed": seed + client_id}
            if attack == "alie":
                attack_kwargs["num_byzantine"] = NUM_BYZANTINE
                attack_kwargs["num_clients"] = num_clients
            elif attack == "random_noise":
                attack_kwargs["noise_scale"] = 1.0
            elif attack == "ipm":
                attack_kwargs["epsilon"] = 1.0

            attack_instance = create_attack(attack, **attack_kwargs)

        # Get client data
        client_indices = client_dict[client_id]
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(client_indices),
            num_workers=4
        )

        return create_client(
            client_id=client_id,
            trainloader=train_loader,
            testloader=test_loader,
            device=DEVICE,
            is_byzantine=is_byzantine,
            attack=attack_instance,
            local_epochs=local_epochs,
            learning_rate=LEARNING_RATE,
        ).to_client()

    # Metrics logger
    os.makedirs(output_dir, exist_ok=True)
    logger = MetricsLogger(log_dir=output_dir, experiment_name=experiment_name)

    # Track experiment timing
    start_time = time.time()
    round_accuracies = []

    # Evaluation function
    def evaluate_fn(server_round, parameters, config):
        model = SimpleCNN(num_classes=10).to(DEVICE)
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
            print(f"Round {server_round:2d}/{num_rounds}: Acc={accuracy:.2f}%, "
                  f"Loss={loss:.4f} | ETA: {timedelta(seconds=int(eta))}")
        else:
            print(f"Round {server_round:2d}: Acc={accuracy:.2f}%, Loss={loss:.4f}")

        return loss, {"accuracy": accuracy}

    # Metrics aggregation
    def fit_metrics_aggregation_fn(metrics):
        if not metrics:
            return {}
        total = sum([n for n, _ in metrics])
        return {k: sum([m.get(k, 0) * n for n, m in metrics]) / total
                for k in ['loss', 'client_id', 'is_byzantine']}

    # Create strategy
    if strategy == "fedavg":
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
            beta=0.2,  # Trim 20% from each tail
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Ray configuration
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": 64,
        "num_gpus": 2 if DEVICE.type == "cuda" else 0,
        "_memory": 50 * 1024 * 1024 * 1024,
        "object_store_memory": 100 * 1024 * 1024 * 1024,
    }

    # Run simulation
    print(f"\nStarting {strategy} simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.04 if DEVICE.type == "cuda" else 0.0},
        ray_init_args=ray_init_args
    )

    # Save results
    logger.save_csv()
    logger.save_json()

    total_time = time.time() - start_time
    final_accuracy = round_accuracies[-1] if round_accuracies else 0.0
    best_accuracy = max(round_accuracies) if round_accuracies else 0.0

    # Create result summary
    result = {
        "experiment": experiment_name,
        "strategy": strategy,
        "attack": attack,
        "byzantine_ratio": byzantine_ratio,
        "num_clients": num_clients,
        "num_byzantine": NUM_BYZANTINE,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "data_distribution": "iid" if iid else "noniid",
        "alpha": alpha if not iid else None,
        "seed": seed,
        "final_accuracy": final_accuracy,
        "best_accuracy": best_accuracy,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "round_accuracies": round_accuracies,
    }

    # Save result JSON
    result_file = Path(output_dir) / f"{experiment_name}_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Total Time: {timedelta(seconds=int(total_time))}")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")

    return result


def run_full_study(args):
    """Run the complete study with all combinations (IID and non-IID)"""
    results = []
    total_experiments = len(STRATEGIES) * len(ATTACKS) * len(BYZANTINE_RATIOS) * len(DATA_DISTRIBUTIONS)
    current = 0

    print(f"\n{'#'*70}")
    print(f"# FULL STUDY: {total_experiments} experiments")
    print(f"# Strategies: {STRATEGIES}")
    print(f"# Attacks: {ATTACKS}")
    print(f"# Byzantine Ratios: {BYZANTINE_RATIOS}")
    print(f"# Data Distributions: {DATA_DISTRIBUTIONS}")
    print(f"# Clients: {args.num_clients}")
    print(f"{'#'*70}\n")

    for data_dist in DATA_DISTRIBUTIONS:
        is_iid = (data_dist == "iid")
        for strategy in STRATEGIES:
            for attack in ATTACKS:
                for byz_ratio in BYZANTINE_RATIOS:
                    current += 1
                    print(f"\n[{current}/{total_experiments}] {data_dist.upper()} | {strategy} + {attack} + {byz_ratio*100:.0f}% Byzantine")

                    try:
                        result = run_single_experiment(
                            strategy=strategy,
                            attack=attack,
                            byzantine_ratio=byz_ratio,
                            num_rounds=args.num_rounds,
                            num_clients=args.num_clients,
                            local_epochs=args.local_epochs,
                            alpha=args.alpha,
                            seed=args.seed,
                            output_dir=args.output_dir,
                            iid=is_iid,
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"ERROR in experiment: {e}")
                        import traceback
                        traceback.print_exc()
                        results.append({
                            "strategy": strategy,
                            "attack": attack,
                            "byzantine_ratio": byz_ratio,
                            "data_distribution": data_dist,
                            "error": str(e),
                        })

    # Save summary
    summary_file = Path(args.output_dir) / "full_study_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# STUDY COMPLETE")
    print(f"# Results: {len([r for r in results if 'error' not in r])}/{total_experiments} successful")
    print(f"# Summary saved to: {summary_file}")
    print(f"{'#'*70}")

    return results


def main():
    args = parse_args()

    if args.full_study:
        run_full_study(args)
    elif args.all_attacks and args.strategy:
        # Run all attacks for specified strategy
        results = []
        for attack in ATTACKS:
            result = run_single_experiment(
                strategy=args.strategy,
                attack=attack,
                byzantine_ratio=args.byzantine_ratio,
                num_rounds=args.num_rounds,
                num_clients=args.num_clients,
                local_epochs=args.local_epochs,
                alpha=args.alpha,
                seed=args.seed,
                output_dir=args.output_dir,
                iid=args.iid,
            )
            results.append(result)

        # Save summary
        data_dist = "iid" if args.iid else "noniid"
        summary_file = Path(args.output_dir) / f"{args.strategy}_all_attacks_{data_dist}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif args.strategy and args.attack:
        # Run single experiment
        run_single_experiment(
            strategy=args.strategy,
            attack=args.attack,
            byzantine_ratio=args.byzantine_ratio,
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            local_epochs=args.local_epochs,
            alpha=args.alpha,
            seed=args.seed,
            output_dir=args.output_dir,
            iid=args.iid,
        )
    else:
        print("Please specify either:")
        print("  --strategy and --attack for single experiment")
        print("  --strategy and --all_attacks for all attacks on one strategy")
        print("  --full_study for complete study (IID + non-IID)")
        sys.exit(1)


if __name__ == "__main__":
    main()
