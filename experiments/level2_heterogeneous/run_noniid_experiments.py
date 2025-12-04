#!/usr/bin/env python3
"""
Unified experiment runner for Non-IID (heterogeneous) aggregation comparison study.

Extends Level 1 baseline study to explore how aggregation strategies perform
under label heterogeneity using Dirichlet-based partitioning.

Supports:
- Multiple datasets: MNIST, Fashion-MNIST, CIFAR-10
- Multiple strategies: FedAvg, FedMean, FedMedian
- Non-IID data via Dirichlet distribution (configurable alpha)
- Multiple runs with different random seeds for statistical analysis
"""

import sys
import os

# Add parent directory to path for both main process and Ray workers
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import argparse
import time
import json
from datetime import timedelta
import warnings

# Suppress PyTorch pin_memory deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')

import torch
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg, FedMedian, FedAdam, FedTrimmedAvg
from torch.utils.data import DataLoader
from collections import OrderedDict

from shared.data_utils import (
    load_dataset, partition_data_dirichlet, partition_data_dirichlet_equal,
    create_dataloaders, analyze_data_distribution, DATASET_INFO
)

# Conditions for 2x2 factorial design with Level 1
CONDITIONS = ['noniid_equal', 'noniid_unequal']
from shared.models import create_model_for_dataset
from shared.metrics import MetricsLogger, evaluate_model
from level1_fundamentals.fedmean_strategy import FedMean
from level2_heterogeneous.krum_strategy import Krum


# Fixed random seeds for reproducibility across 5 runs
SEEDS = [42, 123, 456, 789, 1011]

# Experiment configurations
WEIGHT_STRATEGIES = ['fedavg', 'fedmean', 'fedmedian']
GRADIENT_STRATEGIES = ['fedsgd', 'fedadam', 'fedtrimmed']
BYZANTINE_STRATEGIES = ['krum', 'multikrum']  # Byzantine-robust methods
ALL_STRATEGIES = WEIGHT_STRATEGIES + GRADIENT_STRATEGIES + BYZANTINE_STRATEGIES
STRATEGIES = WEIGHT_STRATEGIES  # Default to weight strategies for backward compatibility
DATASETS = ['mnist', 'fashion_mnist', 'cifar10']

# Client scaling configurations (matching Level 1)
CLIENT_COUNTS = [10, 25, 50]

# Dirichlet alpha values for non-IID partitioning
# Lower alpha = more heterogeneous
ALPHA_VALUES = {
    'high_heterogeneity': 0.1,   # Very non-IID
    'moderate': 0.5,              # Moderately non-IID (default)
    'mild': 1.0,                  # Mildly non-IID
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Non-IID Aggregation Comparison Experiments (Level 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment with default alpha=0.5
  python run_noniid_experiments.py --dataset mnist --strategy fedavg --seed 42

  # Run with specific alpha value
  python run_noniid_experiments.py --dataset mnist --strategy fedavg --seed 42 --alpha 0.1

  # Run all experiments for a dataset (all strategies, all seeds, alpha=0.5)
  python run_noniid_experiments.py --dataset mnist --all

  # Run all experiments across all datasets
  python run_noniid_experiments.py --all-datasets

  # Run all alpha values for a dataset
  python run_noniid_experiments.py --dataset mnist --all-alpha
        """
    )

    # Single experiment mode
    parser.add_argument('--dataset', type=str, choices=DATASETS,
                        help='Dataset to use')
    parser.add_argument('--strategy', type=str, choices=ALL_STRATEGIES,
                        help='Aggregation strategy')
    parser.add_argument('--gradient-strategies', action='store_true',
                        help='Run gradient-based strategies (fedsgd, fedadam, fedtrimmed)')
    parser.add_argument('--byzantine-strategies', action='store_true',
                        help='Run Byzantine-robust strategies (krum, multikrum)')
    parser.add_argument('--all-strategies', action='store_true',
                        help='Run all strategies (weight + gradient + byzantine)')
    parser.add_argument('--seed', type=int, choices=SEEDS,
                        help='Random seed for this run')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha (lower=more heterogeneous, default: 0.5)')
    parser.add_argument('--condition', type=str, choices=CONDITIONS,
                        default='noniid_unequal',
                        help='noniid_equal (forced equal sizes) or noniid_unequal (natural sizes)')

    # Batch mode
    parser.add_argument('--all', action='store_true',
                        help='Run all strategies/seeds for specified dataset (alpha=0.5)')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run complete experiment suite (all datasets, alpha=0.5)')
    parser.add_argument('--all-alpha', action='store_true',
                        help='Run all alpha values for specified dataset')
    parser.add_argument('--all-conditions', action='store_true',
                        help='Run both noniid_equal and noniid_unequal conditions')
    parser.add_argument('--all-clients', action='store_true',
                        help='Run all client counts (10, 25, 50) for client scaling study')
    parser.add_argument('--client-scaling', action='store_true',
                        help='Run full client scaling study (all conditions × all client counts)')

    # Hyperparameters (matching Level 1 for comparability)
    parser.add_argument('--num_clients', type=int, default=50,
                        help='Number of clients (default: 50)')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of FL rounds (default: 50)')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local epochs per round (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')

    # Output
    parser.add_argument('--results_dir', type=str, default='./results/noniid',
                        help='Directory for results')

    return parser.parse_args()


class FlowerClient(fl.client.NumPyClient):
    """Generic Flower client for any dataset."""

    def __init__(self, cid, train_loader, test_loader, model, device,
                 learning_rate, local_epochs):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                'train_loss': total_loss / total,
                'train_accuracy': 100.0 * correct / total
            }
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return (
            total_loss / total,
            total,
            {'accuracy': 100.0 * correct / total}
        )


def run_experiment(dataset_name, strategy_name, alpha, seed, condition,
                   num_clients, num_rounds, local_epochs, batch_size,
                   learning_rate, results_dir):
    """
    Run a single non-IID experiment.

    Args:
        condition: 'noniid_equal' (forced equal sample sizes) or
                   'noniid_unequal' (natural Dirichlet sizes)

    Returns:
        dict: Experiment results including final accuracy, loss, and trajectory
    """
    print("=" * 70)
    print(f"EXPERIMENT: {dataset_name} | {strategy_name} | {condition} | alpha={alpha} | seed={seed}")
    print("=" * 70)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Condition: {condition}")
    print(f"  Dirichlet alpha: {alpha}")
    print(f"  Seed: {seed}")
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Device: {device}")

    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    train_dataset, test_dataset = load_dataset(dataset_name)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Partition data using Dirichlet (non-IID)
    print(f"\nPartitioning data ({condition}, alpha={alpha})...")
    if condition == 'noniid_equal':
        # Non-IID labels with forced equal sample counts
        client_dict = partition_data_dirichlet_equal(
            train_dataset, num_clients, alpha=alpha, seed=seed
        )
    else:
        # Non-IID labels with natural (unequal) sample counts
        client_dict = partition_data_dirichlet(
            train_dataset, num_clients, alpha=alpha, seed=seed
        )

    # Analyze distribution
    stats = analyze_data_distribution(train_dataset, client_dict)
    client_sizes = list(stats['client_sizes'].values())
    heterogeneity = stats['heterogeneity_metrics']['mean_kl_divergence']
    print(f"  Client sizes: min={min(client_sizes)}, max={max(client_sizes)}, "
          f"mean={np.mean(client_sizes):.0f}")
    print(f"  Heterogeneity (mean KL divergence): {heterogeneity:.4f}")

    # Create dataloaders
    train_loaders = create_dataloaders(
        train_dataset, client_dict, batch_size=batch_size, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Create model
    model = create_model_for_dataset(dataset_name)
    criterion = torch.nn.CrossEntropyLoss()

    # Track metrics
    metrics = {
        'dataset': dataset_name,
        'strategy': strategy_name,
        'condition': condition,
        'alpha': alpha,
        'seed': seed,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'local_epochs': local_epochs,
        'test_accuracy': [],
        'test_loss': [],
        'client_sizes': client_sizes,
        'heterogeneity_kl': heterogeneity,
        'class_distribution': stats['class_distribution'],
    }

    experiment_start_time = time.time()

    # Evaluation function
    def evaluate_fn(server_round, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        results = evaluate_model(model, test_loader, device=device, criterion=criterion)

        metrics['test_accuracy'].append(results['accuracy'])
        metrics['test_loss'].append(results['loss'])

        elapsed = time.time() - experiment_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if server_round > 0:
            avg_time = elapsed / server_round
            remaining = avg_time * (num_rounds - server_round)
            remaining_str = str(timedelta(seconds=int(remaining)))
            print(f"Round {server_round}/{num_rounds}: Acc={results['accuracy']:.2f}%, "
                  f"Loss={results['loss']:.4f} | Elapsed: {elapsed_str}, ETA: {remaining_str}")

        return results['loss'], {'accuracy': results['accuracy']}

    # Metrics aggregation
    def fit_metrics_aggregation_fn(client_metrics):
        total_examples = sum([n for n, _ in client_metrics])
        aggregated = {}
        for metric_name in ['train_loss', 'train_accuracy']:
            weighted_sum = sum([m.get(metric_name, 0) * n for n, m in client_metrics])
            aggregated[metric_name] = weighted_sum / total_examples if total_examples > 0 else 0
        return aggregated

    # Client factory - imports done inside to ensure Ray workers have them
    def client_fn(context):
        # Ensure imports are available in Ray workers
        import sys
        import os
        exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if exp_dir not in sys.path:
            sys.path.insert(0, exp_dir)

        from shared.models import create_model_for_dataset as create_model

        client_id = int(context.node_config.get("partition-id", context.node_id))
        client_model = create_model(dataset_name)
        return FlowerClient(
            cid=client_id,
            train_loader=train_loaders[client_id],
            test_loader=test_loader,
            model=client_model,
            device=device,
            learning_rate=learning_rate,
            local_epochs=local_epochs
        ).to_client()

    # Create strategy
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )

    strategy_kwargs = {
        'fraction_fit': 1.0,
        'fraction_evaluate': 0.0,
        'min_fit_clients': num_clients,
        'min_evaluate_clients': 0,
        'min_available_clients': num_clients,
        'evaluate_fn': evaluate_fn,
        'fit_metrics_aggregation_fn': fit_metrics_aggregation_fn,
        'initial_parameters': initial_parameters,
    }

    if strategy_name == 'fedavg':
        strategy = FedAvg(**strategy_kwargs)
    elif strategy_name == 'fedmean':
        strategy = FedMean(**strategy_kwargs)
    elif strategy_name == 'fedmedian':
        strategy = FedMedian(**strategy_kwargs)
    elif strategy_name == 'fedsgd':
        # FedSGD is FedAvg - difference is local_epochs=1 (set in client)
        strategy = FedAvg(**strategy_kwargs)
    elif strategy_name == 'fedadam':
        # Server-side Adam optimizer
        strategy = FedAdam(
            **strategy_kwargs,
            eta=0.1,      # Server learning rate
            eta_l=learning_rate,  # Client learning rate
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-9,
        )
    elif strategy_name == 'fedtrimmed':
        # Trimmed mean - removes 20% from each tail
        strategy = FedTrimmedAvg(**strategy_kwargs, beta=0.2)
    elif strategy_name == 'krum':
        # Standard Krum - selects single client closest to neighbors
        # For honest clients (no Byzantine), set num_byzantine=0
        strategy = Krum(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            num_byzantine=0,  # No Byzantine clients in this experiment
            num_selected=1,   # Standard Krum: select 1 client
        )
    elif strategy_name == 'multikrum':
        # Multi-Krum - selects and averages n-f-2 clients
        # Better for non-IID as it captures diverse distributions
        num_selected = max(1, num_clients - 0 - 2)  # n - f - 2, with f=0
        strategy = Krum(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            num_byzantine=0,
            num_selected=num_selected,  # Multi-Krum: average multiple clients
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Ray configuration with runtime environment for proper imports
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": min(64, os.cpu_count() or 8),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "runtime_env": {
            "working_dir": EXPERIMENTS_DIR,
            "excludes": [
                "**/data/**",
                "**/results/**",
                "**/*.tar.gz",
                "**/*.zip",
                "**/*.png",
                "**/*.html",
                "**/*.pdf",
            ],
        },
    }

    # Run simulation
    print(f"\nStarting simulation...")
    print("-" * 70)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            'num_cpus': 1,
            'num_gpus': 0.04 if device.type == 'cuda' else 0
        },
        ray_init_args=ray_init_args,
    )

    print("-" * 70)

    # Finalize metrics
    total_time = time.time() - experiment_start_time
    metrics['total_time_seconds'] = total_time
    metrics['final_accuracy'] = metrics['test_accuracy'][-1]
    metrics['final_loss'] = metrics['test_loss'][-1]
    metrics['best_accuracy'] = max(metrics['test_accuracy'])

    print(f"\nExperiment Complete!")
    print(f"  Final Accuracy: {metrics['final_accuracy']:.2f}%")
    print(f"  Best Accuracy: {metrics['best_accuracy']:.2f}%")
    print(f"  Total Time: {str(timedelta(seconds=int(total_time)))}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    # Create organized subdirectory structure: results/<condition>/<dataset>/c<num_clients>/alpha_X_X/
    alpha_str = f"alpha_{alpha}".replace('.', '_')
    client_str = f"c{num_clients}"
    exp_dir = os.path.join(results_dir, condition, dataset_name, client_str, alpha_str)
    os.makedirs(exp_dir, exist_ok=True)

    filename = f"{strategy_name}_seed{seed}"

    # Save JSON
    json_path = os.path.join(exp_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results saved to: {json_path}")

    return metrics


def run_all_for_dataset(dataset_name, alpha, condition, args, strategies=None):
    """Run all experiments for a single dataset with specified alpha and condition."""
    if strategies is None:
        strategies = STRATEGIES

    results = []
    total_experiments = len(strategies) * len(SEEDS)
    current = 0

    for strategy in strategies:
        for seed in SEEDS:
            current += 1
            print(f"\n{'#' * 70}")
            print(f"# EXPERIMENT {current}/{total_experiments}")
            print(f"{'#' * 70}")

            result = run_experiment(
                dataset_name=dataset_name,
                strategy_name=strategy,
                alpha=alpha,
                seed=seed,
                condition=condition,
                num_clients=args.num_clients,
                num_rounds=args.num_rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                results_dir=args.results_dir,
            )
            results.append(result)

    return results


def main():
    args = parse_args()

    # Determine which strategies to run
    if args.all_strategies:
        strategies_to_run = ALL_STRATEGIES
    elif args.gradient_strategies:
        strategies_to_run = GRADIENT_STRATEGIES
    elif args.byzantine_strategies:
        strategies_to_run = BYZANTINE_STRATEGIES
    else:
        strategies_to_run = WEIGHT_STRATEGIES

    if args.client_scaling:
        # Run full client scaling study (all conditions × all client counts × all datasets)
        all_results = []
        total_experiments = len(CONDITIONS) * len(CLIENT_COUNTS) * len(DATASETS) * len(strategies_to_run) * len(SEEDS)
        print(f"\n{'=' * 70}")
        print(f"FULL CLIENT SCALING STUDY")
        print(f"Conditions: {CONDITIONS}")
        print(f"Client counts: {CLIENT_COUNTS}")
        print(f"Strategies: {strategies_to_run}")
        print(f"Total experiments: {total_experiments}")
        print(f"{'=' * 70}")

        for condition in CONDITIONS:
            for num_clients in CLIENT_COUNTS:
                for dataset in DATASETS:
                    print(f"\n{'=' * 70}")
                    print(f"STARTING: {dataset.upper()} | {condition} | {num_clients} clients")
                    print(f"{'=' * 70}")
                    # Temporarily override num_clients
                    original_num_clients = args.num_clients
                    args.num_clients = num_clients
                    results = run_all_for_dataset(dataset, args.alpha, condition, args, strategies_to_run)
                    args.num_clients = original_num_clients
                    all_results.extend(results)

        print(f"\n{'=' * 70}")
        print("CLIENT SCALING STUDY COMPLETE")
        print(f"Total experiments: {len(all_results)}")
        print(f"{'=' * 70}")

    elif args.all_clients:
        # Run all client counts for specified condition and datasets
        all_results = []
        for num_clients in CLIENT_COUNTS:
            for dataset in DATASETS:
                print(f"\n{'=' * 70}")
                print(f"STARTING: {dataset.upper()} | {args.condition} | {num_clients} clients")
                print(f"{'=' * 70}")
                original_num_clients = args.num_clients
                args.num_clients = num_clients
                results = run_all_for_dataset(dataset, args.alpha, args.condition, args, strategies_to_run)
                args.num_clients = original_num_clients
                all_results.extend(results)

        print(f"\n{'=' * 70}")
        print("ALL CLIENT COUNTS COMPLETE")
        print(f"Total experiments: {len(all_results)}")
        print(f"{'=' * 70}")

    elif args.all_conditions:
        # Run both noniid_equal and noniid_unequal for all datasets
        all_results = []
        for condition in CONDITIONS:
            for dataset in DATASETS:
                print(f"\n{'=' * 70}")
                print(f"STARTING: {dataset.upper()} | {condition} (alpha={args.alpha})")
                print(f"{'=' * 70}")
                results = run_all_for_dataset(dataset, args.alpha, condition, args, strategies_to_run)
                all_results.extend(results)

        print(f"\n{'=' * 70}")
        print("ALL CONDITIONS COMPLETE")
        print(f"Total experiments: {len(all_results)}")
        print(f"{'=' * 70}")

    elif args.all_datasets:
        # Run complete experiment suite (all datasets, specified condition)
        all_results = []
        for dataset in DATASETS:
            print(f"\n{'=' * 70}")
            print(f"STARTING: {dataset.upper()} | {args.condition} (alpha={args.alpha})")
            print(f"{'=' * 70}")
            results = run_all_for_dataset(dataset, args.alpha, args.condition, args, strategies_to_run)
            all_results.extend(results)

        print(f"\n{'=' * 70}")
        print("ALL EXPERIMENTS COMPLETE")
        print(f"Total experiments: {len(all_results)}")
        print(f"{'=' * 70}")

    elif args.all_alpha:
        # Run all alpha values for specified dataset
        if not args.dataset:
            print("Error: --dataset required when using --all-alpha")
            sys.exit(1)

        all_results = []
        for alpha_name, alpha in ALPHA_VALUES.items():
            print(f"\n{'=' * 70}")
            print(f"STARTING: {args.dataset.upper()} | {alpha_name} (alpha={alpha})")
            print(f"{'=' * 70}")
            results = run_all_for_dataset(args.dataset, alpha, args.condition, args, strategies_to_run)
            all_results.extend(results)

        print(f"\n{'=' * 70}")
        print("ALL ALPHA EXPERIMENTS COMPLETE")
        print(f"Total experiments: {len(all_results)}")
        print(f"{'=' * 70}")

    elif args.all:
        # Run all for specified dataset with default alpha
        if not args.dataset:
            print("Error: --dataset required when using --all")
            sys.exit(1)
        run_all_for_dataset(args.dataset, args.alpha, args.condition, args, strategies_to_run)

    else:
        # Run single experiment
        if not all([args.dataset, args.strategy, args.seed]):
            print("Error: For single experiment, provide --dataset, --strategy, and --seed")
            print("Or use --all with --dataset, or --all-datasets")
            sys.exit(1)

        run_experiment(
            dataset_name=args.dataset,
            strategy_name=args.strategy,
            alpha=args.alpha,
            seed=args.seed,
            condition=args.condition,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
