"""
Flexible Level 1 experiment runner supporting:
- Different partitioning: IID-Equal, IID-Unequal
- Different aggregation: FedAvg, FedMean, FedMedian
- Different client counts: 10, 25, 50, 100
"""

import sys
import os
import argparse
import time
from datetime import timedelta
import warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress PyTorch pin_memory deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')

import torch
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg, FedMedian
from torch.utils.data import DataLoader

from shared.data_utils import (load_cifar10, partition_data_iid,
                               partition_data_iid_unequal, create_dataloaders)
from shared.models import SimpleCNN
from shared.metrics import MetricsLogger, evaluate_model
from client import create_client_fn
from fedmean_strategy import FedMean


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Flexible Level 1 Experiment Runner')
    parser.add_argument('--partition', type=str, default='iid-equal',
                       choices=['iid-equal', 'iid-unequal'],
                       help='Data partitioning strategy')
    parser.add_argument('--aggregation', type=str, default='fedavg',
                       choices=['fedavg', 'fedmean', 'fedmedian'],
                       help='Aggregation strategy')
    parser.add_argument('--num_clients', type=int, default=50,
                       help='Total number of clients')
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of rounds')
    parser.add_argument('--size_variation', type=float, default=0.5,
                       help='Size variation for iid-unequal (0.5=moderate, 1.0=high)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Level 1: {args.aggregation.upper()} with {args.partition.upper()}")
    print("=" * 60)

    # Configuration
    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    LOCAL_EPOCHS = 5
    SEED = args.seed
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nConfiguration:")
    print(f"  Partition: {args.partition}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Local epochs: {LOCAL_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    if args.partition == 'iid-unequal':
        print(f"  Size variation: {args.size_variation}")

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_dataset, test_dataset = load_cifar10()
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Partition data based on strategy
    print(f"\nPartitioning data ({args.partition})...")
    if args.partition == 'iid-equal':
        client_dict = partition_data_iid(train_dataset, NUM_CLIENTS, seed=SEED)
    elif args.partition == 'iid-unequal':
        client_dict = partition_data_iid_unequal(
            train_dataset, NUM_CLIENTS,
            size_variation=args.size_variation,
            seed=SEED
        )

    # Create dataloaders
    train_loaders = create_dataloaders(
        train_dataset,
        client_dict,
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # Print dataset statistics
    client_sizes = [len(loader.dataset) for loader in train_loaders.values()]
    print(f"  Dataset size statistics:")
    print(f"    Min: {min(client_sizes)}")
    print(f"    Max: {max(client_sizes)}")
    print(f"    Mean: {np.mean(client_sizes):.1f}")
    print(f"    Std: {np.std(client_sizes):.1f}")
    if args.partition == 'iid-unequal':
        print(f"    Coefficient of Variation: {np.std(client_sizes)/np.mean(client_sizes):.3f}")

    # Create client factory
    client_fn = create_client_fn(
        train_loaders=train_loaders,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        local_epochs=LOCAL_EPOCHS
    )

    # Initialize model for server-side evaluation
    model = SimpleCNN(num_classes=10)
    model.to(DEVICE)

    # Initialize metrics logger
    experiment_name = f"level1_{args.partition}_{args.aggregation}_c{NUM_CLIENTS}"
    logger = MetricsLogger(
        log_dir=args.output_dir,
        experiment_name=experiment_name
    )

    # Timing
    experiment_start_time = time.time()

    def evaluate_fn(server_round, parameters, config):
        """Server-side evaluation function"""
        # Update model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Evaluate on test set
        results = evaluate_model(
            model,
            test_loader,
            device=DEVICE,
            criterion=torch.nn.CrossEntropyLoss()
        )

        # Log metrics
        logger.log_round(
            round_num=server_round,
            test_acc=results['accuracy'],
            test_loss=results['loss']
        )

        # Print progress with timing
        elapsed_time = time.time() - experiment_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))

        if server_round > 0:
            avg_time_per_round = elapsed_time / server_round
            remaining_rounds = NUM_ROUNDS - server_round
            estimated_remaining = avg_time_per_round * remaining_rounds
            remaining_str = str(timedelta(seconds=int(estimated_remaining)))
            total_estimated = str(timedelta(seconds=int(elapsed_time + estimated_remaining)))

            print(f"Round {server_round}/{NUM_ROUNDS}: "
                  f"Acc={results['accuracy']:.2f}%, Loss={results['loss']:.4f} | "
                  f"Elapsed: {elapsed_str}, Remaining: ~{remaining_str}, Total: ~{total_estimated}")
        else:
            print(f"Round {server_round}: Acc={results['accuracy']:.2f}%, Loss={results['loss']:.4f}")

        return results['loss'], {'accuracy': results['accuracy']}

    # Create strategy based on aggregation method
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )

    if args.aggregation == 'fedavg':
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=0,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )
    elif args.aggregation == 'fedmean':
        strategy = FedMean(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=0,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )
    elif args.aggregation == 'fedmedian':
        strategy = FedMedian(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=0,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )

    # Configure Ray
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": 128,
        "num_gpus": 2,
        "_memory": 50 * 1024 * 1024 * 1024,
        "object_store_memory": 100 * 1024 * 1024 * 1024,
    }

    print("\nStarting federated learning simulation...")
    print(f"  Ray config: {ray_init_args}")
    print("-" * 60)

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.04 if DEVICE.type == 'cuda' else 0},
        ray_init_args=ray_init_args
    )

    print("-" * 60)
    print("\nSimulation complete!")

    # Save final results
    logger.save_csv()
    logger.save_json()

    # Print final results
    final_accuracy = history.metrics_centralized['accuracy'][-1][1]
    final_loss = history.losses_centralized[-1][1]
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {final_accuracy:.2f}%")
    print(f"  Test Loss: {final_loss:.4f}")
    print(f"  Results saved to: {args.output_dir}/{experiment_name}_metrics.*")


if __name__ == "__main__":
    main()
