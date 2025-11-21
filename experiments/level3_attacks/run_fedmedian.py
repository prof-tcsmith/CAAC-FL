"""
Level 3: FedMedian with Byzantine Attacks

Runs FedMedian aggregation with configurable Byzantine attacks.
Tests coordinate-wise median robustness against attacks.
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
import flwr as fl

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.models import SimpleCNN
from shared.data_utils import load_cifar10, partition_data_dirichlet, analyze_data_distribution
from shared.metrics import evaluate_model, MetricsLogger
from client import create_client
from attacks import create_attack


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Level 3: FedMedian with Byzantine Attacks')
    parser.add_argument(
        '--attack',
        type=str,
        default='none',
        choices=['none', 'random_noise', 'sign_flipping'],
        help='Type of Byzantine attack (default: none)'
    )
    parser.add_argument('--num_rounds', type=int, default=50, help='Number of rounds')
    parser.add_argument('--num_clients', type=int, default=50, help='Total number of clients')
    parser.add_argument('--byzantine_ratio', type=float, default=0.2, help='Fraction of Byzantine clients')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-IID data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()

    # Configuration
    NUM_ROUNDS = args.num_rounds
    NUM_CLIENTS = args.num_clients
    NUM_BYZANTINE = int(NUM_CLIENTS * args.byzantine_ratio)
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Level 3: FedMedian with Byzantine Attacks")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Aggregation Method: FedMedian")
    print(f"  Attack Type: {args.attack}")
    print(f"  Total Clients: {NUM_CLIENTS}")
    print(f"  Byzantine Clients: {NUM_BYZANTINE} ({args.byzantine_ratio*100:.0f}%)")
    print(f"  Honest Clients: {NUM_CLIENTS - NUM_BYZANTINE} ({(1-args.byzantine_ratio)*100:.0f}%)")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Local Epochs: {LOCAL_EPOCHS}")
    print(f"  Dirichlet Alpha: {args.alpha}")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    # Set random seeds
    torch.manual_seed(args.seed)

    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    train_dataset, test_dataset = load_cifar10()

    # Partition data (Non-IID using Dirichlet)
    print(f"\nPartitioning data (Dirichlet alpha={args.alpha})...")
    client_dict = partition_data_dirichlet(
        train_dataset,
        NUM_CLIENTS,
        alpha=args.alpha,
        seed=args.seed
    )

    # Analyze data distribution
    stats = analyze_data_distribution(train_dataset, client_dict)
    client_sizes = list(stats['client_sizes'].values())
    print(f"\nData Distribution Statistics:")
    print(f"  Heterogeneity (KL divergence): {stats['heterogeneity_metrics']['mean_kl_divergence']:.4f}")
    print(f"  Class Imbalance: {stats['heterogeneity_metrics']['mean_class_imbalance']:.4f}")
    print(f"  Client sizes: {min(client_sizes)} - {max(client_sizes)} samples")

    # Determine Byzantine clients (first NUM_BYZANTINE clients)
    byzantine_clients = list(range(NUM_BYZANTINE))
    print(f"\nByzantine clients: {byzantine_clients}")

    # Create test dataloader (shared)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Client function factory
    def client_fn(cid: str):
        """Create a client instance"""
        client_id = int(cid)
        is_byzantine = client_id in byzantine_clients

        # Create attack if Byzantine
        attack = None
        if is_byzantine:
            if args.attack == 'random_noise':
                attack = create_attack('random_noise', noise_scale=1.0, seed=args.seed + client_id)
            elif args.attack == 'sign_flipping':
                attack = create_attack('sign_flipping', seed=args.seed + client_id)

        # Get client's data
        client_indices = client_dict[client_id]
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(client_indices)
        )

        return create_client(
            client_id=client_id,
            trainloader=train_loader,
            testloader=test_loader,
            device=DEVICE,
            is_byzantine=is_byzantine,
            attack=attack,
            local_epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
        )

    # Initialize metrics logger
    attack_suffix = f"_{args.attack}" if args.attack != 'none' else "_no_attack"
    logger = MetricsLogger(
        log_dir='./results',
        experiment_name=f'level3_fedmedian{attack_suffix}'
    )

    # Log heterogeneity metrics
    logger.log_round(
        round_num=0,
        heterogeneity_kl=stats['heterogeneity_metrics']['mean_kl_divergence'],
        class_imbalance=stats['heterogeneity_metrics']['mean_class_imbalance'],
        num_byzantine=NUM_BYZANTINE,
        attack_type=args.attack,
    )

    # Create server evaluation function
    def evaluate_fn(server_round, parameters, config):
        """Centralized evaluation function"""
        model = SimpleCNN(num_classes=10).to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        result = evaluate_model(model, test_loader, DEVICE, criterion=torch.nn.CrossEntropyLoss())
        accuracy = result['accuracy']
        loss = result['loss']

        print(f"Round {server_round}: Test Accuracy = {accuracy:.2f}%, Test Loss = {loss:.4f}")

        # Log metrics
        logger.log_round(
            round_num=server_round,
            test_accuracy=accuracy,
            test_loss=loss
        )

        return loss, {"accuracy": accuracy}

    # Configure strategy (FedMedian)
    strategy = fl.server.strategy.FedMedian(
        fraction_fit=1.0,  # Use all clients
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
    )

    # Start simulation
    print(f"\nStarting FedMedian simulation ({NUM_ROUNDS} rounds)...")
    print("=" * 70)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # Save final results
    logger.save_csv()
    logger.save_json()
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print(f"Results saved to: {logger.log_dir}/{logger.experiment_name}_metrics.csv")
    print(f"                  {logger.log_dir}/{logger.experiment_name}_metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
