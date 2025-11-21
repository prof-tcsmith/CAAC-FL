"""
Run Level 1 experiment with FedMedian aggregation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import flwr as fl
from flwr.server.strategy import FedMedian
from torch.utils.data import DataLoader

from shared.data_utils import load_cifar10, partition_data_iid, create_dataloaders
from shared.models import SimpleCNN
from shared.metrics import MetricsLogger, evaluate_model
from client import create_client_fn


def main():
    print("=" * 60)
    print("Level 1: Federated Learning with FedMedian")
    print("=" * 60)

    # Configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    LOCAL_EPOCHS = 1
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Local epochs: {LOCAL_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_dataset, test_dataset = load_cifar10()
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Partition data (IID)
    print("\nPartitioning data (IID)...")
    client_dict = partition_data_iid(train_dataset, NUM_CLIENTS, seed=SEED)

    # Create dataloaders
    train_loaders = create_dataloaders(
        train_dataset,
        client_dict,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    print(f"  Samples per client: {[len(loader.dataset) for loader in train_loaders.values()]}")

    # Create client factory
    client_fn = create_client_fn(
        train_loaders=train_loaders,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        local_epochs=LOCAL_EPOCHS
    )

    # Initialize model for evaluation function
    model = SimpleCNN(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize metrics logger
    logger = MetricsLogger(
        log_dir='./results',
        experiment_name='level1_fedmedian'
    )

    # Define evaluation function
    def evaluate_fn(server_round: int, parameters, config):
        """Centralized evaluation function."""
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Evaluate
        results = evaluate_model(
            model,
            test_loader,
            device=DEVICE,
            criterion=criterion
        )

        # Log metrics
        logger.log_round(
            round_num=server_round,
            test_acc=results['accuracy'],
            test_loss=results['loss']
        )

        print(f"Round {server_round}: Test Accuracy = {results['accuracy']:.2f}%, Test Loss = {results['loss']:.4f}")

        return results['loss'], {'accuracy': results['accuracy']}

    # Configure strategy
    strategy = FedMedian(
        fraction_fit=1.0,  # All clients participate
        fraction_evaluate=0.0,  # No client-side evaluation
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in model.state_dict().items()]
        )
    )

    # Start simulation
    print("\nStarting federated learning simulation...")
    print("-" * 60)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.1 if DEVICE == 'cuda' else 0}
    )

    print("-" * 60)
    print("\nSimulation complete!")

    # Save results
    logger.save_csv()
    logger.save_json()

    # Print final results
    metrics = logger.get_metrics()
    final_acc = metrics['test_accuracy'][-1]
    final_loss = metrics['test_loss'][-1]

    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {final_acc:.2f}%")
    print(f"  Test Loss: {final_loss:.4f}")
    print(f"  Results saved to: ./results/level1_fedmedian_metrics.*")

    return history


if __name__ == "__main__":
    main()
