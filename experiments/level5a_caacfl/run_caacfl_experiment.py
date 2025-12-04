"""
Level 5a: CAAC-FL Demonstration Experiment

This experiment demonstrates the CAAC-FL algorithm on Fashion-MNIST,
showing that it can:
1. Maintain good accuracy with heterogeneous (non-IID) data
2. Detect and mitigate Byzantine attacks
3. Distinguish legitimate heterogeneity from attacks

This is a proof-of-concept implementation focusing on demonstrating
the method works, not comprehensive benchmarking.

Usage:
    python run_caacfl_experiment.py [--attack ATTACK] [--byzantine_ratio RATIO]

Examples:
    # Clean run (no attacks)
    python run_caacfl_experiment.py

    # With sign flipping attack
    python run_caacfl_experiment.py --attack sign_flipping --byzantine_ratio 0.2

    # With ALIE attack (stealthy)
    python run_caacfl_experiment.py --attack alie --byzantine_ratio 0.2
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import copy

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.data_utils import load_fashion_mnist, partition_data_dirichlet, analyze_data_distribution
from shared.models import create_model_for_dataset
from caacfl import CAACFLAggregator, flatten_model_params, unflatten_model_params, compute_gradient


# ============================================================================
# Attack Implementations (simplified for this experiment)
# ============================================================================

class Attack:
    """Base attack class."""
    def apply(self, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoAttack(Attack):
    """Honest client (no attack)."""
    def apply(self, gradient: np.ndarray) -> np.ndarray:
        return gradient


class SignFlippingAttack(Attack):
    """Reverse gradient direction."""
    def apply(self, gradient: np.ndarray) -> np.ndarray:
        return -gradient


class RandomNoiseAttack(Attack):
    """Add large random noise."""
    def __init__(self, scale: float = 5.0):
        self.scale = scale

    def apply(self, gradient: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*gradient.shape) * self.scale
        return gradient + noise


class ALIEAttack(Attack):
    """A Little Is Enough - stay within bounds but push wrong direction."""
    def __init__(self, z: float = 1.5):
        self.z = z

    def apply(self, gradient: np.ndarray) -> np.ndarray:
        mu = gradient.mean()
        sigma = gradient.std() + 1e-8
        # Push in opposite direction but stay within plausible bounds
        malicious = mu - self.z * sigma * np.sign(gradient)
        return malicious * np.ones_like(gradient)


def create_attack(attack_type: str) -> Attack:
    """Factory function for attacks."""
    attacks = {
        'none': NoAttack(),
        'sign_flipping': SignFlippingAttack(),
        'random_noise': RandomNoiseAttack(scale=5.0),
        'alie': ALIEAttack(z=1.5),
    }
    return attacks.get(attack_type, NoAttack())


# ============================================================================
# Training Utilities
# ============================================================================

def train_local(model: nn.Module,
                train_loader: DataLoader,
                device: torch.device,
                epochs: int = 1,
                lr: float = 0.01) -> int:
    """
    Train model locally for specified epochs.

    Returns:
        Number of samples trained on
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    num_samples = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            num_samples += len(data)

    return num_samples


def evaluate(model: nn.Module,
             test_loader: DataLoader,
             device: torch.device) -> dict:
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * len(data)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(data)

    return {
        'loss': total_loss / total,
        'accuracy': 100.0 * correct / total,
        'correct': correct,
        'total': total
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(args):
    """Run the CAAC-FL experiment."""

    print("=" * 70)
    print("Level 5a: CAAC-FL Demonstration on Fashion-MNIST")
    print("=" * 70)

    # Configuration
    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    LOCAL_EPOCHS = args.local_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    ALPHA = args.alpha  # Dirichlet for non-IID
    SEED = args.seed
    if args.cpu:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Byzantine configuration
    NUM_BYZANTINE = int(NUM_CLIENTS * args.byzantine_ratio)
    ATTACK_TYPE = args.attack

    print(f"\nConfiguration:")
    print(f"  Clients: {NUM_CLIENTS} (Byzantine: {NUM_BYZANTINE})")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Local Epochs: {LOCAL_EPOCHS}")
    print(f"  Dirichlet Alpha: {ALPHA} (Non-IID)")
    print(f"  Attack: {ATTACK_TYPE}")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print("=" * 70)

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load Fashion-MNIST
    print("\n1. Loading Fashion-MNIST dataset...")
    train_dataset, test_dataset = load_fashion_mnist(data_dir='./data')
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Partition data (Non-IID)
    print(f"\n2. Partitioning data (Dirichlet α={ALPHA})...")
    client_dict = partition_data_dirichlet(
        train_dataset,
        NUM_CLIENTS,
        alpha=ALPHA,
        seed=SEED
    )

    # Analyze distribution
    stats = analyze_data_distribution(train_dataset, client_dict)
    print(f"   Heterogeneity (KL): {stats['heterogeneity_metrics']['mean_kl_divergence']:.4f}")
    client_sizes = list(stats['client_sizes'].values())
    print(f"   Client sizes: {min(client_sizes)} - {max(client_sizes)}")

    # Create data loaders
    train_loaders = {}
    for client_id, indices in client_dict.items():
        sampler = SubsetRandomSampler(indices)
        train_loaders[client_id] = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler
        )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Identify Byzantine clients
    byzantine_clients = set(range(NUM_BYZANTINE))
    print(f"\n3. Byzantine clients: {list(byzantine_clients)}")

    # Create attack instances
    attacks = {}
    for client_id in range(NUM_CLIENTS):
        if client_id in byzantine_clients:
            attacks[client_id] = create_attack(ATTACK_TYPE)
        else:
            attacks[client_id] = NoAttack()

    # Initialize global model
    print("\n4. Initializing model and CAAC-FL aggregator...")
    global_model = create_model_for_dataset('fashion_mnist').to(DEVICE)

    # Initialize CAAC-FL aggregator
    aggregator = CAACFLAggregator(
        num_clients=NUM_CLIENTS,
        alpha=0.1,       # EWMA smoothing
        gamma=0.1,       # Reliability update rate
        tau_base=2.0,    # Base threshold (2 std devs)
        beta=0.5,        # Threshold flexibility
        window_size=5,   # History window
    )

    # Training loop
    print("\n5. Starting Federated Training...")
    print("-" * 70)

    results = {
        'config': {
            'num_clients': NUM_CLIENTS,
            'num_byzantine': NUM_BYZANTINE,
            'num_rounds': NUM_ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'alpha': ALPHA,
            'attack': ATTACK_TYPE,
            'seed': SEED,
        },
        'rounds': [],
        'caacfl_stats': [],
    }

    start_time = time.time()

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start = time.time()

        # Get current global model parameters
        global_params = flatten_model_params(global_model)

        # Collect client updates
        client_gradients = {}
        client_samples = {}

        for client_id in range(NUM_CLIENTS):
            # Create local model copy
            local_model = create_model_for_dataset('fashion_mnist').to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())

            # Store pre-training state
            model_before = copy.deepcopy(local_model)

            # Local training
            num_samples = train_local(
                local_model, train_loaders[client_id], DEVICE,
                epochs=LOCAL_EPOCHS, lr=LEARNING_RATE
            )

            # Compute gradient (model update)
            gradient = compute_gradient(model_before, local_model)

            # Apply attack if Byzantine
            gradient = attacks[client_id].apply(gradient)

            client_gradients[client_id] = gradient
            client_samples[client_id] = num_samples

        # CAAC-FL Aggregation
        aggregated_gradient, round_stats = aggregator.aggregate(
            client_gradients, client_samples
        )

        # Update global model
        new_params = global_params + aggregated_gradient
        global_model = unflatten_model_params(new_params, global_model)

        # Evaluate
        eval_result = evaluate(global_model, test_loader, DEVICE)

        # Collect round statistics
        anomalous_clients = [s['client_id'] for s in round_stats if s['is_anomalous']]
        detected_byzantine = len(set(anomalous_clients) & byzantine_clients)
        false_positives = len(set(anomalous_clients) - byzantine_clients)

        round_result = {
            'round': round_num,
            'accuracy': eval_result['accuracy'],
            'loss': eval_result['loss'],
            'anomalous_clients': anomalous_clients,
            'detected_byzantine': detected_byzantine,
            'false_positives': false_positives,
            'mean_anomaly_score': np.mean([s['anomaly_score'] for s in round_stats]),
            'mean_reliability': np.mean([s['reliability'] for s in round_stats]),
        }
        results['rounds'].append(round_result)
        results['caacfl_stats'].append(round_stats)

        round_time = time.time() - round_start

        # Print progress
        print(f"Round {round_num:3d}/{NUM_ROUNDS}: "
              f"Acc={eval_result['accuracy']:5.2f}% | "
              f"Anomalous={len(anomalous_clients):2d} "
              f"(True:{detected_byzantine} FP:{false_positives}) | "
              f"MeanRel={round_result['mean_reliability']:.3f} | "
              f"Time={round_time:.1f}s")

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"\nTraining completed in {total_time:.1f} seconds")

    # Final evaluation
    final_result = evaluate(global_model, test_loader, DEVICE)
    print(f"\nFinal Test Accuracy: {final_result['accuracy']:.2f}%")
    print(f"Final Test Loss: {final_result['loss']:.4f}")

    # CAAC-FL Summary
    print("\n" + "=" * 70)
    print("CAAC-FL Analysis Summary")
    print("=" * 70)

    caacfl_summary = aggregator.get_summary_stats()
    print(f"  Total Rounds: {caacfl_summary['total_rounds']}")
    print(f"  Mean Anomalous/Round: {caacfl_summary['mean_anomalous_per_round']:.2f}")
    print(f"  Mean Anomaly Score: {caacfl_summary['mean_anomaly_score']:.4f}")
    print(f"  Final Mean Reliability: {caacfl_summary['final_mean_reliability']:.4f}")

    # Detection performance
    total_detections = sum(r['detected_byzantine'] for r in results['rounds'])
    total_false_positives = sum(r['false_positives'] for r in results['rounds'])
    potential_detections = NUM_BYZANTINE * NUM_ROUNDS if NUM_BYZANTINE > 0 else 0

    print(f"\nDetection Performance:")
    if potential_detections > 0:
        detection_rate = 100.0 * total_detections / potential_detections
        print(f"  Byzantine Detection Rate: {detection_rate:.1f}% ({total_detections}/{potential_detections})")
    print(f"  Total False Positives: {total_false_positives}")

    # Add summary to results
    results['summary'] = {
        'final_accuracy': final_result['accuracy'],
        'final_loss': final_result['loss'],
        'total_time': total_time,
        'caacfl_summary': caacfl_summary,
        'detection_stats': {
            'total_detections': total_detections,
            'total_false_positives': total_false_positives,
            'detection_rate': detection_rate if potential_detections > 0 else None,
        }
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"./results/caacfl_{ATTACK_TYPE}_{timestamp}.json"
    os.makedirs('./results', exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(result_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {result_file}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Level 5a: CAAC-FL Demonstration Experiment'
    )

    # FL parameters
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients (default: 10)')
    parser.add_argument('--num_rounds', type=int, default=20,
                        help='Number of FL rounds (default: 20)')
    parser.add_argument('--local_epochs', type=int, default=2,
                        help='Local training epochs (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')

    # Data heterogeneity
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID (default: 0.5)')

    # Byzantine parameters
    parser.add_argument('--attack', type=str, default='none',
                        choices=['none', 'sign_flipping', 'random_noise', 'alie'],
                        help='Attack type (default: none)')
    parser.add_argument('--byzantine_ratio', type=float, default=0.2,
                        help='Fraction of Byzantine clients (default: 0.2)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
