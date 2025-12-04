"""
Level 5a: Comprehensive CAAC-FL Experiments

This script runs a systematic evaluation of CAAC-FL across:
- Multiple attack types (none, sign_flipping, random_noise, alie)
- Multiple Byzantine ratios (0%, 10%, 20%, 30%)
- Multiple seeds for statistical significance
- Comparison with baseline (no defense)

Experiment Matrix:
- 4 attack types × 4 Byzantine ratios × 3 seeds = 48 experiments
- Plus baseline comparisons (FedAvg without defense)

Usage:
    # Run all experiments
    python run_comprehensive_experiments.py

    # Run specific subset
    python run_comprehensive_experiments.py --attack sign_flipping --byzantine_ratio 0.2

    # Quick test mode (fewer rounds)
    python run_comprehensive_experiments.py --quick
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
from itertools import product
import pandas as pd
from multiprocessing import Pool, cpu_count
import multiprocessing

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.data_utils import load_fashion_mnist, partition_data_dirichlet, analyze_data_distribution
from shared.models import create_model_for_dataset
from caacfl import CAACFLAggregator, flatten_model_params, unflatten_model_params, compute_gradient


# ============================================================================
# Experiment Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # FL Parameters - matching level3 experiments
    'num_clients': 25,
    'num_rounds': 50,
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,

    # Data Parameters
    'dataset': 'fashion_mnist',
    'alpha': 0.5,  # Dirichlet for non-IID

    # Attack Parameters (to be varied)
    'attacks': ['none', 'sign_flipping', 'random_noise', 'alie'],
    'byzantine_ratios': [0.0, 0.1, 0.2, 0.3],

    # CAAC-FL Parameters
    'caacfl_alpha': 0.1,       # EWMA smoothing
    'caacfl_gamma': 0.1,       # Reliability update rate
    'caacfl_tau_base': 2.0,    # Base threshold
    'caacfl_beta': 0.5,        # Threshold flexibility
    'caacfl_window': 5,        # History window

    # Cold-start parameters
    'warmup_rounds': 5,
    'warmup_factor': 0.5,
    'min_rounds_for_trust': 3,
    'new_client_weight': 0.5,

    # Reproducibility
    'seeds': [42, 123, 456],
}

QUICK_CONFIG = {
    **DEFAULT_CONFIG,
    'num_rounds': 10,
    'seeds': [42],
    'byzantine_ratios': [0.0, 0.2],
    'attacks': ['none', 'sign_flipping'],
}


# ============================================================================
# Attack Implementations
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
    """Train model locally for specified epochs."""
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
# Baseline Aggregation (FedAvg without defense)
# ============================================================================

def fedavg_aggregate(client_gradients: dict, client_samples: dict) -> np.ndarray:
    """Simple FedAvg aggregation without any defense."""
    total_samples = sum(client_samples.values())
    aggregated = None

    for client_id, grad in client_gradients.items():
        weight = client_samples[client_id] / total_samples
        if aggregated is None:
            aggregated = weight * grad
        else:
            aggregated += weight * grad

    return aggregated


# ============================================================================
# Single Experiment Runner
# ============================================================================

def run_single_experiment(config: dict) -> dict:
    """Run a single experiment with given configuration."""

    # Extract config
    NUM_CLIENTS = config['num_clients']
    NUM_ROUNDS = config['num_rounds']
    LOCAL_EPOCHS = config['local_epochs']
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['learning_rate']
    ALPHA = config['alpha']
    SEED = config['seed']
    ATTACK_TYPE = config['attack']
    BYZANTINE_RATIO = config['byzantine_ratio']
    USE_CAACFL = config.get('use_caacfl', True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() and not config.get('cpu', False) else "cpu")

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data
    train_dataset, test_dataset = load_fashion_mnist(data_dir='./data')

    # Partition data (Non-IID)
    client_dict = partition_data_dirichlet(
        train_dataset,
        NUM_CLIENTS,
        alpha=ALPHA,
        seed=SEED
    )

    # Analyze distribution
    stats = analyze_data_distribution(train_dataset, client_dict)

    # Create data loaders
    train_loaders = {}
    for client_id, indices in client_dict.items():
        sampler = SubsetRandomSampler(indices)
        train_loaders[client_id] = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler
        )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Identify Byzantine clients
    NUM_BYZANTINE = int(NUM_CLIENTS * BYZANTINE_RATIO)
    byzantine_clients = set(range(NUM_BYZANTINE))

    # Create attack instances
    attacks = {}
    for client_id in range(NUM_CLIENTS):
        if client_id in byzantine_clients:
            attacks[client_id] = create_attack(ATTACK_TYPE)
        else:
            attacks[client_id] = NoAttack()

    # Initialize global model
    global_model = create_model_for_dataset('fashion_mnist').to(DEVICE)

    # Initialize aggregator (CAAC-FL or baseline)
    if USE_CAACFL:
        aggregator = CAACFLAggregator(
            num_clients=NUM_CLIENTS,
            alpha=config['caacfl_alpha'],
            gamma=config['caacfl_gamma'],
            tau_base=config['caacfl_tau_base'],
            beta=config['caacfl_beta'],
            window_size=config['caacfl_window'],
            warmup_rounds=config['warmup_rounds'],
            warmup_factor=config['warmup_factor'],
            min_rounds_for_trust=config['min_rounds_for_trust'],
            new_client_weight=config['new_client_weight'],
        )
    else:
        aggregator = None

    # Results storage
    results = {
        'config': {
            'num_clients': NUM_CLIENTS,
            'num_byzantine': NUM_BYZANTINE,
            'num_rounds': NUM_ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'alpha': ALPHA,
            'attack': ATTACK_TYPE,
            'byzantine_ratio': BYZANTINE_RATIO,
            'seed': SEED,
            'use_caacfl': USE_CAACFL,
        },
        'rounds': [],
        'caacfl_stats': [] if USE_CAACFL else None,
        'data_stats': {
            'heterogeneity_kl': stats['heterogeneity_metrics']['mean_kl_divergence'],
            'client_sizes': {str(k): v for k, v in stats['client_sizes'].items()},
        }
    }

    start_time = time.time()

    # Training loop
    for round_num in range(1, NUM_ROUNDS + 1):
        global_params = flatten_model_params(global_model)

        # Collect client updates
        client_gradients = {}
        client_samples = {}

        for client_id in range(NUM_CLIENTS):
            local_model = create_model_for_dataset('fashion_mnist').to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())

            model_before = copy.deepcopy(local_model)

            num_samples = train_local(
                local_model, train_loaders[client_id], DEVICE,
                epochs=LOCAL_EPOCHS, lr=LEARNING_RATE
            )

            gradient = compute_gradient(model_before, local_model)
            gradient = attacks[client_id].apply(gradient)

            client_gradients[client_id] = gradient
            client_samples[client_id] = num_samples

        # Aggregation
        if USE_CAACFL:
            aggregated_gradient, round_stats = aggregator.aggregate(
                client_gradients, client_samples
            )
            results['caacfl_stats'].append(round_stats)
        else:
            aggregated_gradient = fedavg_aggregate(client_gradients, client_samples)
            round_stats = []

        # Update global model
        new_params = global_params + aggregated_gradient
        global_model = unflatten_model_params(new_params, global_model)

        # Evaluate
        eval_result = evaluate(global_model, test_loader, DEVICE)

        # Collect round statistics
        if USE_CAACFL:
            anomalous_clients = [s['client_id'] for s in round_stats if s['is_anomalous']]
            detected_byzantine = len(set(anomalous_clients) & byzantine_clients)
            false_positives = len(set(anomalous_clients) - byzantine_clients)
        else:
            anomalous_clients = []
            detected_byzantine = 0
            false_positives = 0

        round_result = {
            'round': round_num,
            'accuracy': eval_result['accuracy'],
            'loss': eval_result['loss'],
            'anomalous_clients': anomalous_clients,
            'detected_byzantine': detected_byzantine,
            'false_positives': false_positives,
        }

        if USE_CAACFL:
            round_result['mean_anomaly_score'] = np.mean([s['anomaly_score'] for s in round_stats])
            round_result['mean_reliability'] = np.mean([s['reliability'] for s in round_stats])

        results['rounds'].append(round_result)

    total_time = time.time() - start_time

    # Final evaluation
    final_result = evaluate(global_model, test_loader, DEVICE)

    # Detection performance summary
    if USE_CAACFL and NUM_BYZANTINE > 0:
        total_detections = sum(r['detected_byzantine'] for r in results['rounds'])
        total_false_positives = sum(r['false_positives'] for r in results['rounds'])
        potential_detections = NUM_BYZANTINE * NUM_ROUNDS
        detection_rate = 100.0 * total_detections / potential_detections
    else:
        total_detections = 0
        total_false_positives = 0
        detection_rate = None

    results['summary'] = {
        'final_accuracy': final_result['accuracy'],
        'final_loss': final_result['loss'],
        'best_accuracy': max(r['accuracy'] for r in results['rounds']),
        'total_time': total_time,
        'detection_stats': {
            'total_detections': total_detections,
            'total_false_positives': total_false_positives,
            'detection_rate': detection_rate,
        }
    }

    return results


# ============================================================================
# Comprehensive Experiment Suite
# ============================================================================

def run_single_experiment_worker(args):
    """Worker function for parallel execution."""
    strategy, exp_config, output_dir, exp_idx, total_exp = args

    attack = exp_config['attack']
    byz_ratio = exp_config['byzantine_ratio']
    seed = exp_config['seed']
    gpu_id = exp_config.get('gpu_id', 0)

    exp_name = f"{strategy}_{attack}_byz{int(byz_ratio*100)}_seed{seed}"

    # Set GPU for this worker using CUDA_VISIBLE_DEVICES before any CUDA calls
    if torch.cuda.is_available() and not exp_config.get('cpu', False):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Force torch to see only this GPU
        torch.cuda.set_device(0)  # Device 0 of the visible devices

    print(f"[{exp_idx+1}/{total_exp}] Starting: {exp_name} (GPU {gpu_id})", flush=True)

    try:
        results = run_single_experiment(exp_config)

        # Add metadata
        results['experiment_name'] = exp_name
        results['strategy'] = strategy
        results['timestamp'] = datetime.now().isoformat()

        # Save individual result
        result_file = os.path.join(output_dir, f"{exp_name}.json")
        save_results(results, result_file)

        acc = results['summary']['final_accuracy']
        det_rate = results['summary']['detection_stats'].get('detection_rate')
        det_str = f", DetRate: {det_rate:.1f}%" if det_rate is not None else ""
        print(f"[{exp_idx+1}/{total_exp}] Done: {exp_name} -> Acc: {acc:.2f}%{det_str}", flush=True)

        return results

    except Exception as e:
        print(f"[{exp_idx+1}/{total_exp}] ERROR in {exp_name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_experiments(config: dict, output_dir: str = './results/comprehensive', num_workers: int = 4):
    """Run all experiments in the configuration matrix with parallel execution."""

    os.makedirs(output_dir, exist_ok=True)

    attacks = config['attacks']
    byzantine_ratios = config['byzantine_ratios']
    seeds = config['seeds']

    # Generate experiment matrix
    experiments = []

    # CAAC-FL experiments
    for attack, byz_ratio, seed in product(attacks, byzantine_ratios, seeds):
        # Skip byzantine ratio > 0 for 'none' attack
        if attack == 'none' and byz_ratio > 0:
            continue
        # Skip byzantine ratio 0 for actual attacks
        if attack != 'none' and byz_ratio == 0:
            continue

        exp_config = {
            **config,
            'attack': attack,
            'byzantine_ratio': byz_ratio,
            'seed': seed,
            'use_caacfl': True,
        }
        experiments.append(('caacfl', exp_config))

    # Baseline (FedAvg) experiments for comparison
    for attack, byz_ratio, seed in product(attacks, byzantine_ratios, seeds):
        if attack == 'none' and byz_ratio > 0:
            continue
        if attack != 'none' and byz_ratio == 0:
            continue

        exp_config = {
            **config,
            'attack': attack,
            'byzantine_ratio': byz_ratio,
            'seed': seed,
            'use_caacfl': False,
        }
        experiments.append(('fedavg', exp_config))

    # Determine number of GPUs available
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0 or config.get('cpu', False):
        num_workers = 1  # Serial execution on CPU
        print("Running on CPU (serial execution)")
    else:
        num_workers = min(num_workers, num_gpus)
        print(f"Running on {num_gpus} GPUs with {num_workers} parallel workers")

    print("=" * 70)
    print("CAAC-FL Comprehensive Experiment Suite")
    print("=" * 70)
    print(f"Total experiments: {len(experiments)}")
    print(f"Parallel workers: {num_workers}")
    print(f"Attacks: {attacks}")
    print(f"Byzantine ratios: {byzantine_ratios}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    sys.stdout.flush()

    # Assign GPU IDs to experiments (round-robin)
    worker_args = []
    for i, (strategy, exp_config) in enumerate(experiments):
        exp_config_copy = exp_config.copy()
        exp_config_copy['gpu_id'] = i % num_workers if num_gpus > 0 else 0
        worker_args.append((strategy, exp_config_copy, output_dir, i, len(experiments)))

    # Run experiments in parallel
    if num_workers > 1:
        # Use spawn method for CUDA compatibility
        multiprocessing.set_start_method('spawn', force=True)
        with Pool(processes=num_workers) as pool:
            all_results = pool.map(run_single_experiment_worker, worker_args)
    else:
        # Sequential execution
        all_results = [run_single_experiment_worker(args) for args in worker_args]

    # Filter out failed experiments
    all_results = [r for r in all_results if r is not None]

    # Save combined results
    combined_file = os.path.join(output_dir, "all_experiments.json")
    save_results({'experiments': all_results}, combined_file)

    # Generate summary CSV
    generate_summary_csv(all_results, output_dir)

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return all_results


def save_results(results: dict, filepath: str):
    """Save results to JSON with numpy type conversion."""
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

    with open(filepath, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)


def generate_summary_csv(results: list, output_dir: str):
    """Generate summary CSV from all experiments."""
    rows = []

    for r in results:
        row = {
            'experiment': r.get('experiment_name', ''),
            'strategy': r.get('strategy', ''),
            'attack': r['config']['attack'],
            'byzantine_ratio': r['config']['byzantine_ratio'],
            'num_byzantine': r['config']['num_byzantine'],
            'seed': r['config']['seed'],
            'final_accuracy': r['summary']['final_accuracy'],
            'best_accuracy': r['summary']['best_accuracy'],
            'final_loss': r['summary']['final_loss'],
            'total_time': r['summary']['total_time'],
            'heterogeneity_kl': r['data_stats']['heterogeneity_kl'],
        }

        if r['strategy'] == 'caacfl':
            row['detection_rate'] = r['summary']['detection_stats']['detection_rate']
            row['total_detections'] = r['summary']['detection_stats']['total_detections']
            row['false_positives'] = r['summary']['detection_stats']['total_false_positives']

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_file = os.path.join(output_dir, "experiment_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file}")

    # Also generate aggregated statistics
    agg_rows = []
    for strategy in df['strategy'].unique():
        for attack in df['attack'].unique():
            for byz_ratio in df['byzantine_ratio'].unique():
                subset = df[(df['strategy'] == strategy) &
                           (df['attack'] == attack) &
                           (df['byzantine_ratio'] == byz_ratio)]
                if len(subset) > 0:
                    agg_row = {
                        'strategy': strategy,
                        'attack': attack,
                        'byzantine_ratio': byz_ratio,
                        'accuracy_mean': subset['final_accuracy'].mean(),
                        'accuracy_std': subset['final_accuracy'].std(),
                        'accuracy_min': subset['final_accuracy'].min(),
                        'accuracy_max': subset['final_accuracy'].max(),
                        'n_runs': len(subset),
                    }
                    if 'detection_rate' in subset.columns:
                        agg_row['detection_rate_mean'] = subset['detection_rate'].mean()
                    agg_rows.append(agg_row)

    agg_df = pd.DataFrame(agg_rows)
    agg_file = os.path.join(output_dir, "aggregated_results.csv")
    agg_df.to_csv(agg_file, index=False)
    print(f"Aggregated results saved to: {agg_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Level 5a: Comprehensive CAAC-FL Experiments'
    )

    parser.add_argument('--quick', action='store_true',
                        help='Run quick test mode (fewer experiments)')
    parser.add_argument('--attack', type=str, default=None,
                        choices=['none', 'sign_flipping', 'random_noise', 'alie'],
                        help='Run only specific attack type')
    parser.add_argument('--byzantine_ratio', type=float, default=None,
                        help='Run only specific Byzantine ratio')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only specific seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--output_dir', type=str, default='./results/comprehensive',
                        help='Output directory for results')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel workers (default: 2, max: num_gpus)')

    args = parser.parse_args()

    # Select configuration
    if args.quick:
        config = QUICK_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()

    # Apply filters
    if args.attack:
        config['attacks'] = [args.attack]
    if args.byzantine_ratio is not None:
        config['byzantine_ratios'] = [args.byzantine_ratio]
    if args.seed is not None:
        config['seeds'] = [args.seed]
    if args.cpu:
        config['cpu'] = True

    # Run experiments
    run_comprehensive_experiments(config, args.output_dir, num_workers=args.workers)


if __name__ == "__main__":
    main()
