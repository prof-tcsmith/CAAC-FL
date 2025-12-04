#!/usr/bin/env python3
"""
Parallel experiment runner for Non-IID federated learning experiments.

Runs multiple experiments simultaneously across available GPUs and CPUs.
Designed for systems with multiple GPUs (e.g., 2x RTX 4090) and many CPU cores.
"""

import subprocess
import sys
import os
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Configuration
EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "noniid"

# Experiment parameters
DATASETS = ['mnist', 'fashion_mnist', 'cifar10']
SEEDS = [42, 123, 456, 789, 1011]
WEIGHT_STRATEGIES = ['fedavg', 'fedmean', 'fedmedian']
GRADIENT_STRATEGIES = ['fedsgd', 'fedadam', 'fedtrimmed']
BYZANTINE_STRATEGIES = ['krum', 'multikrum']
CONDITIONS = ['noniid_equal', 'noniid_unequal']
CLIENT_COUNTS = [10, 25, 50]


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    condition: str
    dataset: str
    strategy: str
    seed: int
    num_clients: int
    alpha: float = 0.5
    gpu_id: int = 0

    @property
    def result_path(self) -> Path:
        return (RESULTS_DIR / self.condition / self.dataset /
                f"c{self.num_clients}" / f"alpha_{str(self.alpha).replace('.', '_')}" /
                f"{self.strategy}_seed{self.seed}.json")

    @property
    def is_completed(self) -> bool:
        return self.result_path.exists()

    def __str__(self) -> str:
        return f"{self.condition}/{self.dataset}/c{self.num_clients}/{self.strategy}_seed{self.seed}"


def get_pending_experiments(
    num_clients: Optional[int] = None,
    conditions: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    alpha: float = 0.5
) -> List[ExperimentConfig]:
    """Get list of experiments that haven't been completed yet."""
    pending = []

    client_counts = [num_clients] if num_clients else CLIENT_COUNTS
    conditions = conditions or CONDITIONS
    datasets = datasets or DATASETS
    strategies = strategies or WEIGHT_STRATEGIES

    for nc in client_counts:
        for cond in conditions:
            for ds in datasets:
                for strat in strategies:
                    for seed in SEEDS:
                        exp = ExperimentConfig(
                            condition=cond,
                            dataset=ds,
                            strategy=strat,
                            seed=seed,
                            num_clients=nc,
                            alpha=alpha
                        )
                        if not exp.is_completed:
                            pending.append(exp)

    return pending


def run_single_experiment(exp: ExperimentConfig) -> Tuple[str, bool, str]:
    """
    Run a single experiment using the existing runner.
    Returns (experiment_id, success, message).
    """
    exp_id = str(exp)

    # Set environment variables for GPU assignment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(exp.gpu_id)

    # Build command
    cmd = [
        sys.executable,
        str(EXPERIMENTS_DIR / "run_noniid_experiments.py"),
        "--dataset", exp.dataset,
        "--strategy", exp.strategy,
        "--condition", exp.condition,
        "--num_clients", str(exp.num_clients),
        "--seed", str(exp.seed),
        "--alpha", str(exp.alpha),
    ]

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per experiment
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            return (exp_id, True, f"Completed in {elapsed:.1f}s")
        else:
            return (exp_id, False, f"Failed: {result.stderr[-500:]}")

    except subprocess.TimeoutExpired:
        return (exp_id, False, "Timeout after 1 hour")
    except Exception as e:
        return (exp_id, False, f"Exception: {str(e)}")


def run_experiments_parallel(
    experiments: List[ExperimentConfig],
    num_workers: int = 4,
    num_gpus: int = 2
) -> None:
    """Run experiments in parallel across multiple workers."""

    if not experiments:
        print("No pending experiments to run!")
        return

    # Assign GPUs round-robin
    for i, exp in enumerate(experiments):
        exp.gpu_id = i % num_gpus

    print(f"\n{'='*60}")
    print(f"PARALLEL EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"Pending experiments: {len(experiments)}")
    print(f"Workers: {num_workers}")
    print(f"GPUs: {num_gpus}")
    print(f"{'='*60}\n")

    completed = 0
    failed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all experiments
        futures = {executor.submit(run_single_experiment, exp): exp for exp in experiments}

        # Process as they complete
        for future in as_completed(futures):
            exp = futures[future]
            exp_id, success, message = future.result()

            if success:
                completed += 1
                status = "✓"
            else:
                failed += 1
                status = "✗"

            elapsed = time.time() - start_time
            remaining = len(experiments) - completed - failed
            rate = (completed + failed) / elapsed if elapsed > 0 else 0
            eta = remaining / rate / 60 if rate > 0 else 0

            print(f"[{completed + failed}/{len(experiments)}] {status} {exp_id}")
            print(f"    {message}")
            print(f"    Progress: {completed} done, {failed} failed, ETA: {eta:.1f} min")
            print()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"{'='*60}")
    print(f"Total: {completed + failed} experiments in {total_time/60:.1f} minutes")
    print(f"Success: {completed}, Failed: {failed}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run FL experiments in parallel")
    parser.add_argument("--num_clients", type=int, help="Number of clients (default: all)")
    parser.add_argument("--condition", type=str, choices=CONDITIONS, help="Specific condition")
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Specific dataset")
    parser.add_argument("--strategy", type=str, help="Specific strategy")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs available (default: 2)")
    parser.add_argument("--gradient-strategies", action="store_true",
                       help="Run gradient strategies instead of weight strategies")
    parser.add_argument("--byzantine-strategies", action="store_true",
                       help="Run Byzantine-robust strategies (krum, multikrum)")
    parser.add_argument("--all-strategies", action="store_true",
                       help="Run all strategies (weight + gradient + byzantine)")
    parser.add_argument("--list-pending", action="store_true",
                       help="List pending experiments without running")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")

    args = parser.parse_args()

    # Determine strategies to run
    if args.all_strategies:
        strategies = WEIGHT_STRATEGIES + GRADIENT_STRATEGIES + BYZANTINE_STRATEGIES
    elif args.gradient_strategies:
        strategies = GRADIENT_STRATEGIES
    elif args.byzantine_strategies:
        strategies = BYZANTINE_STRATEGIES
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = WEIGHT_STRATEGIES

    # Get pending experiments
    conditions = [args.condition] if args.condition else None
    datasets = [args.dataset] if args.dataset else None

    pending = get_pending_experiments(
        num_clients=args.num_clients,
        conditions=conditions,
        datasets=datasets,
        strategies=strategies,
        alpha=args.alpha
    )

    if args.list_pending:
        print(f"\nPending experiments: {len(pending)}")
        for exp in pending:
            print(f"  {exp}")
        return

    # Run experiments
    run_experiments_parallel(
        experiments=pending,
        num_workers=args.workers,
        num_gpus=args.gpus
    )


if __name__ == "__main__":
    main()
