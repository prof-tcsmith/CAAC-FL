#!/usr/bin/env python3
"""
Main experiment runner for CAAC-FL
Orchestrates federated learning experiments based on configuration files
"""

import argparse
import yaml
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import random
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.fl_core.server import create_strategy, start_server
from src.fl_core.client import create_client_fn
from src.utils.data_partitioner import partition_dataset
from src.utils.metrics import MetricsAggregator
from src.utils.visualization import plot_results
from src.utils.logging import setup_experiment_logging

import flwr as fl

logger = logging.getLogger(__name__)


def set_reproducible_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def prepare_experiment_dir(config: Dict[str, Any], experiment_name: str) -> Path:
    """Create and prepare experiment output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config['output']['results_dir']) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    
    # Save config to experiment directory
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def setup_federation(
    config: Dict[str, Any],
    dataset_name: str,
    heterogeneity: str,
    byzantine_fraction: float,
    attack_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Set up federated learning configuration.
    
    Args:
        config: Experiment configuration
        dataset_name: Name of dataset
        heterogeneity: Heterogeneity level (mild/extreme)
        byzantine_fraction: Fraction of Byzantine clients
        attack_type: Type of attack to apply
        
    Returns:
        Federation configuration dictionary
    """
    fed_config = config['federation'].copy()
    
    # Set heterogeneity
    alpha = config['federation']['heterogeneity'][heterogeneity]['alpha']
    fed_config['alpha'] = alpha
    
    # Set Byzantine configuration
    num_clients = fed_config['num_clients']
    num_byzantine = int(byzantine_fraction * num_clients)
    
    if num_byzantine > 0 and attack_type and attack_type != "none":
        # Randomly select Byzantine clients
        byzantine_ids = np.random.choice(
            num_clients, 
            size=num_byzantine, 
            replace=False
        ).tolist()
        
        fed_config['byzantine_config'] = {
            'num_byzantine': num_byzantine,
            'byzantine_client_ids': byzantine_ids,
            'attack_type': attack_type,
            'attack_config': config['attacks'].get(attack_type, {}),
            'attack_start_round': 0,  # Start attacks immediately after bootstrap
        }
        logger.info(f"Selected {num_byzantine} Byzantine clients: {byzantine_ids}")
    else:
        fed_config['byzantine_config'] = None
    
    return fed_config


def run_single_experiment(
    config: Dict[str, Any],
    dataset_name: str,
    aggregator_name: str,
    heterogeneity: str,
    byzantine_fraction: float,
    attack_type: str,
    seed: int,
    exp_dir: Path
) -> Dict[str, Any]:
    """
    Run a single federated learning experiment.
    
    Args:
        config: Experiment configuration
        dataset_name: Dataset to use
        aggregator_name: Aggregation method
        heterogeneity: Data heterogeneity level
        byzantine_fraction: Fraction of Byzantine clients
        attack_type: Attack type
        seed: Random seed
        exp_dir: Experiment directory
        
    Returns:
        Experiment results dictionary
    """
    # Set seed
    set_reproducible_seed(seed)
    
    # Create experiment name
    exp_name = f"{dataset_name}_{aggregator_name}_{heterogeneity}_byz{byzantine_fraction}_{attack_type}_seed{seed}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"{'='*60}")
    
    # Set up federation
    fed_config = setup_federation(
        config=config,
        dataset_name=dataset_name,
        heterogeneity=heterogeneity,
        byzantine_fraction=byzantine_fraction,
        attack_type=attack_type
    )
    
    # Partition dataset
    logger.info("Partitioning dataset among clients...")
    client_data_configs = partition_dataset(
        dataset_name=dataset_name,
        num_clients=fed_config['num_clients'],
        alpha=fed_config['alpha'],
        data_dir=Path(config['datasets'][dataset_name]['data_dir']),
        seed=seed
    )
    
    # Create strategy
    aggregator_config = config['aggregators'][aggregator_name].copy()
    aggregator_config.update(config['training'])
    
    if fed_config.get('byzantine_config'):
        aggregator_config['byzantine_config'] = fed_config['byzantine_config']
    
    strategy = create_strategy(
        strategy_name=aggregator_name,
        config=aggregator_config,
        num_clients=fed_config['num_clients'],
        checkpoint_dir=exp_dir / "checkpoints" / exp_name
    )
    
    # Create client factory
    client_fn = create_client_fn(
        client_configs=client_data_configs,
        model_name=config['models'][dataset_name]['name'],
        dataset_name=dataset_name,
        data_dir=Path(config['datasets'][dataset_name]['data_dir']),
        device=config['experiment']['device'],
        byzantine_config=fed_config.get('byzantine_config')
    )
    
    # Start simulation
    logger.info(f"Starting FL simulation for {config['experiment']['num_rounds']} rounds...")
    
    # Run using Flower's simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=fed_config['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['experiment']['num_rounds']),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.1 if torch.cuda.is_available() else 0},
    )
    
    # Process and save results
    results = {
        'experiment_name': exp_name,
        'dataset': dataset_name,
        'aggregator': aggregator_name,
        'heterogeneity': heterogeneity,
        'byzantine_fraction': byzantine_fraction,
        'attack_type': attack_type,
        'seed': seed,
        'history': history,
        'config': {
            'aggregator_config': aggregator_config,
            'fed_config': fed_config,
        }
    }
    
    # Save results
    results_path = exp_dir / "metrics" / f"{exp_name}.json"
    with open(results_path, 'w') as f:
        # Convert history to serializable format
        serializable_results = {
            k: v for k, v in results.items() 
            if k != 'history'
        }
        serializable_results['metrics'] = {
            'losses': history.losses_distributed,
            'metrics': history.metrics_distributed,
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved results to {results_path}")
    
    # Generate plots
    plot_results(
        history=history,
        exp_name=exp_name,
        output_dir=exp_dir / "plots"
    )
    
    return results


def run_experiment_matrix(config: Dict[str, Any], exp_dir: Path):
    """
    Run the full experiment matrix defined in configuration.
    
    Args:
        config: Experiment configuration
        exp_dir: Experiment output directory
    """
    all_results = []
    
    # Run H1 experiments (Heterogeneity Preservation)
    logger.info("\n" + "="*60)
    logger.info("Running H1 experiments (Heterogeneity Preservation)")
    logger.info("="*60)
    
    for exp_config in config['core_experiments']['h1_experiments']:
        dataset = exp_config['dataset']
        for aggregator in exp_config['aggregators']:
            for hetero in exp_config['heterogeneity']:
                for seed in exp_config['seeds']:
                    results = run_single_experiment(
                        config=config,
                        dataset_name=dataset,
                        aggregator_name=aggregator,
                        heterogeneity=hetero,
                        byzantine_fraction=exp_config['byzantine_fraction'],
                        attack_type=exp_config['attack'],
                        seed=seed,
                        exp_dir=exp_dir
                    )
                    all_results.append(results)
    
    # Run H2 experiments (Robustness)
    logger.info("\n" + "="*60)
    logger.info("Running H2 experiments (Multi-Dimensional Robustness)")
    logger.info("="*60)
    
    for exp_config in config['core_experiments']['h2_experiments']:
        dataset = exp_config['dataset']
        for aggregator in exp_config['aggregators']:
            for attack in exp_config['attacks']:
                for seed in exp_config['seeds']:
                    results = run_single_experiment(
                        config=config,
                        dataset_name=dataset,
                        aggregator_name=aggregator,
                        heterogeneity=exp_config['heterogeneity'],
                        byzantine_fraction=exp_config['byzantine_fraction'],
                        attack_type=attack,
                        seed=seed,
                        exp_dir=exp_dir
                    )
                    all_results.append(results)
    
    # Run H3 experiments (Temporal Discrimination)
    logger.info("\n" + "="*60)
    logger.info("Running H3 experiments (Temporal Discrimination)")
    logger.info("="*60)
    
    for exp_config in config['core_experiments']['h3_experiments']:
        dataset = exp_config['dataset']
        for aggregator in exp_config['aggregators']:
            for seed in exp_config['seeds']:
                results = run_single_experiment(
                    config=config,
                    dataset_name=dataset,
                    aggregator_name=aggregator,
                    heterogeneity=exp_config['heterogeneity'],
                    byzantine_fraction=exp_config['byzantine_fraction'],
                    attack_type=exp_config['attack'],
                    seed=seed,
                    exp_dir=exp_dir
                )
                all_results.append(results)
    
    # Aggregate and analyze results
    logger.info("\n" + "="*60)
    logger.info("Aggregating and analyzing results...")
    logger.info("="*60)
    
    metrics_aggregator = MetricsAggregator(all_results)
    summary = metrics_aggregator.generate_summary()
    
    # Save summary
    summary_path = exp_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Experiment matrix completed! Summary saved to {summary_path}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run CAAC-FL experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/core_experiments.yaml",
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single experiment instead of the full matrix"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mimic", "isic", "chestxray"],
        help="Dataset for single experiment"
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        choices=["fedavg", "krum", "fltrust", "caac_fl"],
        help="Aggregator for single experiment"
    )
    parser.add_argument(
        "--heterogeneity",
        type=str,
        choices=["mild", "extreme"],
        default="extreme",
        help="Heterogeneity level for single experiment"
    )
    parser.add_argument(
        "--byzantine-fraction",
        type=float,
        default=0.2,
        help="Byzantine fraction for single experiment"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="sign_flip",
        help="Attack type for single experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(Path(args.config))
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_experiment_logging(log_level)
    
    # Prepare experiment directory
    exp_name = config['experiment']['name']
    exp_dir = prepare_experiment_dir(config, exp_name)
    
    if args.single:
        # Run single experiment
        if not args.dataset or not args.aggregator:
            raise ValueError("--dataset and --aggregator required for single experiment")
        
        results = run_single_experiment(
            config=config,
            dataset_name=args.dataset,
            aggregator_name=args.aggregator,
            heterogeneity=args.heterogeneity,
            byzantine_fraction=args.byzantine_fraction,
            attack_type=args.attack,
            seed=args.seed,
            exp_dir=exp_dir
        )
        
        logger.info("Single experiment completed!")
    else:
        # Run full experiment matrix
        results = run_experiment_matrix(config, exp_dir)
        logger.info(f"All experiments completed! Results saved to {exp_dir}")


if __name__ == "__main__":
    main()