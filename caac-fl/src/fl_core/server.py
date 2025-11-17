"""
Federated Learning Server Implementation using Flower Framework
Implements the server-side logic for CAAC-FL and baseline methods
"""

import flwr as fl
from flwr.server.strategy import Strategy
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from datetime import datetime

from ..aggregators.base import Aggregator
from ..profiles.client_profile import ClientProfileManager
from ..utils.metrics import MetricsTracker
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class CAACFLStrategy(fl.server.strategy.Strategy):
    """
    Custom Flower Strategy implementing CAAC-FL algorithm.
    
    This strategy maintains per-client profiles and performs
    adaptive anomaly-aware aggregation as specified in the protocol.
    """
    
    def __init__(
        self,
        aggregator: Aggregator,
        num_clients: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        bootstrap_rounds: int = 10,
        config: Optional[Dict] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize CAAC-FL strategy.
        
        Args:
            aggregator: Aggregator instance (CAAC-FL, FedAvg, Krum, etc.)
            num_clients: Total number of clients in federation
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients to start
            bootstrap_rounds: Number of bootstrap rounds for CAAC-FL
            config: Additional configuration parameters
            checkpoint_dir: Directory for saving checkpoints
        """
        super().__init__()
        self.aggregator = aggregator
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.bootstrap_rounds = bootstrap_rounds
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize profile manager for CAAC-FL
        self.profile_manager = ClientProfileManager(
            num_clients=num_clients,
            bootstrap_rounds=bootstrap_rounds,
            config=self.config
        )
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Round counter
        self.current_round = 0
        
        logger.info(f"Initialized {self.__class__.__name__} with {num_clients} clients")
    
    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[fl.common.Parameters]:
        """
        Initialize global model parameters.
        
        This is called once at the beginning of the FL process.
        Override this to provide initial model parameters.
        
        Returns:
            Initial parameters or None
        """
        # TODO: Load initial model parameters from a pre-trained model or random initialization
        logger.info("Initializing global model parameters")
        return None
    
    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """
        Configure the training round.
        
        Samples clients and creates training instructions for each.
        
        Args:
            server_round: Current round number
            parameters: Current global model parameters
            client_manager: Manager for client connections
            
        Returns:
            List of (client, FitIns) tuples
        """
        self.current_round = server_round
        
        # Sample clients
        sample_size = max(
            int(self.fraction_fit * self.num_clients),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        
        # Create config for clients
        config = {
            "round": server_round,
            "local_epochs": self.config.get("local_epochs", 1),
            "batch_size": self.config.get("batch_size", 32),
            "learning_rate": self.config.get("learning_rate", 0.001),
            "is_bootstrap": server_round <= self.bootstrap_rounds,
        }
        
        # Add Byzantine client information if in attack mode
        if self.config.get("byzantine_config"):
            byzantine_config = self.config["byzantine_config"]
            if server_round >= byzantine_config.get("attack_start_round", 0):
                config["attack_type"] = byzantine_config.get("attack_type")
                config["byzantine_clients"] = byzantine_config.get("byzantine_client_ids", [])
        
        # Create FitIns for each client
        fit_ins = fl.common.FitIns(parameters, config)
        
        logger.info(f"Round {server_round}: Sampled {len(clients)} clients for training")
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """
        Aggregate training results from clients.
        
        This is where the main CAAC-FL aggregation logic happens.
        
        Args:
            server_round: Current round number
            results: Successful client results
            failures: Failed client results
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}
        
        logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")
        
        # Extract gradients and metadata
        client_updates = []
        client_ids = []
        num_examples = []
        
        for client, fit_res in results:
            client_id = int(client.cid)
            client_ids.append(client_id)
            client_updates.append(fit_res.parameters)
            num_examples.append(fit_res.num_examples)
        
        # Convert to tensors for aggregation
        # TODO: Proper parameter conversion based on model architecture
        
        # Update client profiles (for CAAC-FL)
        if hasattr(self.aggregator, 'requires_profiles') and self.aggregator.requires_profiles:
            self.profile_manager.update_profiles(
                client_ids=client_ids,
                gradients=client_updates,
                round_num=server_round
            )
            profiles = self.profile_manager.get_profiles(client_ids)
        else:
            profiles = None
        
        # Perform aggregation
        aggregated_params = self.aggregator.aggregate(
            updates=client_updates,
            num_examples=num_examples,
            round_num=server_round,
            profiles=profiles,
            is_bootstrap=(server_round <= self.bootstrap_rounds)
        )
        
        # Track metrics
        metrics = {
            "num_clients": len(results),
            "round": server_round,
            "total_examples": sum(num_examples),
        }
        
        # Add CAAC-FL specific metrics
        if profiles:
            anomaly_scores = [p.get_anomaly_score() for p in profiles]
            metrics.update({
                "mean_anomaly_score": np.mean(anomaly_scores),
                "max_anomaly_score": np.max(anomaly_scores),
                "min_anomaly_score": np.min(anomaly_scores),
            })
        
        self.metrics_tracker.log_round_metrics(server_round, metrics)
        
        # Checkpoint if needed
        if server_round % self.config.get("checkpoint_frequency", 10) == 0:
            self._save_checkpoint(server_round, aggregated_params)
        
        return aggregated_params, metrics
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """
        Configure the evaluation round.
        
        Samples clients for evaluation.
        
        Args:
            server_round: Current round number
            parameters: Current global model parameters
            client_manager: Manager for client connections
            
        Returns:
            List of (client, EvaluateIns) tuples
        """
        # Sample clients for evaluation
        sample_size = max(
            int(self.fraction_evaluate * self.num_clients),
            self.min_evaluate_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients
        )
        
        # Create evaluation config
        config = {
            "round": server_round,
            "batch_size": self.config.get("eval_batch_size", 64),
        }
        
        # Create EvaluateIns for each client
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        
        logger.info(f"Round {server_round}: Sampled {len(clients)} clients for evaluation")
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: Successful evaluation results
            failures: Failed evaluation results
            
        Returns:
            Aggregated loss and metrics
        """
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results")
            return None, {}
        
        # Aggregate evaluation metrics
        total_loss = 0.0
        total_examples = 0
        metrics_aggregated = {}
        
        for client, eval_res in results:
            total_loss += eval_res.loss * eval_res.num_examples
            total_examples += eval_res.num_examples
            
            # Aggregate other metrics
            for key, value in eval_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = 0.0
                metrics_aggregated[key] += value * eval_res.num_examples
        
        # Compute weighted averages
        avg_loss = total_loss / total_examples if total_examples > 0 else float('inf')
        
        for key in metrics_aggregated:
            metrics_aggregated[key] /= total_examples
        
        metrics_aggregated["loss"] = avg_loss
        metrics_aggregated["round"] = server_round
        
        logger.info(f"Round {server_round}: Evaluation loss = {avg_loss:.4f}")
        
        return avg_loss, metrics_aggregated
    
    def evaluate(
        self, server_round: int, parameters: fl.common.Parameters
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """
        Evaluate the global model on the server side.
        
        This is optional and only used if server has access to a validation set.
        
        Args:
            server_round: Current round number
            parameters: Current global model parameters
            
        Returns:
            Loss and metrics, or None
        """
        # TODO: Implement server-side evaluation if validation data is available
        return None
    
    def _save_checkpoint(self, round_num: int, parameters: fl.common.Parameters):
        """
        Save checkpoint of current model and profiles.
        
        Args:
            round_num: Current round number
            parameters: Model parameters to save
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pt"
        
        checkpoint = {
            "round": round_num,
            "parameters": parameters,
            "profiles": self.profile_manager.get_all_profiles() if hasattr(self, 'profile_manager') else None,
            "metrics": self.metrics_tracker.get_all_metrics(),
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }
        
        # TODO: Proper serialization of parameters
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def create_strategy(
    strategy_name: str,
    config: Dict,
    num_clients: int,
    checkpoint_dir: Optional[Path] = None
) -> Strategy:
    """
    Factory function to create FL strategies.
    
    Args:
        strategy_name: Name of strategy (caac_fl, fedavg, krum, fltrust)
        config: Configuration dictionary
        num_clients: Total number of clients
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Configured strategy instance
    """
    # Import aggregators
    from ..aggregators import create_aggregator
    
    # Create appropriate aggregator
    aggregator = create_aggregator(strategy_name, config)
    
    # Create strategy
    if strategy_name == "caac_fl":
        strategy = CAACFLStrategy(
            aggregator=aggregator,
            num_clients=num_clients,
            config=config,
            checkpoint_dir=checkpoint_dir,
            bootstrap_rounds=config.get("bootstrap_rounds", 10),
        )
    else:
        # Use standard FedAvg strategy with custom aggregator
        strategy = CAACFLStrategy(  # Can use same class with different aggregators
            aggregator=aggregator,
            num_clients=num_clients,
            config=config,
            checkpoint_dir=checkpoint_dir,
            bootstrap_rounds=0,  # No bootstrap for other methods
        )
    
    return strategy


def start_server(
    strategy: Strategy,
    num_rounds: int,
    server_address: str = "[::]:8080",
    config: Optional[Dict] = None,
) -> fl.server.history.History:
    """
    Start the Flower server.
    
    Args:
        strategy: FL strategy to use
        num_rounds: Number of training rounds
        server_address: Server address and port
        config: Additional server configuration
        
    Returns:
        Training history
    """
    logger.info(f"Starting FL server on {server_address} for {num_rounds} rounds")
    
    # Configure and start server
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    logger.info("FL server finished")
    
    return history