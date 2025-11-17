"""
Federated Learning Client Implementation using Flower Framework
Implements client-side logic for training and evaluation
"""

import flwr as fl
from flwr.client import NumPyClient
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import logging

from ..models import create_model
from ..datasets import get_client_dataloader
from ..attacks import create_attack
from ..utils.training import train_epoch, evaluate
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class FederatedClient(NumPyClient):
    """
    Flower client implementation for federated learning.
    
    Handles local training, evaluation, and potential Byzantine behavior.
    """
    
    def __init__(
        self,
        client_id: int,
        model_name: str,
        dataset_name: str,
        data_path: Path,
        device: str = "cpu",
        is_byzantine: bool = False,
        attack_config: Optional[Dict] = None,
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model_name: Name of model architecture to use
            dataset_name: Name of dataset (mimic, isic, chestxray)
            data_path: Path to client's local data
            device: Device to use for training (cpu/cuda)
            is_byzantine: Whether this client is Byzantine
            attack_config: Configuration for Byzantine attacks
        """
        super().__init__()
        self.client_id = client_id
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.device = torch.device(device)
        self.is_byzantine = is_byzantine
        self.attack_config = attack_config or {}
        
        # Initialize model
        self.model = create_model(model_name, dataset_name).to(self.device)
        
        # Get data loaders for this client
        self.train_loader = get_client_dataloader(
            dataset_name=dataset_name,
            client_id=client_id,
            data_path=data_path,
            split="train",
            batch_size=32,
        )
        self.val_loader = get_client_dataloader(
            dataset_name=dataset_name,
            client_id=client_id,
            data_path=data_path,
            split="val",
            batch_size=64,
        )
        
        # Initialize attack if Byzantine
        if self.is_byzantine:
            self.attack = create_attack(
                attack_type=attack_config.get("type", "sign_flip"),
                attack_config=attack_config
            )
        else:
            self.attack = None
        
        # Training components
        self.criterion = self._get_criterion()
        self.optimizer = None  # Created fresh each round
        
        # Track training history
        self.history = {
            "loss": [],
            "accuracy": [],
            "rounds": [],
        }
        
        logger.info(f"Initialized client {client_id} (Byzantine: {is_byzantine})")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Return current model parameters as a list of NumPy arrays.
        
        Args:
            config: Configuration dictionary from server
            
        Returns:
            List of parameter arrays
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Update model parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of parameter arrays from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Current global model parameters
            config: Training configuration from server
            
        Returns:
            Updated parameters, number of examples, and metrics
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Extract training config
        round_num = config.get("round", 0)
        local_epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 0.001)
        is_bootstrap = config.get("is_bootstrap", False)
        
        logger.info(f"Client {self.client_id} starting round {round_num} training")
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Store initial parameters for gradient computation
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Local training
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(local_epochs):
            epoch_loss, epoch_acc, num_samples = train_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device
            )
            total_loss += epoch_loss * num_samples
            total_correct += epoch_acc * num_samples
            total_samples += num_samples
        
        # Compute gradient update
        gradients = []
        with torch.no_grad():
            for initial, current in zip(initial_params, self.model.parameters()):
                gradients.append(current - initial)
        
        # Apply Byzantine attack if applicable
        if self.is_byzantine and not is_bootstrap:
            attack_type = config.get("attack_type")
            if attack_type and round_num >= self.attack_config.get("start_round", 0):
                logger.warning(f"Client {self.client_id} applying {attack_type} attack")
                gradients = self.attack.apply(
                    gradients=gradients,
                    round_num=round_num,
                    model=self.model,
                    dataloader=self.train_loader
                )
                
                # Update model parameters with attacked gradients
                with torch.no_grad():
                    for param, initial, grad in zip(self.model.parameters(), initial_params, gradients):
                        param.data = initial + grad
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config)
        
        # Compute metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            "loss": float(avg_loss),
            "accuracy": float(avg_acc),
            "client_id": self.client_id,
            "is_byzantine": self.is_byzantine,
        }
        
        # Store history
        self.history["loss"].append(avg_loss)
        self.history["accuracy"].append(avg_acc)
        self.history["rounds"].append(round_num)
        
        logger.info(
            f"Client {self.client_id} round {round_num}: "
            f"loss={avg_loss:.4f}, acc={avg_acc:.4f}"
        )
        
        return updated_parameters, total_samples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Current global model parameters
            config: Evaluation configuration from server
            
        Returns:
            Loss, number of examples, and metrics
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        round_num = config.get("round", 0)
        
        # Evaluate on validation data
        loss, accuracy, auroc, auprc, num_samples = evaluate(
            model=self.model,
            dataloader=self.val_loader,
            criterion=self.criterion,
            device=self.device,
            compute_auroc=True
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "auroc": float(auroc) if auroc is not None else 0.0,
            "auprc": float(auprc) if auprc is not None else 0.0,
            "client_id": self.client_id,
        }
        
        logger.info(
            f"Client {self.client_id} round {round_num} evaluation: "
            f"loss={loss:.4f}, acc={accuracy:.4f}"
        )
        
        return float(loss), num_samples, metrics
    
    def _get_criterion(self) -> nn.Module:
        """
        Get loss function based on dataset.
        
        Returns:
            Loss function
        """
        if self.dataset_name in ["mimic", "isic"]:
            # Binary classification
            return nn.BCEWithLogitsLoss()
        elif self.dataset_name == "chestxray":
            # Multi-label classification
            return nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification
            return nn.CrossEntropyLoss()


def create_client_fn(
    client_configs: Dict[int, Dict],
    model_name: str,
    dataset_name: str,
    data_dir: Path,
    device: str = "cpu",
    byzantine_config: Optional[Dict] = None,
):
    """
    Create a client factory function for Flower.
    
    Args:
        client_configs: Configuration for each client
        model_name: Model architecture name
        dataset_name: Dataset name
        data_dir: Root data directory
        device: Device for training
        byzantine_config: Byzantine attack configuration
        
    Returns:
        Client factory function
    """
    def client_fn(cid: str) -> fl.client.Client:
        """
        Factory function to create a client.
        
        Args:
            cid: Client ID as string
            
        Returns:
            Configured client instance
        """
        client_id = int(cid)
        
        # Check if this client is Byzantine
        is_byzantine = False
        attack_config = None
        
        if byzantine_config:
            byzantine_ids = byzantine_config.get("byzantine_client_ids", [])
            if client_id in byzantine_ids:
                is_byzantine = True
                attack_config = byzantine_config
        
        # Create client
        client = FederatedClient(
            client_id=client_id,
            model_name=model_name,
            dataset_name=dataset_name,
            data_path=data_dir / f"client_{client_id}",
            device=device,
            is_byzantine=is_byzantine,
            attack_config=attack_config,
        )
        
        return client
    
    return client_fn


def start_client(
    server_address: str,
    client_id: int,
    model_name: str,
    dataset_name: str,
    data_path: Path,
    device: str = "cpu",
    is_byzantine: bool = False,
    attack_config: Optional[Dict] = None,
) -> None:
    """
    Start a Flower client and connect to server.
    
    Args:
        server_address: Address of the FL server
        client_id: Client identifier
        model_name: Model architecture name
        dataset_name: Dataset name
        data_path: Path to client data
        device: Device for training
        is_byzantine: Whether client is Byzantine
        attack_config: Attack configuration
    """
    logger.info(f"Starting client {client_id}, connecting to {server_address}")
    
    # Create client
    client = FederatedClient(
        client_id=client_id,
        model_name=model_name,
        dataset_name=dataset_name,
        data_path=data_path,
        device=device,
        is_byzantine=is_byzantine,
        attack_config=attack_config,
    )
    
    # Start client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )
    
    logger.info(f"Client {client_id} finished")