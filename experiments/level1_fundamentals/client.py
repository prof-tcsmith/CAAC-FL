"""
Flower client implementation for Level 1.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple

from shared.models import SimpleCNN
from shared.metrics import train_model, evaluate_model


class CifarClient(fl.client.NumPyClient):
    """
    Flower client for CIFAR-10 federated learning.
    """

    def __init__(
        self,
        cid: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 0.01,
        local_epochs: int = 1
    ):
        """
        Initialize client.

        Args:
            cid: Client ID
            train_loader: DataLoader for training
            test_loader: DataLoader for testing
            device: Device to train on
            learning_rate: Learning rate for optimizer
            local_epochs: Number of local training epochs
        """
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        # Initialize model
        self.model = SimpleCNN(num_classes=10)
        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data.

        Args:
            parameters: Global model parameters
            config: Training configuration

        Returns:
            Updated parameters, number of samples, metrics
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Train model
        stats = train_model(
            self.model,
            self.train_loader,
            self.optimizer,
            self.criterion,
            device=self.device,
            epochs=self.local_epochs
        )

        # Return updated parameters and statistics
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                'train_loss': stats['train_loss'],
                'train_accuracy': stats['train_accuracy']
            }
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Loss, number of samples, metrics
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Evaluate model
        results = evaluate_model(
            self.model,
            self.test_loader,
            device=self.device,
            criterion=self.criterion
        )

        return (
            results['loss'],
            results['num_samples'],
            {'accuracy': results['accuracy']}
        )


def create_client_fn(
    train_loaders: Dict[int, DataLoader],
    test_loader: DataLoader,
    device: str = 'cpu',
    learning_rate: float = 0.01,
    local_epochs: int = 1
):
    """
    Create a client factory function for Flower.

    Args:
        train_loaders: Dict mapping client ID to training DataLoader
        test_loader: Test DataLoader (shared)
        device: Device to train on
        learning_rate: Learning rate
        local_epochs: Number of local epochs

    Returns:
        Client factory function
    """
    def client_fn(cid: str) -> CifarClient:
        """Return a client instance for the given client ID."""
        client_id = int(cid)
        return CifarClient(
            cid=client_id,
            train_loader=train_loaders[client_id],
            test_loader=test_loader,
            device=device,
            learning_rate=learning_rate,
            local_epochs=local_epochs
        )

    return client_fn
