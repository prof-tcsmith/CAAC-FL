"""
Federated Learning Client with Byzantine Attack Support

This client implementation supports both honest and Byzantine behavior,
allowing simulation of attacks in federated learning.
"""

# Suppress PyTorch pin_memory deprecation warnings (from PyTorch internals)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from flwr.client import NumPyClient
from attacks import ByzantineAttack, NoAttack


class FlowerClient(NumPyClient):
    """
    Flower client with Byzantine attack support

    Args:
        model: The neural network model
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        device: Device to run training on
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for optimizer
        client_id: Unique identifier for this client
        is_byzantine: Whether this client is Byzantine (malicious)
        attack: Attack strategy to apply (if Byzantine)
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        client_id: int = 0,
        is_byzantine: bool = False,
        attack: Optional[ByzantineAttack] = None,
    ):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.client_id = client_id
        self.is_byzantine = is_byzantine
        self.attack = attack if attack is not None else NoAttack()

        # Store original model for sign flipping attack
        self.original_model = None

    def get_parameters(self, config: Dict) -> list:
        """Return current model parameters"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list) -> None:
        """
        Set model parameters and store original model state

        Args:
            parameters: Model parameters from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Store a copy of the original model for attacks that need it
        if self.is_byzantine:
            self.original_model = type(self.model)(num_classes=10).to(self.device)
            self.original_model.load_state_dict(self.model.state_dict())

    def fit(self, parameters: list, config: Dict) -> tuple:
        """
        Train the model and optionally apply Byzantine attack

        Args:
            parameters: Model parameters from server
            config: Training configuration

        Returns:
            Updated parameters, number of examples, and metrics
        """
        # Set parameters from server
        self.set_parameters(parameters)

        # Perform local training
        train_loss = self._train()

        # Apply Byzantine attack if this is a malicious client
        if self.is_byzantine and self.attack is not None:
            # Apply attack to model
            self.model = self.attack.apply(self.model, self.original_model)

        # Get updated parameters
        updated_parameters = self.get_parameters(config)

        # Return results
        num_examples = len(self.trainloader.dataset)
        metrics = {
            "loss": float(train_loss),
            "client_id": self.client_id,
            "is_byzantine": int(self.is_byzantine),
        }

        return updated_parameters, num_examples, metrics

    def evaluate(self, parameters: list, config: Dict) -> tuple:
        """
        Evaluate the model

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Loss, number of examples, and metrics (including accuracy)
        """
        self.set_parameters(parameters)

        loss, accuracy = self._test()

        num_examples = len(self.testloader.dataset)
        metrics = {"accuracy": float(accuracy)}

        return float(loss), num_examples, metrics

    def _train(self) -> float:
        """
        Train the model locally

        Returns:
            Average training loss
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _test(self) -> tuple:
        """
        Evaluate the model

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.testloader) if len(self.testloader) > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy


def create_client(
    client_id: int,
    trainloader: DataLoader,
    testloader: DataLoader,
    device: torch.device,
    is_byzantine: bool = False,
    attack: Optional[ByzantineAttack] = None,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
) -> FlowerClient:
    """
    Factory function to create a Flower client

    Args:
        client_id: Unique identifier for this client
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        device: Device to run training on
        is_byzantine: Whether this client is Byzantine
        attack: Attack strategy (if Byzantine)
        local_epochs: Number of local training epochs
        learning_rate: Learning rate

    Returns:
        Configured FlowerClient instance
    """
    from shared.models import SimpleCNN

    model = SimpleCNN(num_classes=10)

    return FlowerClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        client_id=client_id,
        is_byzantine=is_byzantine,
        attack=attack,
    )
