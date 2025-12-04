"""
Federated Learning Client with Byzantine Attack Support

This client implementation supports both honest and Byzantine behavior,
allowing simulation of attacks in federated learning.

================================================================================
DELAYED COMPROMISE THREAT MODEL
================================================================================

This implementation supports the "delayed compromise" threat model where:

1. **Initial Honest Period**: All clients (including future Byzantine ones) behave
   honestly during an initial warmup period (rounds 0 to compromise_round-1).

2. **Profile Building**: During the honest period, the server builds behavioral
   profiles for each client based on their legitimate gradient patterns.

3. **Compromise Event**: At `compromise_round`, designated Byzantine clients
   become compromised and begin executing their attack strategy.

4. **Detection Opportunity**: Since the attack behavior deviates from the
   established honest profile, profile-based detection (like CAAC-FL) should
   be able to identify the behavioral shift.

This threat model is more realistic than immediate-attack scenarios because:
- Real-world compromises often happen after initial deployment
- Defenders have time to establish baseline behavior
- Profile-based detection has historical data to compare against

Configuration:
- `compromise_round`: The round at which Byzantine behavior begins (default: 0)
- If compromise_round=0, behavior matches the traditional immediate-attack model
- Recommended: compromise_round >= warmup_rounds (e.g., 10-15) for CAAC-FL

Example:
    # Byzantine clients behave honestly for rounds 0-14, attack from round 15
    client = FlowerClient(..., is_byzantine=True, compromise_round=15)

================================================================================
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
    Flower client with Byzantine attack support and delayed compromise model.

    This client supports the delayed compromise threat model where Byzantine
    clients initially behave honestly to allow profile building, then begin
    attacking at a specified round.

    Args:
        model: The neural network model
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        device: Device to run training on
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for optimizer
        client_id: Unique identifier for this client
        is_byzantine: Whether this client will become Byzantine (malicious)
        attack: Attack strategy to apply (when Byzantine and past compromise_round)
        compromise_round: Round at which Byzantine behavior begins (default: 0)
            - If 0: Attack from the first round (traditional model)
            - If >0: Behave honestly until this round, then attack
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
        compromise_round: int = 0,
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

        # Delayed compromise support
        self.compromise_round = compromise_round
        self.current_round = 0  # Track rounds for delayed compromise

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
        # Only needed if this client will eventually be Byzantine
        if self.is_byzantine:
            self.original_model = type(self.model)(num_classes=10).to(self.device)
            self.original_model.load_state_dict(self.model.state_dict())

    def _should_attack(self, config: Dict) -> bool:
        """
        Determine if this client should apply Byzantine attack this round.

        Implements the delayed compromise threat model:
        - Returns False if not Byzantine
        - Returns False if current round < compromise_round (honest period)
        - Returns True if Byzantine and current round >= compromise_round

        Args:
            config: Training configuration (may contain 'server_round')

        Returns:
            True if attack should be applied this round
        """
        if not self.is_byzantine:
            return False

        # Get current round from config if available, otherwise use internal counter
        server_round = config.get('server_round', self.current_round)

        # Check if we've passed the compromise round
        return server_round >= self.compromise_round

    def fit(self, parameters: list, config: Dict) -> tuple:
        """
        Train the model and optionally apply Byzantine attack.

        Implements delayed compromise: Byzantine clients behave honestly
        until compromise_round, then begin attacking.

        Args:
            parameters: Model parameters from server
            config: Training configuration (includes 'server_round')

        Returns:
            Updated parameters, number of examples, and metrics
        """
        # Update round tracking
        server_round = config.get('server_round', self.current_round)
        self.current_round = server_round

        # Set parameters from server
        self.set_parameters(parameters)

        # Perform local training (always honest training)
        train_loss = self._train()

        # Determine if attack should be applied (delayed compromise check)
        apply_attack = self._should_attack(config)

        # Apply Byzantine attack if conditions are met
        if apply_attack and self.attack is not None:
            # Apply attack to model
            self.model = self.attack.apply(self.model, self.original_model)

        # Get updated parameters
        updated_parameters = self.get_parameters(config)

        # Return results with attack status for logging
        num_examples = len(self.trainloader.dataset)
        metrics = {
            "loss": float(train_loss),
            "client_id": self.client_id,
            "is_byzantine": int(self.is_byzantine),
            "attack_applied": int(apply_attack),  # Track if attack was applied this round
        }

        # Increment internal round counter for next round
        self.current_round += 1

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
    compromise_round: int = 0,
) -> FlowerClient:
    """
    Factory function to create a Flower client with delayed compromise support.

    Args:
        client_id: Unique identifier for this client
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        device: Device to run training on
        is_byzantine: Whether this client will become Byzantine
        attack: Attack strategy (if Byzantine)
        local_epochs: Number of local training epochs
        learning_rate: Learning rate
        compromise_round: Round at which Byzantine behavior begins (default: 0)
            - Set to 0 for immediate attack (traditional threat model)
            - Set to >0 for delayed compromise (recommended: >= warmup_rounds)

    Returns:
        Configured FlowerClient instance

    Example:
        # Traditional immediate attack
        client = create_client(..., is_byzantine=True, compromise_round=0)

        # Delayed compromise: honest for 15 rounds, then attack
        client = create_client(..., is_byzantine=True, compromise_round=15)
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
        compromise_round=compromise_round,
    )
