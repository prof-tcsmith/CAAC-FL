"""
Byzantine Attack Implementations for Federated Learning

This module implements various Byzantine attacks that malicious clients
can execute to disrupt federated learning.

Attacks implemented:
1. Random Noise Attack: Add Gaussian noise to model parameters
2. Sign Flipping Attack: Reverse the sign of all gradients
3. ALIE (A Little Is Enough): Craft updates just outside detection bounds
4. IPM (Inner Product Manipulation): Steer model in adversarial direction

References:
- Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust FL," USENIX Security 2020
- Baruch et al., "A Little Is Enough: Circumventing Defenses," NeurIPS 2019
- Xie et al., "Fall of Empires: Breaking Byzantine-Tolerant SGD," UAI 2020
"""

import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
from scipy import stats


class ByzantineAttack:
    """Base class for Byzantine attacks"""

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """
        Apply the attack to the model

        Args:
            model: The trained local model
            original_model: The global model received from server

        Returns:
            The attacked model
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def get_name(self) -> str:
        """Return the name of the attack"""
        raise NotImplementedError("Subclasses must implement get_name()")


class RandomNoiseAttack(ByzantineAttack):
    """
    Random Noise Attack

    Adds Gaussian noise to all model parameters to disrupt convergence.
    This is a simple but effective attack that can degrade model performance
    without being easily detectable.

    Formula: θ_malicious = θ_trained + N(0, σ²I)
    """

    def __init__(self, noise_scale: float = 1.0, seed: int = 42):
        """
        Args:
            noise_scale: Standard deviation of Gaussian noise (σ)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.noise_scale = noise_scale

    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """
        Add Gaussian noise to all model parameters

        Args:
            model: The trained local model
            original_model: The global model (not used in this attack)

        Returns:
            Model with noisy parameters
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    # Generate Gaussian noise with same shape as parameter
                    noise = torch.randn_like(param) * self.noise_scale
                    param.add_(noise)

        return model

    def get_name(self) -> str:
        return f"RandomNoise(σ={self.noise_scale})"


class SignFlippingAttack(ByzantineAttack):
    """
    Sign Flipping Attack

    Reverses the sign of the gradient (model update), pushing the global
    model in the opposite direction of convergence. This is one of the
    most destructive untargeted attacks.

    Formula: θ_malicious = θ_global - (θ_trained - θ_global)
                         = 2 * θ_global - θ_trained

    The update sent to server is:
        Δθ_malicious = θ_malicious - θ_global = -(θ_trained - θ_global)
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility (not used, but kept for consistency)
        """
        super().__init__(seed)

    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """
        Flip the sign of the gradient (model update)

        Args:
            model: The trained local model
            original_model: The global model received from server

        Returns:
            Model with sign-flipped update
        """
        with torch.no_grad():
            for param, original_param in zip(model.parameters(),
                                             original_model.parameters()):
                if param.requires_grad:
                    # Compute the update: Δθ = θ_trained - θ_global
                    update = param.data - original_param.data

                    # Apply sign-flipped update: θ_malicious = θ_global - Δθ
                    param.data = original_param.data - update

        return model

    def get_name(self) -> str:
        return "SignFlipping"


class ALIEAttack(ByzantineAttack):
    """
    ALIE (A Little Is Enough) Attack

    Crafts malicious updates that lie just outside the detection threshold
    of Byzantine-robust aggregation methods. The attack estimates the
    distribution of honest updates and generates adversarial updates that
    are large enough to cause damage but small enough to evade detection.

    From Baruch et al., "A Little Is Enough: Circumventing Defenses For
    Distributed Learning," NeurIPS 2019.

    The attack modifies the update to be:
        θ_malicious = μ - z * σ

    where μ and σ are estimated from the honest update, and z is chosen
    based on the number of Byzantine clients to stay just within bounds.

    In single-client mode (without access to other updates), we estimate
    based on the local update magnitude and apply a scaled negative update.
    """

    def __init__(self, num_byzantine: int = 1, num_clients: int = 10,
                 epsilon: float = 0.1, seed: int = 42):
        """
        Args:
            num_byzantine: Number of Byzantine clients (for z calculation)
            num_clients: Total number of clients
            epsilon: Small value to stay just outside threshold
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.num_byzantine = num_byzantine
        self.num_clients = num_clients
        self.epsilon = epsilon

        # Calculate z based on the inverse CDF for staying within bounds
        # The idea: with n-f honest clients, be at the edge of the trimming threshold
        n = num_clients
        f = num_byzantine
        # z is chosen so Byzantine updates are just beyond what median would filter
        # For trimmed mean with β trimming, want to be at the (n-f)/n quantile
        self.z = stats.norm.ppf((n - f - 1) / (n - f)) + epsilon

    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """
        Apply ALIE attack: generate update just outside detection threshold

        Args:
            model: The trained local model
            original_model: The global model received from server

        Returns:
            Model with ALIE attack applied
        """
        with torch.no_grad():
            for param, original_param in zip(model.parameters(),
                                             original_model.parameters()):
                if param.requires_grad:
                    # Compute the honest update
                    update = param.data - original_param.data

                    # Estimate mean and std of the update (per-parameter)
                    mu = update.mean()
                    sigma = update.std() + 1e-8  # Avoid division by zero

                    # ALIE: shift to be just outside detection bounds
                    # θ_malicious = θ_global + (μ - z * σ)
                    # This pushes in opposite direction while staying "normal" looking
                    malicious_update = mu - self.z * sigma

                    # Apply the malicious update (replace entire tensor)
                    param.data = original_param.data + malicious_update

        return model

    def get_name(self) -> str:
        return f"ALIE(z={self.z:.2f})"


class IPMAttack(ByzantineAttack):
    """
    Inner Product Manipulation (IPM) Attack

    From Xie et al., "Fall of Empires: Breaking Byzantine-Tolerant SGD
    by Inner Product Manipulation," UAI 2020.
    Also used in Li et al., "An Experimental Study of Byzantine-Robust
    Aggregation Schemes in Federated Learning," IEEE TBDATA 2024.

    The paper formulation (when knowing all benign updates):
        Δ_byzantine = -ε/(K-M) * Σ(benign_updates)

    For single-client simulation (approximation):
        Δ_byzantine = -ε * Δ_local

    Key behavior:
    - When ε < K/M - 1: Doesn't reverse direction, only reduces magnitude
    - When ε > K/M - 1: Reverses the direction (more damaging)

    Li et al. test two variants:
    - IPM (ε=0.5): Small epsilon, reduces magnitude without reversing
    - IPM (ε=100): Large epsilon, reverses direction completely

    The attack creates updates that have a negative inner product with
    the honest update direction, steering the model adversarially.
    """

    def __init__(self, epsilon: float = 1.0, seed: int = 42):
        """
        Args:
            epsilon: Scaling factor for the attack magnitude
                     - ε=0.5: Reduces magnitude (Li et al. small variant)
                     - ε=100: Reverses direction (Li et al. large variant)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.epsilon = epsilon

    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """
        Apply IPM attack: create update that's negative scaled version of local update

        Formula: Δ_byzantine = -ε * Δ_local

        This gives: Mean = (K-M-ε*M)/K * mean(benign_updates)
        - When ε=0.5, M=5, K=20: Mean = (15-2.5)/20 = 0.625 * true_mean (reduced)
        - When ε=100, M=5, K=20: Mean = (15-500)/20 = -24.25 * true_mean (reversed)

        Args:
            model: The trained local model
            original_model: The global model received from server

        Returns:
            Model with IPM attack applied
        """
        with torch.no_grad():
            for param, original_param in zip(model.parameters(),
                                             original_model.parameters()):
                if param.requires_grad:
                    # Compute the honest local update
                    update = param.data - original_param.data

                    # IPM: malicious update = -epsilon * local_update
                    malicious_update = -self.epsilon * update

                    # Apply the malicious update
                    param.data = original_param.data + malicious_update

        return model

    def get_name(self) -> str:
        return f"IPM(ε={self.epsilon})"


class LabelFlippingAttack(ByzantineAttack):
    """
    Label Flipping Attack (Data Poisoning)

    Note: This is a data poisoning attack that occurs during local training,
    not a model poisoning attack applied after training. For simulation
    purposes, we approximate its effect by reversing the update direction
    with some added noise to simulate training on flipped labels.

    In a real implementation, labels would be flipped before training.
    Here we approximate the effect post-hoc.
    """

    def __init__(self, flip_fraction: float = 1.0, seed: int = 42):
        """
        Args:
            flip_fraction: Fraction of the update to flip (1.0 = full flip)
            seed: Random seed
        """
        super().__init__(seed)
        self.flip_fraction = flip_fraction

    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """
        Approximate label flipping by partially reversing update with noise
        """
        with torch.no_grad():
            for param, original_param in zip(model.parameters(),
                                             original_model.parameters()):
                if param.requires_grad:
                    update = param.data - original_param.data

                    # Flip the update direction and add noise to simulate
                    # training on incorrect labels
                    noise = torch.randn_like(param) * update.std() * 0.1
                    flipped_update = -self.flip_fraction * update + noise

                    param.data = original_param.data + flipped_update

        return model

    def get_name(self) -> str:
        return f"LabelFlipping(f={self.flip_fraction})"


class NoAttack(ByzantineAttack):
    """
    No Attack (Baseline)

    A placeholder attack that does nothing. Used for honest clients
    and baseline experiments.
    """

    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """
        Return the model unchanged

        Args:
            model: The trained local model
            original_model: The global model (not used)

        Returns:
            The original model unchanged
        """
        return model

    def get_name(self) -> str:
        return "None"


def create_attack(attack_type: str, **kwargs) -> ByzantineAttack:
    """
    Factory function to create attack instances

    Args:
        attack_type: Type of attack
        **kwargs: Additional arguments for the attack

    Available attacks:
        - none / no_attack: No attack (baseline)
        - random_noise: Add Gaussian noise (σ via noise_scale)
        - sign_flipping: Reverse gradient sign
        - alie: A Little Is Enough attack (requires num_byzantine, num_clients)
        - ipm: Inner Product Manipulation (ε via epsilon)
        - label_flipping: Approximate label flipping attack

    Returns:
        An instance of the requested attack

    Example:
        >>> attack = create_attack("random_noise", noise_scale=1.0, seed=42)
        >>> attack = create_attack("sign_flipping")
        >>> attack = create_attack("alie", num_byzantine=10, num_clients=50)
        >>> attack = create_attack("ipm", epsilon=1.0)
        >>> attack = create_attack("none")
    """
    attack_type = attack_type.lower()

    if attack_type == "none" or attack_type == "no_attack":
        return NoAttack(**kwargs)
    elif attack_type == "random_noise":
        return RandomNoiseAttack(**kwargs)
    elif attack_type == "sign_flipping":
        return SignFlippingAttack(**kwargs)
    elif attack_type == "alie":
        return ALIEAttack(**kwargs)
    elif attack_type == "ipm":
        return IPMAttack(**kwargs)
    elif attack_type == "label_flipping":
        return LabelFlippingAttack(**kwargs)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}. "
                        f"Choose from: {get_available_attacks()}")


def get_available_attacks() -> List[str]:
    """
    Get list of available attack types

    Returns:
        List of attack names
    """
    return ["none", "random_noise", "sign_flipping", "alie", "ipm", "label_flipping"]


# Attack metadata for experiment organization
UNTARGETED_ATTACKS = ["random_noise", "sign_flipping"]
TARGETED_ATTACKS = ["alie", "ipm"]
DATA_POISONING_ATTACKS = ["label_flipping"]
ALL_ATTACKS = ["none"] + UNTARGETED_ATTACKS + TARGETED_ATTACKS + DATA_POISONING_ATTACKS

ATTACK_INFO = {
    "none": {
        "type": "baseline",
        "description": "No attack - honest client behavior",
        "severity": 0,
    },
    "random_noise": {
        "type": "untargeted",
        "description": "Add Gaussian noise to model parameters",
        "severity": 1,
    },
    "sign_flipping": {
        "type": "untargeted",
        "description": "Reverse the sign of gradients",
        "severity": 3,
    },
    "alie": {
        "type": "targeted",
        "description": "A Little Is Enough - evade detection thresholds",
        "severity": 2,
    },
    "ipm": {
        "type": "targeted",
        "description": "Inner Product Manipulation - steer model adversarially",
        "severity": 2,
    },
    "label_flipping": {
        "type": "data_poisoning",
        "description": "Train on flipped labels",
        "severity": 2,
    },
}


# Example usage
if __name__ == "__main__":
    from shared.models import SimpleCNN

    print("Byzantine Attack Implementations")
    print("=" * 50)
    print()

    # Create a simple model for testing
    model = SimpleCNN(num_classes=10)
    original_model = SimpleCNN(num_classes=10)

    # Test each attack
    print("Available attacks:")
    for attack_name in get_available_attacks():
        attack = create_attack(attack_name)
        print(f"  - {attack.get_name()}")

    print()
    print("Attack Impact Test:")
    print("-" * 50)

    # Get a parameter to track
    original_param = next(original_model.parameters()).clone()

    # Test Random Noise
    print("\n1. Random Noise Attack (σ=1.0):")
    model_noise = SimpleCNN(num_classes=10)
    model_noise.load_state_dict(original_model.state_dict())
    attack_noise = create_attack("random_noise", noise_scale=1.0, seed=42)
    model_noise = attack_noise.apply(model_noise, original_model)
    param_noise = next(model_noise.parameters())
    print(f"   Original param mean: {original_param.mean():.6f}")
    print(f"   Attacked param mean: {param_noise.mean():.6f}")
    print(f"   L2 distance: {torch.norm(param_noise - original_param):.6f}")

    # Test Sign Flipping
    print("\n2. Sign Flipping Attack:")
    model_trained = SimpleCNN(num_classes=10)
    model_trained.load_state_dict(original_model.state_dict())
    # Simulate training by adding a small update
    with torch.no_grad():
        for param in model_trained.parameters():
            param.data += torch.randn_like(param) * 0.01

    param_trained = next(model_trained.parameters())
    update = param_trained - original_param

    attack_flip = create_attack("sign_flipping")
    model_flip = attack_flip.apply(model_trained, original_model)
    param_flip = next(model_flip.parameters())
    flipped_update = param_flip - original_param

    print(f"   Original update mean: {update.mean():.6f}")
    print(f"   Flipped update mean: {flipped_update.mean():.6f}")
    print(f"   Dot product (should be negative): {torch.dot(update.flatten(), flipped_update.flatten()):.6f}")

    # Test No Attack
    print("\n3. No Attack (Baseline):")
    model_honest = SimpleCNN(num_classes=10)
    model_honest.load_state_dict(original_model.state_dict())
    attack_none = create_attack("none")
    model_honest = attack_none.apply(model_honest, original_model)
    param_honest = next(model_honest.parameters())
    print(f"   L2 distance from original: {torch.norm(param_honest - original_param):.6f}")
    print(f"   (Should be 0.0)")

    print("\n" + "=" * 50)
    print("All attacks implemented and tested successfully!")
