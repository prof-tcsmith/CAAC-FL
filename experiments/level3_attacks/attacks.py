"""
Byzantine Attack Implementations for Federated Learning

This module implements various Byzantine attacks that malicious clients
can execute to disrupt federated learning.

Attacks implemented:
1. Random Noise Attack: Add Gaussian noise to model parameters
2. Sign Flipping Attack: Reverse the sign of all gradients

References:
- Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust FL," USENIX Security 2020
- Baruch et al., "A Little Is Enough: Circumventing Defenses," NeurIPS 2019
"""

import numpy as np
from typing import List
import torch
import torch.nn as nn


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
        attack_type: Type of attack ("none", "random_noise", "sign_flipping")
        **kwargs: Additional arguments for the attack

    Returns:
        An instance of the requested attack

    Example:
        >>> attack = create_attack("random_noise", noise_scale=1.0, seed=42)
        >>> attack = create_attack("sign_flipping")
        >>> attack = create_attack("none")
    """
    attack_type = attack_type.lower()

    if attack_type == "none" or attack_type == "no_attack":
        return NoAttack(**kwargs)
    elif attack_type == "random_noise":
        return RandomNoiseAttack(**kwargs)
    elif attack_type == "sign_flipping":
        return SignFlippingAttack(**kwargs)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}. "
                        f"Choose from: none, random_noise, sign_flipping")


def get_available_attacks() -> List[str]:
    """
    Get list of available attack types

    Returns:
        List of attack names
    """
    return ["none", "random_noise", "sign_flipping"]


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
