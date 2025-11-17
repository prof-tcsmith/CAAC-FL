"""
Attacks module
Byzantine attack implementations for federated learning
"""

from .base import Attack
from .sign_flip import SignFlipAttack
from .alie import ALIEAttack
from .slow_drift import SlowDriftAttack
from .random_noise import RandomNoiseAttack
from .ipm import InnerProductManipulationAttack

from typing import Dict, Any, Optional


def create_attack(attack_type: str, attack_config: Optional[Dict[str, Any]] = None) -> Attack:
    """
    Factory function to create attacks.
    
    Args:
        attack_type: Type of attack (sign_flip, alie, slow_drift, etc.)
        attack_config: Attack configuration parameters
        
    Returns:
        Attack instance
    """
    attack_type = attack_type.lower()
    config = attack_config or {}
    
    if attack_type == "sign_flip":
        return SignFlipAttack(
            scale=config.get("scale", 10.0)
        )
    elif attack_type == "alie":
        return ALIEAttack(
            z_score=config.get("z_score", 2.5),
            num_byzantine=config.get("num_byzantine", 1)
        )
    elif attack_type == "slow_drift":
        return SlowDriftAttack(
            start_round=config.get("start_round", 20),
            end_round=config.get("end_round", 50),
            target_direction=config.get("target_direction", None)
        )
    elif attack_type == "random_noise":
        return RandomNoiseAttack(
            noise_scale=config.get("noise_scale", 1.0)
        )
    elif attack_type == "ipm" or attack_type == "inner_product":
        return InnerProductManipulationAttack(
            epsilon=config.get("epsilon", 0.1)
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


__all__ = [
    "Attack",
    "SignFlipAttack",
    "ALIEAttack",
    "SlowDriftAttack",
    "RandomNoiseAttack",
    "InnerProductManipulationAttack",
    "create_attack",
]