"""
Sign-Flipping Attack
Inverts and scales gradients to destabilize training
"""

import torch
from typing import List
import logging

from .base import Attack

logger = logging.getLogger(__name__)


class SignFlipAttack(Attack):
    """
    Sign-flipping attack.
    
    Multiplies gradients by a negative scale factor to push the model
    in the opposite direction.
    """
    
    def __init__(self, scale: float = 10.0):
        """
        Initialize sign-flipping attack.
        
        Args:
            scale: Scaling factor for the attack (higher = more aggressive)
        """
        super().__init__()
        self.scale = scale
        logger.info(f"Initialized sign-flip attack with scale={scale}")
    
    def apply(
        self,
        gradients: List[torch.Tensor],
        round_num: int = 0,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Apply sign-flipping attack to gradients.
        
        Args:
            gradients: List of gradient tensors
            round_num: Current round number
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Attacked gradients (sign-flipped and scaled)
        """
        attacked = []
        
        for grad in gradients:
            # Flip sign and scale
            attacked_grad = -self.scale * grad
            attacked.append(attacked_grad)
        
        logger.debug(f"Round {round_num}: Applied sign-flip attack with scale={self.scale}")
        
        return attacked