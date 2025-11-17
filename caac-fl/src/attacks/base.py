"""
Base Attack Class
Abstract base class for Byzantine attacks
"""

from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Attack(ABC):
    """
    Abstract base class for Byzantine attacks.
    
    All attacks must implement the apply method.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize attack.
        
        Args:
            config: Attack configuration parameters
        """
        self.config = config or {}
    
    @abstractmethod
    def apply(
        self,
        gradients: List[torch.Tensor],
        round_num: int = 0,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Apply attack to gradients.
        
        Args:
            gradients: List of gradient tensors to attack
            round_num: Current round number
            **kwargs: Additional attack-specific parameters
            
        Returns:
            Attacked gradients
        """
        pass
    
    def reset(self):
        """
        Reset internal state.
        
        Called at the beginning of each experiment.
        """
        pass