"""
Base Aggregator Class
Abstract base class for all federated learning aggregation methods
"""

from abc import ABC, abstractmethod
import torch
from typing import List, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class Aggregator(ABC):
    """
    Abstract base class for federated aggregation methods.
    
    All aggregators must implement the aggregate method.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.requires_profiles = False  # Most aggregators don't need profiles
    
    @abstractmethod
    def aggregate(
        self,
        updates: List[torch.Tensor],
        num_examples: Optional[List[int]] = None,
        round_num: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate client updates into a global update.
        
        Args:
            updates: List of client gradient/parameter updates
            num_examples: Number of examples per client (for weighted averaging)
            round_num: Current training round
            **kwargs: Additional method-specific parameters
            
        Returns:
            Aggregated update tensor
        """
        pass
    
    def reset(self):
        """
        Reset internal state.
        
        Called at the beginning of each experiment.
        """
        pass
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information from the aggregator.
        
        Returns:
            Dictionary of diagnostic metrics
        """
        return {}