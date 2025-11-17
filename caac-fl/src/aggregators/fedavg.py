"""
FedAvg Aggregator
Standard Federated Averaging (non-robust baseline)
"""

import torch
from typing import List, Optional
import logging

from .base import Aggregator

logger = logging.getLogger(__name__)


class FedAvgAggregator(Aggregator):
    """
    Federated Averaging aggregator.
    
    Computes weighted average of client updates based on number of examples.
    """
    
    def __init__(self):
        """Initialize FedAvg aggregator."""
        super().__init__()
        logger.info("Initialized FedAvg aggregator")
    
    def aggregate(
        self,
        updates: List[torch.Tensor],
        num_examples: Optional[List[int]] = None,
        round_num: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform FedAvg aggregation.
        
        Args:
            updates: List of client updates
            num_examples: Number of examples per client
            round_num: Current round number
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Weighted average of updates
        """
        n_clients = len(updates)
        
        if n_clients == 0:
            raise ValueError("No updates to aggregate")
        
        # Convert to tensors if needed
        if not isinstance(updates[0], torch.Tensor):
            updates = [torch.tensor(u) if not isinstance(u, torch.Tensor) else u for u in updates]
        
        # Stack updates
        stacked = torch.stack(updates)
        
        if num_examples is None:
            # Simple average if no example counts provided
            logger.debug(f"Round {round_num}: Simple averaging {n_clients} updates")
            return stacked.mean(dim=0)
        else:
            # Weighted average based on number of examples
            weights = torch.tensor(num_examples, dtype=torch.float32)
            weights = weights / weights.sum()
            weights = weights.view(-1, 1)
            
            logger.debug(f"Round {round_num}: Weighted averaging {n_clients} updates")
            return (stacked * weights).sum(dim=0)