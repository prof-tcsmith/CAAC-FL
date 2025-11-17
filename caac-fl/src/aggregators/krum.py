"""
Krum Aggregator
Selects updates based on minimal sum of distances to other updates
"""

import torch
from typing import List, Optional
import logging

from .base import Aggregator

logger = logging.getLogger(__name__)


class KrumAggregator(Aggregator):
    """
    Krum aggregator for Byzantine-robust federated learning.
    
    Selects the update(s) with the smallest sum of squared distances
    to other updates.
    """
    
    def __init__(self, num_byzantine: int = 0, num_selected: int = 1):
        """
        Initialize Krum aggregator.
        
        Args:
            num_byzantine: Expected number of Byzantine clients
            num_selected: Number of updates to select (1 for Krum, >1 for Multi-Krum)
        """
        super().__init__()
        self.num_byzantine = num_byzantine
        self.num_selected = num_selected
        logger.info(f"Initialized Krum aggregator (f={num_byzantine}, m={num_selected})")
    
    def aggregate(
        self,
        updates: List[torch.Tensor],
        num_examples: Optional[List[int]] = None,
        round_num: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform Krum aggregation.
        
        Args:
            updates: List of client updates
            num_examples: Number of examples per client (ignored)
            round_num: Current round number
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Selected update or average of selected updates
        """
        n_clients = len(updates)
        
        if n_clients == 0:
            raise ValueError("No updates to aggregate")
        
        # Convert to tensors and flatten if needed
        flattened = []
        for u in updates:
            if not isinstance(u, torch.Tensor):
                u = torch.tensor(u)
            flattened.append(u.flatten())
        
        stacked = torch.stack(flattened)
        
        # If not enough clients, return simple average
        if n_clients <= 2 * self.num_byzantine + 2:
            logger.warning(f"Not enough clients for Krum (n={n_clients}, f={self.num_byzantine})")
            return stacked.mean(dim=0)
        
        # Compute pairwise distances
        distances = torch.cdist(stacked, stacked, p=2)
        
        # For each client, compute sum of distances to n-f-2 nearest neighbors
        n_neighbors = n_clients - self.num_byzantine - 2
        scores = []
        
        for i in range(n_clients):
            # Get distances to other clients
            dists = distances[i, :]
            # Exclude self (distance = 0)
            dists = torch.cat([dists[:i], dists[i+1:]])
            # Sort and take n-f-2 smallest
            sorted_dists, _ = torch.sort(dists)
            score = sorted_dists[:n_neighbors].sum()
            scores.append(score)
        
        scores = torch.tensor(scores)
        
        # Select clients with smallest scores
        _, selected_indices = torch.topk(scores, k=min(self.num_selected, n_clients), largest=False)
        
        logger.debug(f"Round {round_num}: Selected clients {selected_indices.tolist()}")
        
        # Return average of selected updates
        selected_updates = stacked[selected_indices]
        return selected_updates.mean(dim=0)