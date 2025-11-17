"""
CAAC-FL Aggregator Implementation
Client-Adaptive Anomaly-Aware Clipping for Byzantine-Robust Federated Learning
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .base import Aggregator
from ..profiles.client_profile import ClientProfile

logger = logging.getLogger(__name__)


@dataclass
class CAACFLConfig:
    """Configuration for CAAC-FL aggregator."""
    
    # EWMA parameters
    beta: float = 0.9  # EWMA decay for statistics
    gamma: float = 0.1  # Reliability score smoothing
    
    # Anomaly weights
    lambda_mag: float = 0.4  # Weight for magnitude anomaly
    lambda_dir: float = 0.4  # Weight for directional anomaly
    lambda_temp: float = 0.2  # Weight for temporal anomaly
    
    # Thresholds
    tau_anomaly: float = 2.0  # Anomaly threshold for reliability
    f_min: float = 0.25  # Minimum threshold scaling factor
    f_max: float = 2.0  # Maximum threshold scaling factor
    alpha: float = 0.5  # Anomaly impact on threshold
    delta: float = 0.5  # Reliability impact on threshold
    
    # Aggregation
    eta_w: float = 0.5  # Weight scaling factor
    
    # Bootstrap
    bootstrap_rounds: int = 10  # Number of bootstrap rounds
    bootstrap_clip_factor: float = 0.5  # Clipping factor during bootstrap
    
    # Numerical stability
    epsilon: float = 1e-8


class CAACFLAggregator(Aggregator):
    """
    CAAC-FL aggregator implementing the canonical algorithm from the protocol.
    
    This aggregator maintains per-client behavioral profiles and performs
    adaptive anomaly-aware clipping and weighted aggregation.
    """
    
    def __init__(self, config: Optional[CAACFLConfig] = None):
        """
        Initialize CAAC-FL aggregator.
        
        Args:
            config: Configuration object, uses defaults if None
        """
        super().__init__()
        self.config = config or CAACFLConfig()
        self.requires_profiles = True
        
        # Store previous global gradient for directional comparison
        self.prev_global_gradient = None
        
        logger.info(f"Initialized CAAC-FL aggregator with config: {self.config}")
    
    def aggregate(
        self,
        updates: List[torch.Tensor],
        num_examples: Optional[List[int]] = None,
        round_num: int = 0,
        profiles: Optional[List[ClientProfile]] = None,
        is_bootstrap: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform CAAC-FL aggregation.
        
        Args:
            updates: List of client gradient updates
            num_examples: Number of examples per client (optional)
            round_num: Current round number
            profiles: Client behavioral profiles
            is_bootstrap: Whether in bootstrap phase
            **kwargs: Additional arguments
            
        Returns:
            Aggregated gradient update
        """
        n_clients = len(updates)
        
        if n_clients == 0:
            raise ValueError("No updates to aggregate")
        
        # Convert updates to tensors if needed
        updates_tensor = self._prepare_updates(updates)
        
        # Bootstrap phase: simple clipped averaging
        if is_bootstrap or round_num <= self.config.bootstrap_rounds:
            logger.info(f"Round {round_num}: Bootstrap aggregation")
            return self._bootstrap_aggregation(updates_tensor)
        
        # Check profiles availability
        if profiles is None or len(profiles) != n_clients:
            logger.warning("Profiles not available, falling back to bootstrap aggregation")
            return self._bootstrap_aggregation(updates_tensor)
        
        # Full CAAC-FL aggregation
        logger.info(f"Round {round_num}: Full CAAC-FL aggregation with {n_clients} clients")
        
        # Step 1: Compute norms and cosine similarities
        norms, cosines = self._compute_gradient_properties(updates_tensor)
        
        # Step 2: Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(
            norms=norms,
            cosines=cosines,
            profiles=profiles
        )
        
        # Step 3: Update reliability scores
        reliabilities = self._update_reliability_scores(
            anomaly_scores=anomaly_scores,
            profiles=profiles
        )
        
        # Step 4: Compute adaptive clipping thresholds
        thresholds = self._compute_thresholds(
            norms=norms,
            anomaly_scores=anomaly_scores,
            reliabilities=reliabilities
        )
        
        # Step 5: Clip gradients
        clipped_updates = self._clip_gradients(
            updates=updates_tensor,
            norms=norms,
            thresholds=thresholds
        )
        
        # Step 6: Compute aggregation weights
        weights = self._compute_weights(
            anomaly_scores=anomaly_scores,
            reliabilities=reliabilities
        )
        
        # Step 7: Weighted aggregation
        aggregated = self._weighted_aggregate(clipped_updates, weights)
        
        # Store for next round's directional comparison
        self.prev_global_gradient = aggregated.clone()
        
        # Log diagnostics
        self._log_diagnostics(
            round_num=round_num,
            anomaly_scores=anomaly_scores,
            reliabilities=reliabilities,
            weights=weights,
            thresholds=thresholds
        )
        
        return aggregated
    
    def _prepare_updates(self, updates: List[Any]) -> torch.Tensor:
        """
        Convert updates to tensor format.
        
        Args:
            updates: List of updates in various formats
            
        Returns:
            Stacked tensor of updates
        """
        tensors = []
        for update in updates:
            if isinstance(update, torch.Tensor):
                tensors.append(update.flatten())
            elif isinstance(update, np.ndarray):
                tensors.append(torch.from_numpy(update).flatten())
            elif isinstance(update, list):
                # Assume list of parameter tensors
                flat = torch.cat([torch.tensor(p).flatten() for p in update])
                tensors.append(flat)
            else:
                raise TypeError(f"Unsupported update type: {type(update)}")
        
        return torch.stack(tensors)
    
    def _bootstrap_aggregation(self, updates: torch.Tensor) -> torch.Tensor:
        """
        Simple aggregation for bootstrap phase.
        
        Uses global norm clipping and averaging.
        
        Args:
            updates: Tensor of client updates
            
        Returns:
            Aggregated update
        """
        # Compute norms
        norms = torch.norm(updates, dim=1)
        
        # Global median norm
        median_norm = torch.median(norms)
        
        # Clip to fraction of median
        clip_threshold = self.config.bootstrap_clip_factor * median_norm
        
        # Clip updates
        clipped = []
        for update, norm in zip(updates, norms):
            if norm > clip_threshold:
                clipped.append(update * (clip_threshold / (norm + self.config.epsilon)))
            else:
                clipped.append(update)
        
        # Simple average
        return torch.stack(clipped).mean(dim=0)
    
    def _compute_gradient_properties(
        self,
        updates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute norms and cosine similarities.
        
        Args:
            updates: Tensor of gradient updates
            
        Returns:
            Norms and cosine similarities
        """
        # Compute norms
        norms = torch.norm(updates, dim=1)
        
        # Compute cosine similarities with previous global gradient
        if self.prev_global_gradient is not None:
            prev_norm = torch.norm(self.prev_global_gradient)
            if prev_norm > 0:
                cosines = torch.matmul(updates, self.prev_global_gradient) / (
                    norms * prev_norm + self.config.epsilon
                )
            else:
                cosines = torch.ones(len(updates))
        else:
            cosines = torch.ones(len(updates))
        
        return norms, cosines
    
    def _compute_anomaly_scores(
        self,
        norms: torch.Tensor,
        cosines: torch.Tensor,
        profiles: List[ClientProfile]
    ) -> torch.Tensor:
        """
        Compute multi-dimensional anomaly scores.
        
        Args:
            norms: Gradient norms
            cosines: Cosine similarities
            profiles: Client profiles
            
        Returns:
            Composite anomaly scores
        """
        n_clients = len(norms)
        anomaly_scores = torch.zeros(n_clients)
        
        for i, profile in enumerate(profiles):
            # Magnitude anomaly
            mu_norm = profile.mu_norm if profile.mu_norm > 0 else norms[i].item()
            sigma_norm = profile.sigma_norm if profile.sigma_norm > 0 else 1.0
            a_mag = abs(norms[i].item() - mu_norm) / (sigma_norm + self.config.epsilon)
            
            # Directional anomaly (drop in cosine)
            rho = profile.rho_direction if profile.rounds_seen > 0 else cosines[i].item()
            a_dir = max(0, rho - cosines[i].item())
            
            # Temporal anomaly (change in norm EWMA)
            new_mu = self.config.beta * mu_norm + (1 - self.config.beta) * norms[i].item()
            if mu_norm > 0:
                a_temp = abs(new_mu - mu_norm) / (mu_norm + self.config.epsilon)
            else:
                a_temp = 0
            
            # Composite anomaly
            anomaly = np.sqrt(
                self.config.lambda_mag * a_mag**2 +
                self.config.lambda_dir * a_dir**2 +
                self.config.lambda_temp * a_temp**2
            )
            
            anomaly_scores[i] = anomaly
        
        return anomaly_scores
    
    def _update_reliability_scores(
        self,
        anomaly_scores: torch.Tensor,
        profiles: List[ClientProfile]
    ) -> torch.Tensor:
        """
        Update and return reliability scores.
        
        Args:
            anomaly_scores: Current anomaly scores
            profiles: Client profiles
            
        Returns:
            Updated reliability scores
        """
        reliabilities = torch.zeros(len(anomaly_scores))
        
        for i, (anomaly, profile) in enumerate(zip(anomaly_scores, profiles)):
            # EWMA update
            is_benign = float(anomaly < self.config.tau_anomaly)
            new_reliability = (
                (1 - self.config.gamma) * profile.reliability +
                self.config.gamma * is_benign
            )
            
            # Clip to [0, 1]
            new_reliability = max(0.0, min(1.0, new_reliability))
            
            # Update profile
            profile.reliability = new_reliability
            reliabilities[i] = new_reliability
        
        return reliabilities
    
    def _compute_thresholds(
        self,
        norms: torch.Tensor,
        anomaly_scores: torch.Tensor,
        reliabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive clipping thresholds.
        
        Args:
            norms: Gradient norms
            anomaly_scores: Anomaly scores
            reliabilities: Reliability scores
            
        Returns:
            Per-client clipping thresholds
        """
        # Global median norm
        median_norm = torch.median(norms)
        
        # Compute scale factors
        scale_factors = torch.exp(-self.config.alpha * anomaly_scores) * (
            1 + self.config.delta * reliabilities
        )
        
        # Clamp scale factors
        scale_factors = torch.clamp(scale_factors, self.config.f_min, self.config.f_max)
        
        # Compute thresholds
        thresholds = median_norm * scale_factors
        
        return thresholds
    
    def _clip_gradients(
        self,
        updates: torch.Tensor,
        norms: torch.Tensor,
        thresholds: torch.Tensor
    ) -> torch.Tensor:
        """
        Clip gradient updates.
        
        Args:
            updates: Gradient updates
            norms: Gradient norms
            thresholds: Clipping thresholds
            
        Returns:
            Clipped updates
        """
        clipped = []
        
        for update, norm, threshold in zip(updates, norms, thresholds):
            if norm > threshold:
                clipped.append(update * (threshold / (norm + self.config.epsilon)))
            else:
                clipped.append(update)
        
        return torch.stack(clipped)
    
    def _compute_weights(
        self,
        anomaly_scores: torch.Tensor,
        reliabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute aggregation weights.
        
        Args:
            anomaly_scores: Anomaly scores
            reliabilities: Reliability scores
            
        Returns:
            Normalized weights
        """
        # Raw weights
        raw_weights = reliabilities * torch.exp(-self.config.eta_w * anomaly_scores)
        
        # Normalize
        weights = raw_weights / (torch.sum(raw_weights) + self.config.epsilon)
        
        return weights
    
    def _weighted_aggregate(
        self,
        updates: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform weighted aggregation.
        
        Args:
            updates: Clipped gradient updates
            weights: Aggregation weights
            
        Returns:
            Weighted average
        """
        # Reshape weights for broadcasting
        weights = weights.view(-1, 1)
        
        # Weighted sum
        aggregated = torch.sum(updates * weights, dim=0)
        
        return aggregated
    
    def _log_diagnostics(
        self,
        round_num: int,
        anomaly_scores: torch.Tensor,
        reliabilities: torch.Tensor,
        weights: torch.Tensor,
        thresholds: torch.Tensor
    ):
        """
        Log diagnostic information.
        
        Args:
            round_num: Current round
            anomaly_scores: Anomaly scores
            reliabilities: Reliability scores
            weights: Aggregation weights
            thresholds: Clipping thresholds
        """
        logger.info(
            f"Round {round_num} CAAC-FL diagnostics:\n"
            f"  Anomaly scores: mean={anomaly_scores.mean():.3f}, "
            f"max={anomaly_scores.max():.3f}, min={anomaly_scores.min():.3f}\n"
            f"  Reliabilities: mean={reliabilities.mean():.3f}, "
            f"max={reliabilities.max():.3f}, min={reliabilities.min():.3f}\n"
            f"  Weights: max={weights.max():.3f}, min={weights.min():.3f}, "
            f"entropy={-(weights * torch.log(weights + 1e-8)).sum():.3f}\n"
            f"  Thresholds: mean={thresholds.mean():.3f}, "
            f"std={thresholds.std():.3f}"
        )