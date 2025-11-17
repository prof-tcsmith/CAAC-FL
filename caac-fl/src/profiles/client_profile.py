"""
Client Profile Management for CAAC-FL
Maintains behavioral statistics for each client
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClientProfile:
    """
    Behavioral profile for a single client.
    
    Tracks EWMA statistics and anomaly history as per CAAC-FL protocol.
    """
    
    client_id: int
    
    # EWMA statistics
    mu_norm: float = 0.0  # EWMA of gradient norm
    sigma_norm: float = 1.0  # EWMA of norm standard deviation
    rho_direction: float = 0.0  # EWMA of cosine similarity
    reliability: float = 0.5  # Reliability score [0, 1]
    
    # Tracking
    rounds_seen: int = 0  # Number of rounds participated
    last_round: int = -1  # Last round participated
    
    # History (for temporal analysis)
    norm_history: List[float] = field(default_factory=list)
    cosine_history: List[float] = field(default_factory=list)
    anomaly_history: List[float] = field(default_factory=list)
    
    # Configuration
    max_history_len: int = 20  # Maximum history length to maintain
    
    def update(
        self,
        norm: float,
        cosine: float,
        anomaly: float,
        round_num: int,
        beta: float = 0.9
    ):
        """
        Update profile statistics.
        
        Args:
            norm: Current gradient norm
            cosine: Current cosine similarity
            anomaly: Current anomaly score
            round_num: Current round number
            beta: EWMA decay factor
        """
        # Update EWMA statistics
        if self.rounds_seen == 0:
            # First round - initialize
            self.mu_norm = norm
            self.sigma_norm = 0.1 * norm  # Initial estimate
            self.rho_direction = cosine
        else:
            # EWMA updates
            old_mu = self.mu_norm
            self.mu_norm = beta * self.mu_norm + (1 - beta) * norm
            
            # Update variance estimate
            variance = beta * (self.sigma_norm ** 2) + (1 - beta) * ((norm - self.mu_norm) ** 2)
            self.sigma_norm = np.sqrt(max(variance, 1e-8))
            
            self.rho_direction = beta * self.rho_direction + (1 - beta) * cosine
        
        # Update history
        self.norm_history.append(norm)
        self.cosine_history.append(cosine)
        self.anomaly_history.append(anomaly)
        
        # Trim history if needed
        if len(self.norm_history) > self.max_history_len:
            self.norm_history = self.norm_history[-self.max_history_len:]
            self.cosine_history = self.cosine_history[-self.max_history_len:]
            self.anomaly_history = self.anomaly_history[-self.max_history_len:]
        
        # Update tracking
        self.rounds_seen += 1
        self.last_round = round_num
        
        logger.debug(
            f"Client {self.client_id} profile updated: "
            f"mu={self.mu_norm:.3f}, sigma={self.sigma_norm:.3f}, "
            f"rho={self.rho_direction:.3f}, reliability={self.reliability:.3f}"
        )
    
    def get_anomaly_score(self) -> float:
        """
        Get the most recent anomaly score.
        
        Returns:
            Latest anomaly score or 0 if no history
        """
        return self.anomaly_history[-1] if self.anomaly_history else 0.0
    
    def get_temporal_drift(self, window: int = 5) -> float:
        """
        Compute temporal drift in norm over recent rounds.
        
        Args:
            window: Number of recent rounds to consider
            
        Returns:
            Drift metric
        """
        if len(self.norm_history) < 2:
            return 0.0
        
        recent = self.norm_history[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Compute coefficient of variation
        mean_norm = np.mean(recent)
        std_norm = np.std(recent)
        
        if mean_norm > 0:
            return std_norm / mean_norm
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            'client_id': self.client_id,
            'mu_norm': float(self.mu_norm),
            'sigma_norm': float(self.sigma_norm),
            'rho_direction': float(self.rho_direction),
            'reliability': float(self.reliability),
            'rounds_seen': self.rounds_seen,
            'last_round': self.last_round,
            'norm_history': self.norm_history[-10:],  # Last 10 for saving
            'cosine_history': self.cosine_history[-10:],
            'anomaly_history': self.anomaly_history[-10:],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClientProfile':
        """Create profile from dictionary."""
        profile = cls(client_id=data['client_id'])
        profile.mu_norm = data.get('mu_norm', 0.0)
        profile.sigma_norm = data.get('sigma_norm', 1.0)
        profile.rho_direction = data.get('rho_direction', 0.0)
        profile.reliability = data.get('reliability', 0.5)
        profile.rounds_seen = data.get('rounds_seen', 0)
        profile.last_round = data.get('last_round', -1)
        profile.norm_history = data.get('norm_history', [])
        profile.cosine_history = data.get('cosine_history', [])
        profile.anomaly_history = data.get('anomaly_history', [])
        return profile


class ClientProfileManager:
    """
    Manages behavioral profiles for all clients.
    """
    
    def __init__(
        self,
        num_clients: int,
        bootstrap_rounds: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize profile manager.
        
        Args:
            num_clients: Total number of clients
            bootstrap_rounds: Number of bootstrap rounds
            config: Additional configuration
        """
        self.num_clients = num_clients
        self.bootstrap_rounds = bootstrap_rounds
        self.config = config or {}
        
        # Initialize profiles for all clients
        self.profiles = {
            i: ClientProfile(client_id=i)
            for i in range(num_clients)
        }
        
        self.current_round = 0
        
        logger.info(f"Initialized profile manager for {num_clients} clients")
    
    def update_profiles(
        self,
        client_ids: List[int],
        gradients: List[Any],
        round_num: int
    ):
        """
        Update profiles for participating clients.
        
        Args:
            client_ids: List of participating client IDs
            gradients: List of client gradients
            round_num: Current round number
        """
        self.current_round = round_num
        
        # TODO: Compute norms and cosines from gradients
        # This is a placeholder - actual implementation needs gradient processing
        
        for client_id in client_ids:
            if client_id in self.profiles:
                # Placeholder values - should compute from actual gradients
                norm = np.random.uniform(0.5, 1.5)
                cosine = np.random.uniform(0.8, 1.0)
                anomaly = 0.0  # Will be computed by aggregator
                
                self.profiles[client_id].update(
                    norm=norm,
                    cosine=cosine,
                    anomaly=anomaly,
                    round_num=round_num,
                    beta=self.config.get('beta', 0.9)
                )
    
    def get_profile(self, client_id: int) -> Optional[ClientProfile]:
        """
        Get profile for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client profile or None if not found
        """
        return self.profiles.get(client_id)
    
    def get_profiles(self, client_ids: List[int]) -> List[ClientProfile]:
        """
        Get profiles for multiple clients.
        
        Args:
            client_ids: List of client identifiers
            
        Returns:
            List of client profiles
        """
        return [self.profiles[cid] for cid in client_ids if cid in self.profiles]
    
    def get_all_profiles(self) -> Dict[int, ClientProfile]:
        """
        Get all client profiles.
        
        Returns:
            Dictionary of all profiles
        """
        return self.profiles.copy()
    
    def is_bootstrap_phase(self) -> bool:
        """
        Check if currently in bootstrap phase.
        
        Returns:
            True if in bootstrap phase
        """
        return self.current_round <= self.bootstrap_rounds
    
    def get_anomaly_statistics(self) -> Dict[str, float]:
        """
        Get aggregate anomaly statistics across all clients.
        
        Returns:
            Dictionary of statistics
        """
        anomaly_scores = [
            p.get_anomaly_score() 
            for p in self.profiles.values() 
            if p.rounds_seen > 0
        ]
        
        if not anomaly_scores:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
            }
        
        return {
            'mean': float(np.mean(anomaly_scores)),
            'std': float(np.std(anomaly_scores)),
            'min': float(np.min(anomaly_scores)),
            'max': float(np.max(anomaly_scores)),
            'median': float(np.median(anomaly_scores)),
        }
    
    def save_profiles(self, filepath: str):
        """
        Save all profiles to file.
        
        Args:
            filepath: Path to save file
        """
        import json
        
        data = {
            'round': self.current_round,
            'profiles': {
                cid: profile.to_dict()
                for cid, profile in self.profiles.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved profiles to {filepath}")
    
    def load_profiles(self, filepath: str):
        """
        Load profiles from file.
        
        Args:
            filepath: Path to load file
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.current_round = data['round']
        self.profiles = {
            int(cid): ClientProfile.from_dict(profile_data)
            for cid, profile_data in data['profiles'].items()
        }
        
        logger.info(f"Loaded profiles from {filepath}")