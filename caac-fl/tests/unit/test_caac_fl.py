"""
Unit tests for CAAC-FL aggregator
Tests core functionality of the CAAC-FL algorithm
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.aggregators.caac_fl import CAACFLAggregator, CAACFLConfig
from src.profiles.client_profile import ClientProfile, ClientProfileManager


class TestCAACFLAggregator:
    """Test suite for CAAC-FL aggregator."""
    
    @pytest.fixture
    def config(self):
        """Create default CAAC-FL configuration."""
        return CAACFLConfig(
            beta=0.9,
            gamma=0.1,
            lambda_mag=0.4,
            lambda_dir=0.4,
            lambda_temp=0.2,
            tau_anomaly=2.0,
            bootstrap_rounds=10
        )
    
    @pytest.fixture
    def aggregator(self, config):
        """Create CAAC-FL aggregator instance."""
        return CAACFLAggregator(config=config)
    
    @pytest.fixture
    def sample_updates(self):
        """Create sample gradient updates."""
        # 5 clients with gradient vectors of dimension 100
        torch.manual_seed(42)
        benign_updates = [
            torch.randn(100) * 0.1 for _ in range(3)  # Benign clients
        ]
        byzantine_updates = [
            torch.randn(100) * 10.0,  # Large magnitude attack
            -torch.randn(100) * 5.0,  # Sign-flip attack
        ]
        return benign_updates + byzantine_updates
    
    @pytest.fixture
    def client_profiles(self):
        """Create sample client profiles."""
        profiles = []
        for i in range(5):
            profile = ClientProfile(client_id=i)
            # Initialize with some history
            profile.mu_norm = 0.1 if i < 3 else 1.0
            profile.sigma_norm = 0.01
            profile.rho_direction = 0.9
            profile.reliability = 0.8 if i < 3 else 0.3
            profile.rounds_seen = 10
            profiles.append(profile)
        return profiles
    
    def test_initialization(self, aggregator, config):
        """Test aggregator initialization."""
        assert aggregator.config == config
        assert aggregator.requires_profiles == True
        assert aggregator.prev_global_gradient is None
    
    def test_bootstrap_aggregation(self, aggregator, sample_updates):
        """Test bootstrap phase aggregation."""
        # During bootstrap, should use simple clipped averaging
        result = aggregator.aggregate(
            updates=sample_updates,
            round_num=5,  # Within bootstrap phase
            is_bootstrap=True
        )
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_updates[0].shape
        
        # Result should be more influenced by benign clients (smaller norms)
        benign_mean = torch.stack(sample_updates[:3]).mean(dim=0)
        distance_to_benign = torch.norm(result - benign_mean)
        
        full_mean = torch.stack(sample_updates).mean(dim=0)
        distance_to_full = torch.norm(result - full_mean)
        
        # Bootstrap clipping should make result closer to benign mean
        assert distance_to_benign < distance_to_full
    
    def test_full_aggregation(self, aggregator, sample_updates, client_profiles):
        """Test full CAAC-FL aggregation with profiles."""
        result = aggregator.aggregate(
            updates=sample_updates,
            round_num=15,  # After bootstrap
            profiles=client_profiles,
            is_bootstrap=False
        )
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_updates[0].shape
        
        # Check that previous global gradient is stored
        assert aggregator.prev_global_gradient is not None
        assert torch.equal(aggregator.prev_global_gradient, result)
    
    def test_anomaly_score_computation(self, aggregator):
        """Test anomaly score computation."""
        # Create controlled scenario
        norms = torch.tensor([0.1, 0.1, 0.1, 5.0, 10.0])
        cosines = torch.tensor([0.9, 0.9, 0.9, 0.2, -0.5])
        
        profiles = []
        for i in range(5):
            profile = ClientProfile(client_id=i)
            profile.mu_norm = 0.1
            profile.sigma_norm = 0.01
            profile.rho_direction = 0.9
            profile.rounds_seen = 10
            profiles.append(profile)
        
        anomaly_scores = aggregator._compute_anomaly_scores(
            norms=norms,
            cosines=cosines,
            profiles=profiles
        )
        
        assert len(anomaly_scores) == 5
        
        # Benign clients (0-2) should have low anomaly scores
        assert all(anomaly_scores[i] < 1.0 for i in range(3))
        
        # Byzantine clients (3-4) should have high anomaly scores
        assert all(anomaly_scores[i] > 2.0 for i in range(3, 5))
    
    def test_reliability_update(self, aggregator, client_profiles):
        """Test reliability score updates."""
        # Low anomaly scores -> reliability should increase
        low_anomaly = torch.tensor([0.5, 0.5, 0.5, 3.0, 4.0])
        
        initial_reliabilities = [p.reliability for p in client_profiles]
        
        new_reliabilities = aggregator._update_reliability_scores(
            anomaly_scores=low_anomaly,
            profiles=client_profiles
        )
        
        # Clients with low anomaly should have increased reliability
        for i in range(3):
            assert client_profiles[i].reliability > initial_reliabilities[i]
        
        # Clients with high anomaly should have decreased reliability
        for i in range(3, 5):
            assert client_profiles[i].reliability < initial_reliabilities[i]
        
        # All reliabilities should be in [0, 1]
        assert all(0 <= r <= 1 for r in new_reliabilities)
    
    def test_adaptive_thresholds(self, aggregator):
        """Test adaptive threshold computation."""
        norms = torch.tensor([0.1, 0.1, 0.1, 5.0, 10.0])
        anomaly_scores = torch.tensor([0.5, 0.5, 0.5, 3.0, 4.0])
        reliabilities = torch.tensor([0.9, 0.9, 0.9, 0.2, 0.1])
        
        thresholds = aggregator._compute_thresholds(
            norms=norms,
            anomaly_scores=anomaly_scores,
            reliabilities=reliabilities
        )
        
        assert len(thresholds) == 5
        assert all(t > 0 for t in thresholds)
        
        # Benign clients should have higher thresholds (more lenient)
        assert all(thresholds[i] > thresholds[4] for i in range(3))
    
    def test_gradient_clipping(self, aggregator):
        """Test gradient clipping."""
        updates = torch.stack([
            torch.ones(10) * 0.1,
            torch.ones(10) * 0.1,
            torch.ones(10) * 10.0,  # Large gradient
        ])
        norms = torch.norm(updates, dim=1)
        thresholds = torch.tensor([1.0, 1.0, 0.5])
        
        clipped = aggregator._clip_gradients(
            updates=updates,
            norms=norms,
            thresholds=thresholds
        )
        
        # First two should be unchanged (within threshold)
        assert torch.allclose(clipped[0], updates[0])
        assert torch.allclose(clipped[1], updates[1])
        
        # Third should be clipped
        assert torch.norm(clipped[2]) <= thresholds[2] + 1e-6
        
        # Direction should be preserved
        assert torch.allclose(
            clipped[2] / torch.norm(clipped[2]),
            updates[2] / torch.norm(updates[2]),
            atol=1e-6
        )
    
    def test_weight_computation(self, aggregator):
        """Test aggregation weight computation."""
        anomaly_scores = torch.tensor([0.1, 0.1, 0.1, 3.0, 4.0])
        reliabilities = torch.tensor([0.9, 0.9, 0.9, 0.2, 0.1])
        
        weights = aggregator._compute_weights(
            anomaly_scores=anomaly_scores,
            reliabilities=reliabilities
        )
        
        assert len(weights) == 5
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
        assert all(w >= 0 for w in weights)
        
        # Benign clients should have higher weights
        assert all(weights[i] > weights[4] for i in range(3))
    
    def test_edge_cases(self, aggregator):
        """Test edge cases and error handling."""
        # Empty updates
        with pytest.raises(ValueError):
            aggregator.aggregate(updates=[], round_num=0)
        
        # Single client
        single_update = [torch.randn(100)]
        result = aggregator.aggregate(
            updates=single_update,
            round_num=0,
            is_bootstrap=True
        )
        assert torch.allclose(result, single_update[0])
        
        # All zeros
        zero_updates = [torch.zeros(100) for _ in range(3)]
        result = aggregator.aggregate(
            updates=zero_updates,
            round_num=0,
            is_bootstrap=True
        )
        assert torch.allclose(result, torch.zeros(100))


class TestClientProfile:
    """Test suite for client profiles."""
    
    def test_profile_initialization(self):
        """Test profile initialization."""
        profile = ClientProfile(client_id=0)
        
        assert profile.client_id == 0
        assert profile.mu_norm == 0.0
        assert profile.sigma_norm == 1.0
        assert profile.reliability == 0.5
        assert profile.rounds_seen == 0
        assert len(profile.norm_history) == 0
    
    def test_profile_update(self):
        """Test profile update."""
        profile = ClientProfile(client_id=0)
        
        # First update
        profile.update(
            norm=1.0,
            cosine=0.9,
            anomaly=0.5,
            round_num=1,
            beta=0.9
        )
        
        assert profile.rounds_seen == 1
        assert profile.last_round == 1
        assert profile.mu_norm == 1.0  # First round initialization
        assert len(profile.norm_history) == 1
        
        # Second update
        profile.update(
            norm=1.1,
            cosine=0.85,
            anomaly=0.6,
            round_num=2,
            beta=0.9
        )
        
        assert profile.rounds_seen == 2
        assert profile.mu_norm == 0.9 * 1.0 + 0.1 * 1.1  # EWMA
        assert len(profile.norm_history) == 2
    
    def test_profile_serialization(self):
        """Test profile serialization."""
        profile = ClientProfile(client_id=0)
        profile.mu_norm = 1.5
        profile.reliability = 0.7
        profile.norm_history = [1.0, 1.1, 1.2]
        
        # Convert to dict
        data = profile.to_dict()
        assert data['client_id'] == 0
        assert data['mu_norm'] == 1.5
        assert data['reliability'] == 0.7
        
        # Reconstruct from dict
        profile2 = ClientProfile.from_dict(data)
        assert profile2.client_id == 0
        assert profile2.mu_norm == 1.5
        assert profile2.reliability == 0.7
        assert profile2.norm_history == [1.0, 1.1, 1.2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])