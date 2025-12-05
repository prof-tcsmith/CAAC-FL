"""
CAAC-FL: Client-Adaptive Anomaly-Aware Clipping for Federated Learning

This module implements the CAAC-FL algorithm as described in the WITS 2025 paper.

The algorithm maintains per-client behavioral profiles and uses three-dimensional
anomaly detection (magnitude, directional, temporal) to distinguish legitimate
heterogeneity from Byzantine attacks.

Key Components:
1. ClientProfile: EWMA-based behavioral tracking per client
2. AnomalyDetector: Three-dimensional anomaly scoring
3. CAACFLAggregator: Adaptive threshold and gradient clipping

================================================================================
COLD-START PROBLEM AND MITIGATION STRATEGIES
================================================================================

The cold-start problem occurs because profile-based detection requires historical
data that doesn't exist for new clients. Byzantine clients attacking from round 1
can establish malicious behavior as their "normal" baseline.

IMPLEMENTED MITIGATIONS AND THEIR SPECIFIC VALUES:

1. **Conservative Initial Thresholds**
   - Parameter: warmup_rounds = 10 (rounds), warmup_factor = 0.3
   - Implementation: During rounds 0-9, threshold is computed as:
       τ = τ_base × (warmup_factor + (1 - warmup_factor) × (round / warmup_rounds))
     Example at round 0: τ = 1.2 × 0.3 = 0.36 (stricter)
     Example at round 5: τ = 1.2 × (0.3 + 0.7 × 0.5) = 1.2 × 0.65 = 0.78
     Example at round 10+: τ = 1.2 (normal base threshold)
   - Rationale: Be more suspicious of all clients until profiles stabilize
   - Trade-off: Higher false positive rate in early rounds

2. **Cross-Client Comparison**
   - Parameter: use_cross_comparison = True
   - Implementation: For each client gradient g_i, compute cosine similarity
     with all other clients' gradients in the same round:
       sim_ij = cos(g_i, g_j) for all j ≠ i
     Use median similarity (robust to outliers) to compute anomaly:
       A_cross = 1 - median(sim_ij)
   - Weights during cold-start (round_count < 3 or warmup):
       composite = 0.2×|A_mag| + 0.3×A_dir + 0.5×A_cross
   - Weights after warmup:
       composite = w_1×|A_mag| + w_2×A_dir + w_3×|A_temp| + 0.2×A_cross
   - Rationale: Honest clients should have correlated gradients
   - Trade-off: Can backfire with highly heterogeneous (non-IID) data

3. **Global Gradient Reference**
   - Parameter: use_global_comparison = True (in AnomalyDetector)
   - Implementation: Store previous round's aggregated gradient g_global.
     When computing directional anomaly, include:
       cos(g_i, g_global) with double weight in the similarity average
   - Rationale: Sign-flipping attacks produce cos ≈ -1 with global direction
   - Trade-off: Requires at least one round of history (no effect round 0)

4. **Delayed Profile Trust**
   - Parameter: min_rounds_for_trust = 5 (rounds)
   - Implementation: The reliability bonus is only applied after participation:
       if round_count >= min_rounds_for_trust:
           τ = τ × (1 + β × reliability)    # β = 0.5 default
       else:
           τ = τ  # No reliability bonus, use stricter base threshold
   - Rationale: Prevents Byzantine clients from quickly earning trust by
     appearing "normal" in early rounds
   - Trade-off: Legitimate clients also remain under stricter scrutiny

5. **Population-Based Initialization**
   - Parameter: use_population_init = True
   - Implementation: After each round, update population statistics:
       pop_mu = 0.2 × round_mu + 0.8 × pop_mu  (EWMA with α=0.2)
       pop_sigma = 0.2 × round_sigma + 0.8 × pop_sigma
     For new clients (round_count == 0), initialize:
       profile.mu = pop_mu
       profile.sigma = max(pop_sigma, 0.1)
   - Rationale: New clients are immediately compared against "typical" behavior
   - Trade-off: Assumes some population homogeneity

6. **New Client Weight Reduction**
   - Parameter: new_client_weight = 0.3
   - Implementation: During aggregation, effective sample count is reduced:
       if round_count < min_rounds_for_trust:
           weight_factor = new_client_weight +
               (1 - new_client_weight) × (round_count / min_rounds_for_trust)
           effective_samples = samples × weight_factor
     Example (with min_rounds_for_trust=5):
              round_count=0 → weight_factor=0.30 (30% contribution)
              round_count=1 → weight_factor=0.44 (44% contribution)
              round_count=2 → weight_factor=0.58 (58% contribution)
              round_count=3 → weight_factor=0.72 (72% contribution)
              round_count=4 → weight_factor=0.86 (86% contribution)
              round_count=5+ → weight_factor=1.0 (full contribution)
   - Rationale: Limits damage from cold-start attacks
   - Trade-off: Slows down convergence contribution from legitimate new clients

================================================================================
COLD-START PARAMETER SUMMARY TABLE
================================================================================

| Parameter            | Default | Type  | Description                        |
|---------------------|---------|-------|-------------------------------------|
| warmup_rounds       | 10      | int   | Rounds with conservative thresholds |
| warmup_factor       | 0.3     | float | Threshold multiplier at round 0     |
| min_rounds_for_trust| 5       | int   | Rounds before reliability bonus     |
| use_cross_comparison| True    | bool  | Enable cross-client comparison      |
| use_population_init | True    | bool  | Initialize from population stats    |
| new_client_weight   | 0.3     | float | Weight multiplier for new clients   |

FUTURE DIRECTIONS (not yet implemented):

7. **Trusted Seed Clients**
   - Designate a subset of clients as trusted (verified honest)
   - Use their behavior to bootstrap detection thresholds
   - Similar to FLTrust but applied to profile initialization

8. **Multi-Round Validation Window**
   - Retroactively re-evaluate early rounds once profiles stabilize
   - Apply learned thresholds to stored historical gradients
   - Can identify clients who were malicious during cold-start

References:
- Smith, Bhattacherjee, Komara. "Distinguishing Medical Diversity from Byzantine
  Attacks: Client-Adaptive Anomaly Detection for Healthcare FL." WITS 2025.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import copy


@dataclass
class ClientProfile:
    """
    Per-client behavioral profile using EWMA tracking.

    Maintains statistics about a client's historical gradient behavior,
    including magnitude mean/variance and recent gradient directions.

    Attributes:
        client_id: Unique identifier for this client
        window_size: Number of historical gradients to store (determines deque maxlen)
        mu: EWMA mean of gradient magnitudes
        sigma: EWMA standard deviation of gradient magnitudes
        reliability: Trust score based on past behavior [0, 1]
        gradient_history: Recent gradients for directional analysis
        sigma_history: Historical variance values for temporal analysis
        round_count: Number of rounds this client has participated
    """
    client_id: int
    window_size: int = 5  # Default matches CAACFLAggregator.window_size
    mu: float = 0.0
    sigma: float = 0.1  # Initial small value to avoid division by zero
    reliability: float = 0.5  # Start neutral
    gradient_history: deque = field(default_factory=deque)
    sigma_history: deque = field(default_factory=deque)
    round_count: int = 0

    def __post_init__(self):
        """Initialize deques with proper maxlen after dataclass creation."""
        # Reinitialize deques with correct maxlen based on window_size
        # gradient_history: stores window_size gradients for directional comparison
        # sigma_history: needs window_size + 1 elements for temporal anomaly
        #   (compares current sigma at [-1] with past sigma at [-(window_size+1)])
        self.gradient_history = deque(self.gradient_history, maxlen=self.window_size)
        self.sigma_history = deque(self.sigma_history, maxlen=self.window_size + 1)

    def update_ewma(self, gradient_norm: float, alpha: float = 0.1):
        """
        Update EWMA statistics with new observation.

        Formula 1: μ_i^t = α · ||g_i^t||_2 + (1 - α) · μ_i^{t-1}
        Formula 2: (σ_i^t)² = α · (||g_i^t||_2 - μ_i^{t-1})² + (1 - α) · (σ_i^{t-1})²

        Note: Variance uses deviation from PREVIOUS mean (μ^{t-1}), not the
        newly updated mean. Using the new mean would underestimate variance
        since the new mean is already adjusted toward the observation.

        Args:
            gradient_norm: L2 norm of the current gradient
            alpha: EWMA smoothing factor (default 0.1)
        """
        if self.round_count == 0:
            # First observation: initialize
            self.mu = gradient_norm
            self.sigma = 0.1  # Small initial variance
        else:
            # Compute variance update FIRST using previous mean
            # This ensures we measure deviation from expected value, not from
            # the value after adjustment
            deviation_sq = (gradient_norm - self.mu) ** 2
            variance = alpha * deviation_sq + (1 - alpha) * (self.sigma ** 2)
            self.sigma = np.sqrt(variance)

            # THEN update mean
            self.mu = alpha * gradient_norm + (1 - alpha) * self.mu

        # Store sigma for temporal analysis
        self.sigma_history.append(self.sigma)
        self.round_count += 1

    def update_reliability(self, passed_check: bool, gamma: float = 0.1):
        """
        Update reliability score based on anomaly check result.

        Formula 8: R_i^t = γ · 1_{[A_i^t < τ_i^t]} + (1 - γ) · R_i^{t-1}

        Args:
            passed_check: Whether the client passed the anomaly threshold
            gamma: Reliability update rate (default 0.1)
        """
        indicator = 1.0 if passed_check else 0.0
        self.reliability = gamma * indicator + (1 - gamma) * self.reliability

    def store_gradient(self, gradient: np.ndarray):
        """
        Store gradient for directional consistency analysis.

        Args:
            gradient: Flattened gradient vector
        """
        self.gradient_history.append(gradient.copy())


class AnomalyDetector:
    """
    Three-dimensional anomaly detection for CAAC-FL.

    Computes anomaly scores across three dimensions:
    1. Magnitude: How unusual is the gradient size for this client?
    2. Directional: How consistent is the gradient direction with history?
    3. Temporal: Is the client's variance changing suspiciously?

    Parameters:
        epsilon: Stability constant for division (default 1e-8)
        window_size: Number of rounds for historical comparisons (default 5)
        weights: Tuple of (w_mag, w_dir, w_temp) weights (default equal)
    """

    def __init__(self,
                 epsilon: float = 1e-8,
                 window_size: int = 5,
                 weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
                 use_global_comparison: bool = True):
        self.epsilon = epsilon
        self.window_size = window_size
        self.weights = weights
        self.use_global_comparison = use_global_comparison
        self.last_aggregated_gradient = None

    def compute_magnitude_anomaly(self,
                                   gradient_norm: float,
                                   profile: ClientProfile) -> float:
        """
        Compute magnitude anomaly score (z-score).

        Formula 3: A_mag^{i,t} = (||g_i^t||_2 - μ_i^{t-1}) / (σ_i^{t-1} + ε)

        Args:
            gradient_norm: L2 norm of current gradient
            profile: Client's behavioral profile

        Returns:
            Magnitude anomaly score (can be negative or positive)
        """
        if profile.round_count < 2:
            return 0.0  # Not enough history

        z_score = (gradient_norm - profile.mu) / (profile.sigma + self.epsilon)
        return z_score

    def compute_directional_anomaly(self,
                                     gradient: np.ndarray,
                                     profile: ClientProfile) -> float:
        """
        Compute directional anomaly score.

        Formula 4 & 5:
        cos(g_i^t, g_i^k) = <g_i^t, g_i^k> / (||g_i^t|| · ||g_i^k||)
        A_dir^{i,t} = 1 - (1/W) Σ cos(g_i^t, g_i^k)

        Also compares with global aggregated gradient if available.

        Args:
            gradient: Current gradient vector (flattened)
            profile: Client's behavioral profile

        Returns:
            Directional anomaly score in [0, 2]
        """
        current_norm = np.linalg.norm(gradient)
        if current_norm < self.epsilon:
            return 1.0  # Zero gradient is suspicious

        cosine_sims = []

        # Compare with historical gradients
        if len(profile.gradient_history) >= 1:
            for hist_grad in profile.gradient_history:
                hist_norm = np.linalg.norm(hist_grad)
                if hist_norm < self.epsilon:
                    continue

                # Cosine similarity
                cos_sim = np.dot(gradient, hist_grad) / (current_norm * hist_norm)
                cosine_sims.append(cos_sim)

        # Also compare with last aggregated gradient (global direction)
        if self.use_global_comparison and self.last_aggregated_gradient is not None:
            global_norm = np.linalg.norm(self.last_aggregated_gradient)
            if global_norm > self.epsilon:
                global_cos = np.dot(gradient, self.last_aggregated_gradient) / (current_norm * global_norm)
                # Weight global comparison more heavily for detecting sign-flipping
                cosine_sims.append(global_cos)
                cosine_sims.append(global_cos)  # Double weight

        if not cosine_sims:
            return 0.0

        # Average cosine similarity, convert to anomaly score
        avg_cos = np.mean(cosine_sims[-self.window_size:])
        anomaly = 1.0 - avg_cos  # [0, 2] range

        return anomaly

    def set_global_gradient(self, aggregated_gradient: np.ndarray):
        """Store the last aggregated gradient for global comparison."""
        self.last_aggregated_gradient = aggregated_gradient.copy()

    def compute_temporal_anomaly(self, profile: ClientProfile) -> float:
        """
        Compute temporal anomaly score (variance drift).

        Formula 6: A_temp^{i,t} = (σ_i^t - σ_i^{t-W}) / (σ_i^{t-W} + ε)

        Args:
            profile: Client's behavioral profile

        Returns:
            Temporal anomaly score (can be negative or positive)
        """
        if len(profile.sigma_history) < self.window_size + 1:
            return 0.0  # Not enough history

        current_sigma = profile.sigma_history[-1]
        past_sigma = profile.sigma_history[-(self.window_size + 1)]

        drift = (current_sigma - past_sigma) / (past_sigma + self.epsilon)
        return drift

    def compute_composite_score(self,
                                 gradient: np.ndarray,
                                 gradient_norm: float,
                                 profile: ClientProfile,
                                 all_gradients: Dict[int, np.ndarray] = None,
                                 is_warmup: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite anomaly score from all three dimensions.

        Formula 7: A_i^t = w_1 · |A_mag| + w_2 · A_dir + w_3 · |A_temp|

        During warmup/cold-start, cross-client comparison is weighted more heavily
        since individual profiles haven't been established yet.

        Args:
            gradient: Current gradient vector (flattened)
            gradient_norm: L2 norm of gradient
            profile: Client's behavioral profile
            all_gradients: Optional dict of all client gradients for cross-comparison
            is_warmup: Whether we're in the warmup period (cold-start)

        Returns:
            Tuple of (composite_score, individual_scores_dict)
        """
        w_mag, w_dir, w_temp = self.weights

        # Compute individual scores
        a_mag = self.compute_magnitude_anomaly(gradient_norm, profile)
        a_dir = self.compute_directional_anomaly(gradient, profile)
        a_temp = self.compute_temporal_anomaly(profile)

        # Cross-client comparison for early detection
        # This is crucial for cold-start because it doesn't depend on history
        a_cross = 0.0
        median_sim = None
        if all_gradients and len(all_gradients) > 1:
            current_norm = np.linalg.norm(gradient)
            if current_norm > self.epsilon:
                cross_sims = []
                for other_id, other_grad in all_gradients.items():
                    if other_id == profile.client_id:
                        continue
                    other_norm = np.linalg.norm(other_grad)
                    if other_norm > self.epsilon:
                        cos_sim = np.dot(gradient, other_grad) / (current_norm * other_norm)
                        cross_sims.append(cos_sim)

                if cross_sims:
                    # Use median instead of mean to be robust to outliers
                    median_sim = np.median(cross_sims)
                    # Convert to anomaly score: negative similarity = high anomaly
                    # Range: [-1, 1] -> [2, 0] (inverted so negative = high anomaly)
                    a_cross = 1.0 - median_sim

        # Composite score with adaptive weighting
        # All weight combinations are normalized to sum to 1.0 for consistent scoring
        if is_warmup or profile.round_count < 3:
            # During cold-start: heavily weight cross-client comparison
            # because individual profiles are unreliable
            if a_cross > 0:
                # Cross-client gets 50% weight, directional 30%, magnitude 20%
                # Weights: 0.2 + 0.3 + 0.5 = 1.0 ✓
                composite = 0.2 * abs(a_mag) + 0.3 * a_dir + 0.5 * a_cross
            else:
                # If no cross-client data, use directional + magnitude
                # Weights: 0.4 + 0.6 = 1.0 ✓
                composite = 0.4 * abs(a_mag) + 0.6 * a_dir
        else:
            # After warmup: use full 3D scoring with optional cross-client
            if a_cross > 0:
                # Scale down the 3D weights to make room for cross-client (20%)
                # Original: w_mag=0.5, w_dir=0.3, w_temp=0.2 (sum=1.0)
                # Scaled: 0.8 * original + 0.2 * cross = 1.0
                scale = 0.8
                composite = (scale * w_mag * abs(a_mag) +
                            scale * w_dir * a_dir +
                            scale * w_temp * abs(a_temp) +
                            0.2 * a_cross)
            else:
                # No cross-client: use original weights (sum to 1.0)
                composite = w_mag * abs(a_mag) + w_dir * a_dir + w_temp * abs(a_temp)

        scores = {
            'magnitude': a_mag,
            'directional': a_dir,
            'temporal': a_temp,
            'cross_client': a_cross,
            'median_cross_sim': median_sim if median_sim is not None else 0.0,
            'composite': composite
        }

        return composite, scores


class CAACFLAggregator:
    """
    CAAC-FL Aggregation Strategy with Adaptive Clipping.

    Implements the full CAAC-FL pipeline:
    1. Maintain per-client profiles
    2. Score incoming gradients using 3D anomaly detection
    3. Apply adaptive thresholding based on reliability
    4. Soft-clip anomalous gradients
    5. Aggregate using weighted average (FedAvg-style)

    Parameters:
        num_clients: Total number of clients
        alpha: EWMA smoothing factor (default 0.1)
        gamma: Reliability update rate (default 0.1)
        tau_base: Base anomaly threshold (default 2.0)
        beta: Threshold flexibility factor (default 0.5)
        window_size: History window size (default 5)
        weights: Anomaly dimension weights (default equal)

    Cold-Start Mitigation Parameters:
        warmup_rounds: Number of rounds to use conservative thresholds (default 5)
        warmup_factor: Multiplier for stricter warmup threshold (default 0.5)
        min_rounds_for_trust: Minimum rounds before reliability can increase (default 3)
        use_cross_comparison: Enable cross-client gradient comparison (default True)
        use_population_init: Initialize profiles from population stats (default True)
        new_client_weight: Weight multiplier for new clients (default 0.5)
    """

    def __init__(self,
                 num_clients: int,
                 alpha: float = 0.05,  # Slower EWMA updates to resist profile poisoning
                 gamma: float = 0.1,
                 tau_base: float = 1.2,  # Lower threshold to catch more anomalies
                 beta: float = 0.5,
                 window_size: int = 5,
                 weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),  # Prioritize magnitude for random noise
                 # Cold-start mitigation parameters
                 warmup_rounds: int = 10,  # Longer conservative period
                 warmup_factor: float = 0.3,  # Stricter during warmup
                 min_rounds_for_trust: int = 5,  # Longer trust building period
                 use_cross_comparison: bool = True,
                 use_population_init: bool = True,
                 new_client_weight: float = 0.3):  # Less influence for new clients

        self.num_clients = num_clients
        self.alpha = alpha
        self.gamma = gamma
        self.tau_base = tau_base
        self.beta = beta

        # Cold-start mitigation settings
        self.warmup_rounds = warmup_rounds
        self.warmup_factor = warmup_factor  # Lower = stricter threshold during warmup
        self.min_rounds_for_trust = min_rounds_for_trust
        self.use_cross_comparison = use_cross_comparison
        self.use_population_init = use_population_init
        self.new_client_weight = new_client_weight

        # Track current round for warmup logic
        self.current_round = 0

        # Store window_size for profile creation
        self.window_size = window_size

        # Population statistics for profile initialization
        self.population_mu = None  # Will be computed from first round
        self.population_sigma = None

        # Initialize per-client profiles with consistent window_size
        self.profiles: Dict[int, ClientProfile] = {
            i: ClientProfile(client_id=i, window_size=window_size) for i in range(num_clients)
        }

        # Anomaly detector (uses same window_size for consistency)
        self.detector = AnomalyDetector(
            window_size=window_size,
            weights=weights,
            use_global_comparison=True  # Enable global gradient comparison
        )

        # Statistics tracking
        self.round_stats: List[Dict] = []

    def compute_adaptive_threshold(self, client_id: int) -> float:
        """
        Compute client-specific adaptive threshold with cold-start mitigation.

        Formula 9: τ_i^t = τ_base · (1 + β · R_i^{t-1})

        Cold-Start Mitigations Applied:
        1. During warmup period, threshold is reduced by warmup_factor
        2. New clients (< min_rounds_for_trust) use base threshold only
        3. Reliability bonus only applies after minimum trust period

        Args:
            client_id: ID of the client

        Returns:
            Adaptive threshold for this client
        """
        profile = self.profiles[client_id]

        # Base threshold calculation
        threshold = self.tau_base

        # Cold-start mitigation 1: Conservative warmup period
        # During warmup, use stricter (lower) thresholds for everyone
        if self.current_round < self.warmup_rounds:
            # warmup_factor < 1 means stricter threshold
            # Gradually relax as we approach end of warmup
            warmup_progress = self.current_round / self.warmup_rounds
            current_factor = self.warmup_factor + (1 - self.warmup_factor) * warmup_progress
            threshold *= current_factor

        # Cold-start mitigation 4: Delayed profile trust
        # Only apply reliability bonus after client has enough history
        if profile.round_count >= self.min_rounds_for_trust:
            # Apply reliability-based flexibility
            threshold *= (1 + self.beta * profile.reliability)
        # else: keep at (possibly reduced) base threshold

        return threshold

    def clip_gradient(self,
                      gradient: np.ndarray,
                      anomaly_score: float,
                      threshold: float) -> Tuple[np.ndarray, float]:
        """
        Apply soft clipping to gradient based on anomaly score.

        Formula 10:
        g̃_i^t = g_i^t                       if A_i^t ≤ τ_i^t
        g̃_i^t = g_i^t · (τ_i^t / A_i^t)    if A_i^t > τ_i^t

        Args:
            gradient: Original gradient vector
            anomaly_score: Computed anomaly score
            threshold: Adaptive threshold

        Returns:
            Tuple of (clipped_gradient, scaling_factor)
        """
        if anomaly_score <= threshold:
            return gradient, 1.0
        else:
            scaling = threshold / (anomaly_score + 1e-8)
            return gradient * scaling, scaling

    def _initialize_profile_from_population(self, profile: ClientProfile):
        """
        Initialize a new client's profile using population statistics.

        Cold-Start Mitigation 5: Population-Based Initialization
        Instead of starting with default values, initialize new clients
        with the mean/variance observed from the population.

        Args:
            profile: Client profile to initialize
        """
        if self.population_mu is not None and profile.round_count == 0:
            # Initialize with population statistics
            profile.mu = self.population_mu
            profile.sigma = self.population_sigma if self.population_sigma > 0.1 else 0.1

    def _update_population_statistics(self, gradient_norms: List[float]):
        """
        Update population-level statistics from current round's gradients.

        This provides the baseline for population-based initialization
        of new clients in future rounds.

        Args:
            gradient_norms: List of gradient norms from current round
        """
        if not gradient_norms:
            return

        round_mu = np.mean(gradient_norms)
        round_sigma = np.std(gradient_norms) if len(gradient_norms) > 1 else 0.1

        if self.population_mu is None:
            # First round: initialize population stats
            self.population_mu = round_mu
            self.population_sigma = round_sigma
        else:
            # EWMA update of population statistics
            pop_alpha = 0.2  # Moderate smoothing for population stats
            self.population_mu = pop_alpha * round_mu + (1 - pop_alpha) * self.population_mu
            self.population_sigma = pop_alpha * round_sigma + (1 - pop_alpha) * self.population_sigma

    def process_client_update(self,
                               client_id: int,
                               gradient: np.ndarray,
                               num_samples: int,
                               all_gradients: Dict[int, np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Process a single client's gradient update through CAAC-FL pipeline.

        Args:
            client_id: ID of the client
            gradient: Flattened gradient vector
            num_samples: Number of training samples used
            all_gradients: Optional dict of all client gradients for cross-comparison

        Returns:
            Tuple of (processed_gradient, stats_dict)
        """
        # Handle unknown client_id by creating a new profile
        if client_id not in self.profiles:
            self.profiles[client_id] = ClientProfile(
                client_id=client_id,
                window_size=self.window_size
            )

        profile = self.profiles[client_id]

        # Cold-Start Mitigation 5: Initialize new clients from population stats
        if self.use_population_init:
            self._initialize_profile_from_population(profile)

        # Step 1: Compute gradient norm
        gradient_norm = np.linalg.norm(gradient)

        # Step 2: Compute anomaly scores
        # Pass warmup flag for adaptive weighting during cold-start
        is_warmup = self.current_round < self.warmup_rounds
        composite_score, individual_scores = self.detector.compute_composite_score(
            gradient, gradient_norm, profile, all_gradients, is_warmup
        )

        # Step 3: Compute adaptive threshold
        threshold = self.compute_adaptive_threshold(client_id)

        # Step 4: Determine if anomalous and clip if needed
        is_anomalous = composite_score > threshold
        clipped_gradient, scaling = self.clip_gradient(
            gradient, composite_score, threshold
        )

        # Save round_count BEFORE updating (needed for correct weight calculation)
        round_count_before_update = profile.round_count

        # Step 5: Update profile
        profile.update_ewma(gradient_norm, self.alpha)
        profile.store_gradient(gradient)
        profile.update_reliability(not is_anomalous, self.gamma)

        # Collect statistics
        # Note: round_count_at_start is the count BEFORE this round's update,
        # used for proper new client weight calculation in aggregate()
        stats = {
            'client_id': client_id,
            'num_samples': num_samples,
            'gradient_norm': gradient_norm,
            'anomaly_score': composite_score,
            'threshold': threshold,
            'is_anomalous': is_anomalous,
            'scaling_factor': scaling,
            'reliability': profile.reliability,
            'round_count_at_start': round_count_before_update,
            **{f'score_{k}': v for k, v in individual_scores.items()}
        }

        return clipped_gradient, stats

    def aggregate(self,
                  client_gradients: Dict[int, np.ndarray],
                  client_samples: Dict[int, int]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Aggregate client gradients using CAAC-FL + FedAvg weighting.

        Formula 11: w^{t+1} = w^t - η · Σ (n_i / Σn_j) · g̃_i^t

        Cold-Start Mitigations Applied:
        - New client weight reduction (via new_client_weight parameter)
        - Population statistics update for future profile initialization
        - Round counter increment for warmup logic

        Args:
            client_gradients: Dict mapping client_id to gradient
            client_samples: Dict mapping client_id to sample count

        Returns:
            Tuple of (aggregated_gradient, list_of_client_stats)
            Returns (None, []) if no clients provided
        """
        # Handle empty client list
        if not client_gradients:
            self.current_round += 1
            return None, []

        all_stats = []
        processed_gradients = {}
        processed_samples = {}
        gradient_norms = []  # For population statistics

        # Process each client's update
        for client_id, gradient in client_gradients.items():
            num_samples = client_samples.get(client_id, 1)

            clipped_grad, stats = self.process_client_update(
                client_id, gradient, num_samples, client_gradients
            )

            processed_gradients[client_id] = clipped_grad
            processed_samples[client_id] = num_samples
            all_stats.append(stats)
            gradient_norms.append(stats['gradient_norm'])

        # Update population statistics for future profile initialization
        self._update_population_statistics(gradient_norms)

        # Weighted aggregation (FedAvg style) with new client weight reduction
        # Cold-Start Mitigation 8 (partial): Gradual Weight Ramp-Up
        # New clients contribute with reduced weight initially
        #
        # Build lookup of round_count_at_start from stats (before profile update)
        round_counts = {s['client_id']: s['round_count_at_start'] for s in all_stats}

        effective_samples = {}
        for client_id, samples in processed_samples.items():
            # Use round_count from BEFORE the update (stored in stats)
            round_count = round_counts.get(client_id, 0)
            if round_count < self.min_rounds_for_trust:
                # New clients get reduced weight
                # Protect against division by zero
                if self.min_rounds_for_trust > 0:
                    weight_factor = self.new_client_weight + \
                        (1 - self.new_client_weight) * (round_count / self.min_rounds_for_trust)
                else:
                    weight_factor = 1.0
                effective_samples[client_id] = samples * weight_factor
            else:
                effective_samples[client_id] = samples

        total_samples = sum(effective_samples.values())

        aggregated = None
        for client_id, grad in processed_gradients.items():
            weight = effective_samples[client_id] / total_samples
            if aggregated is None:
                aggregated = weight * grad
            else:
                aggregated += weight * grad

        # Store round statistics
        round_summary = {
            'round': self.current_round,
            'num_clients': len(client_gradients),
            'total_samples': sum(processed_samples.values()),  # Original samples
            'effective_samples': total_samples,  # After weight reduction
            'num_anomalous': sum(1 for s in all_stats if s['is_anomalous']),
            'mean_anomaly_score': np.mean([s['anomaly_score'] for s in all_stats]),
            'mean_reliability': np.mean([s['reliability'] for s in all_stats]),
            'population_mu': self.population_mu,
            'population_sigma': self.population_sigma,
        }
        self.round_stats.append(round_summary)

        # Store aggregated gradient for next round's comparison
        if aggregated is not None:
            self.detector.set_global_gradient(aggregated)

        # Increment round counter for warmup logic
        self.current_round += 1

        return aggregated, all_stats

    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all rounds."""
        if not self.round_stats:
            return {}

        return {
            'total_rounds': len(self.round_stats),
            'mean_anomalous_per_round': np.mean([r['num_anomalous'] for r in self.round_stats]),
            'mean_anomaly_score': np.mean([r['mean_anomaly_score'] for r in self.round_stats]),
            'final_mean_reliability': self.round_stats[-1]['mean_reliability'] if self.round_stats else 0,
        }


def flatten_model_params(model: torch.nn.Module) -> np.ndarray:
    """Flatten all model parameters into a single numpy array."""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().flatten())
    return np.concatenate(params)


def unflatten_model_params(flat_params: np.ndarray,
                           model: torch.nn.Module) -> torch.nn.Module:
    """Unflatten parameters and load into model."""
    idx = 0
    for param in model.parameters():
        num_elements = param.numel()
        param_data = flat_params[idx:idx + num_elements].reshape(param.shape)
        param.data = torch.tensor(param_data, dtype=param.dtype, device=param.device)
        idx += num_elements
    return model


def compute_gradient(model_before: torch.nn.Module,
                     model_after: torch.nn.Module) -> np.ndarray:
    """Compute gradient as difference between model states."""
    params_before = flatten_model_params(model_before)
    params_after = flatten_model_params(model_after)
    return params_after - params_before


if __name__ == "__main__":
    print("CAAC-FL Implementation Test")
    print("=" * 60)

    # Create aggregator
    num_clients = 10
    aggregator = CAACFLAggregator(num_clients=num_clients)

    # Simulate some rounds
    np.random.seed(42)

    for round_num in range(5):
        print(f"\n--- Round {round_num + 1} ---")

        # Generate fake gradients (1000 parameters)
        client_gradients = {}
        client_samples = {}

        for client_id in range(num_clients):
            # Normal clients get consistent gradients
            if client_id < 8:
                base_grad = np.random.randn(1000) * 0.1
                noise = np.random.randn(1000) * 0.01
                grad = base_grad + noise + client_id * 0.01  # Slight heterogeneity
            else:
                # Byzantine clients (2 out of 10)
                grad = np.random.randn(1000) * 10.0  # Much larger magnitude

            client_gradients[client_id] = grad
            client_samples[client_id] = np.random.randint(100, 500)

        # Aggregate
        agg_grad, stats = aggregator.aggregate(client_gradients, client_samples)

        # Print stats
        anomalous = [s for s in stats if s['is_anomalous']]
        print(f"  Aggregated gradient norm: {np.linalg.norm(agg_grad):.4f}")
        print(f"  Anomalous clients: {[s['client_id'] for s in anomalous]}")
        print(f"  Mean reliability: {np.mean([s['reliability'] for s in stats]):.4f}")

    # Final summary
    print("\n" + "=" * 60)
    print("Final Summary:")
    summary = aggregator.get_summary_stats()
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("CAAC-FL implementation test completed successfully!")
