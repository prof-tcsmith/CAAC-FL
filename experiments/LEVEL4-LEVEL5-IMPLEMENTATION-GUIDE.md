# Level 4 & Level 5 Implementation Guide

**Source:** Extracted from EXPERIMENT-PLAN.md
**Purpose:** Quick reference for implementing advanced attacks and CAAC-FL

---

## Level 4: Advanced Attacks and Detection

### Goal
Implement sophisticated attacks and begin behavioral tracking as a prototype for CAAC-FL.

### New Elements

#### Advanced Attacks
1. **ALIE (A Little Is Enough)** - variance-based stealthy attack
   - Stays within expected gradient variance to avoid detection
   - Circumvents distance-based defenses like Krum
   - More sophisticated than random noise

2. **Label Flipping** - data poisoning attack
   - Malicious clients train on corrupted labels
   - Harder to detect than gradient manipulation

#### Byzantine Detection Metrics
- **True Positive Rate (TPR)**: % Byzantine clients correctly identified
- **False Positive Rate (FPR)**: % Benign clients incorrectly flagged
- **F1 Score**: Harmonic mean of precision and recall
- **Detection Latency**: Rounds until Byzantine client identified

#### Behavioral Tracking (Prototype)
- Track gradient norms per client
- Track gradient direction consistency
- Simple threshold-based detection

### Configuration
- **Clients:** 20 total (16 benign, 4 Byzantine = 20%)
- **Data:** Non-IID (Dirichlet α=0.5)
- **Rounds:** 50
- **Test Multiple Attack Types:** ALIE, Label Flipping, Random Noise, Sign Flipping

### Aggregation Methods
1. FedAvg (baseline)
2. Krum
3. FedMedian
4. **Simple Behavioral Filter** (prototype CAAC-FL component)
   - Track client gradient norms
   - Flag clients with anomalous norms
   - Use threshold-based detection

### Key Metrics
1. Test Accuracy (per attack type)
2. Attack Success Rate (for each method)
3. **True Positive Rate** (% Byzantine correctly identified)
4. **False Positive Rate** (% Benign incorrectly flagged)
5. **Detection Latency** (rounds until Byzantine identified)
6. Convergence Speed

### Expected Outcomes
- ALIE circumvents Krum and Median (as literature shows)
- Simple behavioral tracking shows promise
- Need for temporal consistency becomes apparent
- TPR/FPR tradeoff visible
- Sets foundation for Level 5

### Deliverables
```
level4_advanced_attacks/
├── README.md
├── attacks.py                  # ALIE, Label Flipping implementations
├── behavioral_filter.py        # Prototype gradient norm tracking
├── run_fedavg.py
├── run_krum.py
├── run_fedmedian.py
├── run_behavioral.py           # Simple behavioral filter
├── client.py                   # Client with attack capabilities
├── analyze_results.py          # Detection metrics (TPR, FPR)
├── run_all.sh
├── results/                    # ROC curves, detection stats
└── LEVEL4-ANALYSIS.md
```

### Implementation Notes

#### ALIE Attack (Simplified)
```python
def alie_attack(honest_gradients, malicious_gradient, z_max=1.5):
    """
    ALIE: A Little Is Enough attack

    Stays within z_max standard deviations of honest gradient distribution
    to avoid detection by statistical defenses.

    Args:
        honest_gradients: List of gradients from honest clients
        malicious_gradient: Original gradient from Byzantine client
        z_max: Maximum z-score to stay within (default 1.5)
    """
    # Compute statistics of honest gradients (per coordinate)
    honest_stack = torch.stack(honest_gradients)
    mu = torch.mean(honest_stack, dim=0)
    sigma = torch.std(honest_stack, dim=0)

    # Flip the gradient direction (attack)
    flipped = -malicious_gradient

    # Clip to stay within z_max standard deviations
    deviation = flipped - mu
    z_scores = deviation / (sigma + 1e-8)

    # Scale down where z_score exceeds threshold
    mask = torch.abs(z_scores) > z_max
    clipped = torch.where(
        mask,
        mu + z_max * sigma * torch.sign(deviation),
        flipped
    )

    return clipped
```

#### Simple Behavioral Filter
```python
class SimpleBehavioralFilter:
    """Prototype behavioral tracking for Level 4"""

    def __init__(self, num_clients, threshold=2.0):
        self.norm_history = {i: [] for i in range(num_clients)}
        self.threshold = threshold  # z-score threshold

    def update(self, client_id, gradient):
        """Track gradient norm"""
        norm = torch.norm(gradient).item()
        self.norm_history[client_id].append(norm)

    def is_anomalous(self, client_id):
        """Simple z-score based detection"""
        if len(self.norm_history[client_id]) < 5:
            return False  # Need history

        norms = self.norm_history[client_id]
        mu = np.mean(norms)
        sigma = np.std(norms)

        current_norm = norms[-1]
        z_score = abs(current_norm - mu) / (sigma + 1e-8)

        return z_score > self.threshold
```

---

## Level 5: Full CAAC-FL Implementation

### Goal
Complete client-adaptive behavioral profiling system with multi-dimensional anomaly detection.

### CAAC-FL Core Components

#### 1. Client Behavioral Profiles
- **EWMA-based gradient norm tracking** (μ, σ per client)
  - Exponentially Weighted Moving Average for temporal smoothing
- **Directional consistency tracking**
  - Cosine similarity with historical gradients
- **Temporal variance monitoring**
  - Detect sudden behavioral changes

#### 2. Multi-dimensional Anomaly Detection
Three anomaly dimensions:
- **Magnitude anomaly:** `|‖g‖ - μ| / σ`
  - Is gradient norm unusual for this client?
- **Directional anomaly:** `1 - mean(cos(g_t, g_hist))`
  - Is gradient direction consistent with history?
- **Temporal anomaly:** Variance drift detection
  - Is client behavior becoming more erratic?

#### 3. Adaptive Clipping Thresholds
- Client-specific thresholds based on historical behavior
- Dynamic adjustment per round
- High reliability → more flexible thresholds
- Low reliability → more restrictive thresholds

#### 4. Trust Score System
- Exponentially weighted reliability scores
- Decay for anomalous behavior
- Used to weight client contributions

### Configuration
- **Clients:** 20 total, test with 4-8 Byzantine (20-40%)
- **Data:** Non-IID (Dirichlet α=0.5) + Power-law dataset sizes
- **Rounds:** 50
- **Attacks:** All from Level 4 + slow-drift poisoning + adaptive attacks

### Aggregation Methods
1. FedAvg (baseline)
2. Krum
3. FedMedian
4. FLTrust (if implementable - requires root dataset)
5. **CAAC-FL** (full implementation)

### Complete CAAC-FL Algorithm

```python
import torch
import torch.nn.functional as F
import numpy as np
import math

class BehavioralProfile:
    """Client-specific behavioral profile for CAAC-FL"""

    def __init__(self, client_id, window_size=10, alpha=0.1):
        self.client_id = client_id
        self.window_size = window_size
        self.alpha = alpha  # EWMA decay parameter

        # Historical tracking
        self.norm_history = []
        self.direction_history = []

        # EWMA statistics
        self.mu = None  # Mean gradient norm
        self.sigma = None  # Standard deviation

        # Reliability score (starts at 1.0 = fully trusted)
        self.reliability = 1.0

    def update(self, gradient):
        """Update profile with new gradient"""
        norm = torch.norm(gradient).item()
        self.norm_history.append(norm)

        # EWMA update for mean and std
        if self.mu is None:
            self.mu = norm
            self.sigma = 0
        else:
            # Update mean
            self.mu = self.alpha * norm + (1 - self.alpha) * self.mu

            # Update standard deviation
            self.sigma = math.sqrt(
                self.alpha * (norm - self.mu)**2 +
                (1 - self.alpha) * self.sigma**2
            )

        # Direction tracking
        self.direction_history.append(gradient.clone())

        # Maintain sliding window
        if len(self.norm_history) > self.window_size:
            self.norm_history.pop(0)
            self.direction_history.pop(0)

    def compute_anomaly_score(self, gradient):
        """
        Compute multi-dimensional anomaly score

        Returns composite score combining:
        - Magnitude anomaly (z-score)
        - Directional anomaly (cosine similarity)
        - Temporal anomaly (variance drift)
        """
        if self.mu is None:
            return 0.0  # No history yet

        norm = torch.norm(gradient).item()

        # 1. Magnitude Anomaly (z-score)
        A_mag = abs(norm - self.mu) / (self.sigma + 1e-8)

        # 2. Directional Anomaly (cosine similarity)
        if len(self.direction_history) > 0:
            cos_sims = [
                F.cosine_similarity(
                    gradient.flatten().unsqueeze(0),
                    hist.flatten().unsqueeze(0)
                ).item()
                for hist in self.direction_history
            ]
            A_dir = 1 - np.mean(cos_sims)
        else:
            A_dir = 0.0

        # 3. Temporal Consistency (variance drift)
        if len(self.norm_history) > self.window_size // 2:
            # Split history into old and new halves
            sigma_old = np.std(self.norm_history[:self.window_size//2])
            sigma_new = np.std(self.norm_history[self.window_size//2:])
            A_temp = abs(sigma_new - sigma_old) / (sigma_old + 1e-8)
        else:
            A_temp = 0.0

        # Composite score (Euclidean norm of components)
        A_composite = math.sqrt(A_mag**2 + A_dir**2 + A_temp**2)

        return A_composite

    def get_clipping_threshold(self, global_median_norm):
        """
        Compute client-specific adaptive clipping threshold

        High reliability → higher threshold (more flexible)
        Low reliability → lower threshold (more restrictive)
        """
        base_threshold = global_median_norm

        # Reliability factor: 0.5 to 1.0
        reliability_factor = 0.5 + 0.5 * self.reliability

        return base_threshold * reliability_factor


class CAACFLAggregator:
    """
    CAAC-FL: Client-Adaptive Aggregation with Context-aware Filtering

    Implements behavioral profiling and anomaly detection for Byzantine
    robustness in federated learning.
    """

    def __init__(self, num_clients, anomaly_threshold=2.0, gamma=0.1):
        """
        Args:
            num_clients: Total number of clients
            anomaly_threshold: Threshold for flagging anomalies (default 2.0)
            gamma: Reliability decay rate (default 0.1)
        """
        # Create behavioral profile for each client
        self.profiles = {
            i: BehavioralProfile(i)
            for i in range(num_clients)
        }

        self.anomaly_threshold = anomaly_threshold
        self.gamma = gamma  # Reliability update rate

        # Tracking for analysis
        self.anomaly_scores_history = []
        self.reliability_history = []

    def aggregate(self, client_updates):
        """
        Main CAAC-FL aggregation function

        Args:
            client_updates: Dict mapping client_id -> gradient tensor

        Returns:
            Aggregated gradient tensor
        """
        # Step 1: Update all client profiles
        for client_id, gradient in client_updates.items():
            profile = self.profiles[client_id]
            profile.update(gradient)

        # Step 2: Compute anomaly scores
        anomaly_scores = {}
        for client_id, gradient in client_updates.items():
            profile = self.profiles[client_id]
            anomaly_scores[client_id] = profile.compute_anomaly_score(gradient)

        # Step 3: Update reliability scores
        for client_id, anomaly_score in anomaly_scores.items():
            profile = self.profiles[client_id]

            # Is this client normal this round?
            is_normal = anomaly_score < self.anomaly_threshold

            # EWMA update of reliability
            profile.reliability = (
                self.gamma * (1.0 if is_normal else 0.0) +
                (1 - self.gamma) * profile.reliability
            )

        # Step 4: Compute global median norm for clipping reference
        all_norms = [torch.norm(g).item() for g in client_updates.values()]
        global_median_norm = np.median(all_norms)

        # Step 5: Apply client-specific adaptive clipping
        clipped_updates = {}
        for client_id, gradient in client_updates.items():
            profile = self.profiles[client_id]
            threshold = profile.get_clipping_threshold(global_median_norm)

            norm = torch.norm(gradient)
            if norm > threshold:
                # Clip to threshold
                clipped_updates[client_id] = gradient * (threshold / norm)
            else:
                clipped_updates[client_id] = gradient

        # Step 6: Weighted aggregation by reliability
        weighted_sum = torch.zeros_like(list(clipped_updates.values())[0])
        total_weight = 0.0

        for client_id, gradient in clipped_updates.items():
            weight = self.profiles[client_id].reliability
            weighted_sum += weight * gradient
            total_weight += weight

        # Step 7: Normalize and return
        aggregated = weighted_sum / (total_weight + 1e-8)

        # Track for analysis
        self.anomaly_scores_history.append(anomaly_scores.copy())
        self.reliability_history.append({
            cid: p.reliability
            for cid, p in self.profiles.items()
        })

        return aggregated

    def get_detected_byzantine(self, reliability_threshold=0.5):
        """
        Return clients flagged as Byzantine (low reliability)

        Args:
            reliability_threshold: Threshold below which client is flagged

        Returns:
            List of client IDs flagged as Byzantine
        """
        return [
            cid for cid, profile in self.profiles.items()
            if profile.reliability < reliability_threshold
        ]
```

### Advanced Attacks for Level 5

```python
class SlowDriftAttack:
    """
    Gradually intensify attack to avoid detection
    """
    def __init__(self, max_rounds=50, drift_rate=0.02):
        self.max_rounds = max_rounds
        self.drift_rate = drift_rate
        self.current_round = 0

    def apply(self, gradient):
        """Gradually flip gradient over time"""
        intensity = min(1.0, self.current_round * self.drift_rate)
        self.current_round += 1

        # Gradually transition from honest to fully flipped
        return (1 - intensity) * gradient + intensity * (-gradient)


class ProfileAwareAttack:
    """
    Attack that adapts to CAAC-FL's detection

    Stays within expected behavioral profile
    """
    def __init__(self, profile_mean, profile_std):
        self.mu = profile_mean
        self.sigma = profile_std

    def apply(self, gradient, z_max=1.5):
        """Similar to ALIE but uses client's own profile"""
        flipped = -gradient

        # Stay within z_max std devs of client's own behavior
        norm = torch.norm(flipped)
        expected_norm = np.random.normal(self.mu, self.sigma)

        if abs(norm - expected_norm) / self.sigma > z_max:
            # Scale to be less suspicious
            scale = expected_norm / norm
            return scale * flipped

        return flipped
```

### Comprehensive Metrics

```python
def compute_detection_metrics(predicted_byzantine, true_byzantine):
    """
    Compute detection performance metrics

    Args:
        predicted_byzantine: List of client IDs flagged as Byzantine
        true_byzantine: List of actual Byzantine client IDs

    Returns:
        dict with TPR, FPR, F1, precision, recall
    """
    pred_set = set(predicted_byzantine)
    true_set = set(true_byzantine)

    # True positives: Correctly identified Byzantine
    TP = len(pred_set & true_set)

    # False positives: Benign clients flagged as Byzantine
    FP = len(pred_set - true_set)

    # False negatives: Byzantine clients not detected
    FN = len(true_set - pred_set)

    # True negatives: Benign clients correctly identified
    # (total_clients - len(true_set)) - FP

    # Metrics
    TPR = TP / len(true_set) if len(true_set) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TPR
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'TPR': TPR,
        'FPR': FP / (total_clients - len(true_set)) if total_clients > len(true_set) else 0,
        'F1': F1,
        'precision': precision,
        'recall': recall,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }
```

### Success Criteria for Level 5
- ✅ CAAC-FL outperforms all baselines under attacks
- ✅ **TPR > 90%**: Detects most Byzantine clients
- ✅ **FPR < 10%**: Rarely flags honest clients
- ✅ **Detection latency < 10 rounds**: Quick identification
- ✅ **Maintains >70% accuracy** with 20-40% Byzantine clients
- ✅ **Distinguishes heterogeneity from attacks**: Key advantage over static methods

### Deliverables
```
level5_caacfl/
├── README.md
├── caacfl.py                   # Full CAAC-FL implementation (above)
├── attacks.py                  # All attack types
├── client.py                   # Client with behavioral tracking
├── run_fedavg.py
├── run_krum.py
├── run_fedmedian.py
├── run_caacfl.py               # Main CAAC-FL experiment
├── analyze_results.py          # Comprehensive analysis
├── visualize_profiles.py       # Behavioral profile visualization
├── run_all.sh                  # Run all experiments
├── results/
│   ├── comparison_table.csv
│   ├── roc_curves.png
│   ├── convergence_plots.png
│   ├── behavioral_profiles.png
│   ├── detection_timing.csv
│   └── level5_summary.csv
├── LEVEL5-ANALYSIS.md
└── paper_figures.py            # Publication-quality figures
```

---

## Key Differences: Level 4 vs Level 5

| Aspect | Level 4 | Level 5 |
|--------|---------|---------|
| **Detection** | Simple gradient norm threshold | Multi-dimensional anomaly score |
| **Adaptation** | Static threshold | Client-specific adaptive thresholds |
| **Temporal** | Basic history tracking | EWMA with variance drift detection |
| **Reliability** | Binary (flag or not) | Continuous score with decay |
| **Clipping** | Global threshold | Client-specific adaptive |
| **Attacks** | ALIE, Label Flipping | + Slow drift, Profile-aware |

---

## Implementation Order

### Level 4 (Prototype)
1. Implement ALIE attack
2. Implement simple behavioral filter (gradient norm tracking)
3. Add detection metrics (TPR, FPR)
4. Run experiments and analyze
5. Document limitations → motivate Level 5

### Level 5 (Full CAAC-FL)
1. Implement BehavioralProfile class
2. Implement CAACFLAggregator class
3. Implement advanced attacks (slow drift, profile-aware)
4. Integrate with Flower framework
5. Run comprehensive experiments (4 methods × 5 attacks)
6. Generate all visualizations
7. Write complete analysis

---

## References from EXPERIMENT-PLAN.md

- Full experimental plan: `experiments/EXPERIMENT-PLAN.md`
- Lines 154-210: Level 4 details
- Lines 212-398: Level 5 details (including full algorithm)
- Lines 490-517: Success criteria

---

## Next Steps After Level 3

1. Complete Level 3 with 50 clients
2. Analyze if Krum and Trimmed Mean recovered
3. Begin Level 4 implementation (prototype behavioral tracking)
4. Use Level 4 results to inform Level 5 hyperparameters
5. Implement full CAAC-FL for Level 5
6. Generate publication-ready results
