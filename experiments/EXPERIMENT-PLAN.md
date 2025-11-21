# CAAC-FL Experimental Progression Plan

## Overview

This document outlines a 5-level experimental progression from basic federated learning to the full CAAC-FL protocol. Each level builds upon the previous, adding complexity and capabilities systematically.

## Design Principles

1. **Incremental Complexity**: Each level adds one major new dimension
2. **Comparative Analysis**: Every level compares multiple methods
3. **Reproducibility**: All experiments use fixed seeds and documented parameters
4. **Clear Metrics**: Well-defined success criteria at each level
5. **Building Blocks**: Components from early levels reused in later ones

---

## Level 1: Federated Learning Fundamentals

**Goal**: Establish baseline federated learning with IID data, no attacks

### Components
- **Framework**: Flower (Federated Learning framework)
- **Dataset**: CIFAR-10 (standard benchmark, enables comparison to literature)
- **Data Distribution**: IID (independent and identically distributed)
- **Clients**: 10 clients, all benign
- **Aggregation Methods**:
  1. **FedAvg** (baseline - simple averaging)
  2. **FedMedian** or **FedTrimmedAvg** (built-in robust method from Flower)

### Experimental Setup
- **Model**: Simple CNN (2 conv layers, 2 FC layers)
- **Local epochs**: 5
- **Batch size**: 32
- **Learning rate**: 0.01
- **Communication rounds**: 50
- **Client selection**: 100% participation (all 10 clients per round)

### Metrics
1. **Test Accuracy** (primary) - overall model performance
2. **Training Loss** - convergence behavior
3. **Convergence Speed** - rounds to reach 70% accuracy
4. **Per-client Accuracy** - fairness metric

### Expected Outcomes
- FedAvg and FedMedian should perform similarly (no attacks)
- Both should converge smoothly
- ~70-75% accuracy on CIFAR-10
- Establishes baseline for comparison

### Deliverables
- `level1_fedavg.py` - FedAvg implementation
- `level1_fedmedian.py` - FedMedian implementation
- `level1_utils.py` - Shared utilities (data loading, model, metrics)
- `level1_results/` - Plots and CSV files
- `level1_analysis.md` - Results interpretation

---

## Level 2: Heterogeneous Data Distribution

**Goal**: Demonstrate impact of non-IID data on federated learning

### New Elements
- **Data Distribution**: Non-IID via Dirichlet allocation (α = 0.5)
- **Additional Method**: Krum (distance-based robust aggregation)

### Components
- Inherits from Level 1
- **Aggregation Methods**:
  1. FedAvg (baseline)
  2. FedMedian
  3. **Krum** (geometric defense)

### Experimental Setup
- Same as Level 1, but:
  - **Non-IID partitioning**: Dirichlet(α=0.5) for label distribution
  - Clients have skewed class distributions
  - Some clients may have no samples from certain classes

### Metrics
1. Test Accuracy (overall)
2. Training Loss
3. **Per-client Class Distribution** - visualization of heterogeneity
4. **Client Performance Variance** - std dev of client accuracies
5. Convergence Speed

### Expected Outcomes
- FedAvg performance degrades slightly (heterogeneity penalty)
- Krum may perform worse than FedAvg (as Li et al. showed ~10% on extreme non-IID)
- FedMedian middle ground
- Visualize class imbalance across clients

### Deliverables
- `level2_fedavg.py`
- `level2_fedmedian.py`
- `level2_krum.py`
- `level2_utils.py` (includes Dirichlet partitioning)
- `level2_results/` - Including data distribution visualizations
- `level2_analysis.md`

---

## Level 3: Byzantine Attacks - Basic

**Goal**: Introduce adversarial clients with simple attacks

### New Elements
- **Byzantine Clients**: 2 out of 10 (20%)
- **Attack Types**:
  1. **Random Noise**: Add Gaussian noise to gradients
  2. **Sign Flipping**: Negate all gradient values

### Components
- Inherits from Level 2
- **Aggregation Methods**:
  1. FedAvg (vulnerable baseline)
  2. FedMedian (coordinate-wise defense)
  3. Krum (distance-based defense)
  4. **FedTrimmedMean** (coordinate-wise robust)

### Experimental Setup
- Non-IID data (Dirichlet α=0.5)
- 10 clients: 8 benign, 2 Byzantine
- Byzantine clients:
  - Train locally (appear normal during training)
  - Corrupt gradients before sending to server
- Otherwise same as Level 2

### Metrics
1. Test Accuracy (with/without attack)
2. **Attack Impact** = Accuracy(no attack) - Accuracy(with attack)
3. Training Loss (observe divergence under attack)
4. **Convergence Stability** (variance across runs)
5. **Per-method Robustness** (relative performance degradation)

### Expected Outcomes
- FedAvg severely degraded under sign flipping
- FedMedian/TrimmedMean more robust to random noise
- Krum effectiveness depends on attack type
- Clear demonstration of vulnerability

### Deliverables
- `level3_attacks.py` - Attack implementations
- `level3_fedavg.py`
- `level3_fedmedian.py`
- `level3_krum.py`
- `level3_trimmedmean.py`
- `level3_utils.py`
- `level3_results/` - Comparison tables and plots
- `level3_analysis.md` - Attack impact analysis

---

## Level 4: Advanced Attacks and Detection

**Goal**: Implement sophisticated attacks and begin behavioral tracking

### New Elements
- **Advanced Attacks**:
  1. **ALIE** (A Little Is Enough) - variance-based stealthy attack
  2. **Label Flipping** - data poisoning
- **Byzantine Detection Metrics**:
  - True Positive Rate (TPR)
  - False Positive Rate (FPR)
- **Behavioral Tracking (prototype)**:
  - Track gradient norms per client
  - Track gradient direction consistency

### Components
- Inherits from Level 3
- **Aggregation Methods**:
  1. FedAvg (baseline)
  2. Krum
  3. FedMedian
  4. **Simple Behavioral Filter** (prototype CAAC-FL component)
     - Track client gradient norms
     - Flag clients with anomalous norms
     - Use threshold-based detection

### Experimental Setup
- Non-IID data (Dirichlet α=0.5)
- 20 clients: 16 benign, 4 Byzantine (20%)
- Byzantine fraction increased to test robustness limits
- Multiple attack strategies tested separately

### Metrics
1. Test Accuracy (per attack type)
2. Attack Success Rate (for each method)
3. **True Positive Rate** (% Byzantine clients correctly identified)
4. **False Positive Rate** (% benign clients incorrectly flagged)
5. **Detection Latency** (rounds until Byzantine identified)
6. Convergence Speed

### Expected Outcomes
- ALIE circumvents Krum and Median (as literature shows)
- Simple behavioral tracking shows promise
- Need for temporal consistency becomes apparent
- TPR/FPR tradeoff visible

### Deliverables
- `level4_attacks.py` - ALIE implementation
- `level4_behavioral_filter.py` - Prototype tracking
- `level4_fedavg.py`
- `level4_krum.py`
- `level4_fedmedian.py`
- `level4_utils.py` - Detection metrics
- `level4_results/` - ROC curves, detection statistics
- `level4_analysis.md` - Detection performance analysis

---

## Level 5: Full CAAC-FL Implementation

**Goal**: Complete client-adaptive behavioral profiling system

### New Elements
- **CAAC-FL Core Components**:
  1. **Client Behavioral Profiles**:
     - EWMA-based gradient norm tracking (μ, σ per client)
     - Directional consistency tracking
     - Temporal variance monitoring
  2. **Multi-dimensional Anomaly Detection**:
     - Magnitude anomaly: |‖g‖ - μ| / σ
     - Directional anomaly: 1 - mean(cos(g_t, g_hist))
     - Temporal anomaly: variance drift detection
  3. **Adaptive Clipping Thresholds**:
     - Client-specific thresholds based on historical behavior
     - Dynamic adjustment per round
  4. **Trust Score System**:
     - Exponentially weighted reliability scores
     - Decay for anomalous behavior
- **Advanced Attacks**:
  - Slow-drift poisoning (gradual attack intensification)
  - Profile-aware adaptive attack
  - Combined attack scenarios

### Components
- **Aggregation Methods**:
  1. FedAvg (baseline)
  2. Krum
  3. FedMedian
  4. FLTrust (if implementable - requires root dataset)
  5. **CAAC-FL** (full implementation)

### Experimental Setup
- Non-IID data (Dirichlet α=0.5)
- 20 clients: 12 benign, 4 Byzantine (20%), test up to 8 Byzantine (40%)
- Power-law dataset sizes (as mentioned in CAAC-FL paper)
- Multiple attack types tested

### CAAC-FL Algorithm

```python
class BehavioralProfile:
    def __init__(self, client_id, window_size=10, alpha=0.1):
        self.client_id = client_id
        self.window_size = window_size
        self.alpha = alpha  # EWMA decay

        # Historical tracking
        self.norm_history = []
        self.direction_history = []

        # EWMA statistics
        self.mu = None  # Mean gradient norm
        self.sigma = None  # Std dev

        # Reliability score
        self.reliability = 1.0

    def update(self, gradient):
        """Update profile with new gradient"""
        norm = torch.norm(gradient).item()
        self.norm_history.append(norm)

        # EWMA update
        if self.mu is None:
            self.mu = norm
            self.sigma = 0
        else:
            self.mu = self.alpha * norm + (1 - self.alpha) * self.mu
            self.sigma = math.sqrt(
                self.alpha * (norm - self.mu)**2 +
                (1 - self.alpha) * self.sigma**2
            )

        # Direction tracking
        self.direction_history.append(gradient.clone())

        # Trim to window size
        if len(self.norm_history) > self.window_size:
            self.norm_history.pop(0)
            self.direction_history.pop(0)

    def compute_anomaly_score(self, gradient):
        """Compute multi-dimensional anomaly score"""
        if self.mu is None:
            return 0.0

        norm = torch.norm(gradient).item()

        # Magnitude anomaly
        A_mag = abs(norm - self.mu) / (self.sigma + 1e-8)

        # Directional anomaly
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

        # Temporal consistency
        if len(self.norm_history) > self.window_size // 2:
            sigma_old = np.std(self.norm_history[:self.window_size//2])
            sigma_new = np.std(self.norm_history[self.window_size//2:])
            A_temp = abs(sigma_new - sigma_old) / (sigma_old + 1e-8)
        else:
            A_temp = 0.0

        # Composite score (equal weights initially)
        A_composite = math.sqrt(A_mag**2 + A_dir**2 + A_temp**2)

        return A_composite

    def get_clipping_threshold(self, global_median_norm):
        """Compute client-specific clipping threshold"""
        # Adaptive threshold based on reliability and anomaly history
        base_threshold = global_median_norm

        # Adjust based on reliability
        # High reliability → higher threshold (more flexible)
        # Low reliability → lower threshold (more restrictive)
        reliability_factor = 0.5 + 0.5 * self.reliability

        return base_threshold * reliability_factor

class CAACFLAggregator:
    def __init__(self, num_clients, anomaly_threshold=2.0):
        self.profiles = {i: BehavioralProfile(i) for i in range(num_clients)}
        self.anomaly_threshold = anomaly_threshold
        self.gamma = 0.1  # Reliability decay rate

    def aggregate(self, client_updates):
        """CAAC-FL aggregation with behavioral profiling"""
        # Update profiles
        for client_id, gradient in client_updates.items():
            profile = self.profiles[client_id]
            profile.update(gradient)

        # Compute anomaly scores
        anomaly_scores = {}
        for client_id, gradient in client_updates.items():
            profile = self.profiles[client_id]
            anomaly_scores[client_id] = profile.compute_anomaly_score(gradient)

        # Update reliability scores
        for client_id, anomaly_score in anomaly_scores.items():
            profile = self.profiles[client_id]
            is_normal = anomaly_score < self.anomaly_threshold
            profile.reliability = (
                self.gamma * (1.0 if is_normal else 0.0) +
                (1 - self.gamma) * profile.reliability
            )

        # Compute global median norm for clipping
        all_norms = [torch.norm(g).item() for g in client_updates.values()]
        global_median_norm = np.median(all_norms)

        # Apply client-specific clipping
        clipped_updates = {}
        for client_id, gradient in client_updates.items():
            profile = self.profiles[client_id]
            threshold = profile.get_clipping_threshold(global_median_norm)

            norm = torch.norm(gradient)
            if norm > threshold:
                clipped_updates[client_id] = gradient * (threshold / norm)
            else:
                clipped_updates[client_id] = gradient

        # Weight by reliability
        weighted_sum = torch.zeros_like(list(clipped_updates.values())[0])
        total_weight = 0.0

        for client_id, gradient in clipped_updates.items():
            weight = self.profiles[client_id].reliability
            weighted_sum += weight * gradient
            total_weight += weight

        # Return weighted average
        return weighted_sum / (total_weight + 1e-8)
```

### Metrics
1. **Performance Metrics**:
   - Test Accuracy (per attack, per method)
   - Attack Impact
   - Convergence Speed
2. **Detection Metrics**:
   - True Positive Rate (TPR)
   - False Positive Rate (FPR)
   - F1 Score
   - Detection Latency (rounds)
3. **Robustness Metrics**:
   - Accuracy Degradation vs. Byzantine Fraction
   - Performance across Dirichlet α values
4. **Behavioral Metrics**:
   - Profile stability (variance over time for benign clients)
   - Anomaly score distributions (benign vs. Byzantine)
   - Threshold adaptation over time

### Expected Outcomes
- CAAC-FL outperforms all baselines under attacks
- Lower FPR than static threshold methods (distinguishes heterogeneity)
- Detects slow-drift attacks (temporal advantage)
- Graceful degradation with increasing Byzantine fraction

### Deliverables
- `level5_caacfl.py` - Full CAAC-FL implementation
- `level5_attacks.py` - All attack types
- `level5_baselines.py` - All comparison methods
- `level5_utils.py` - Comprehensive utilities
- `level5_experiments.py` - Orchestration script
- `level5_results/` - Comprehensive results
  - Comparison tables (methods × attacks)
  - ROC curves
  - Behavioral profile visualizations
  - Convergence plots
  - Detection timing analysis
- `level5_analysis.md` - Full experimental analysis
- `level5_paper_figures.py` - Generate publication-quality figures

---

## Cross-Level Comparisons

After completing all levels, generate:

### Comprehensive Comparison Report
- **Table**: Accuracy across all levels and methods
- **Figure**: Convergence comparison (all levels)
- **Figure**: Robustness progression (levels 3-5)
- **Analysis**: When CAAC-FL advantages emerge

### Key Insights Document
- Non-IID impact (Level 1 vs 2)
- Attack vulnerability (Level 2 vs 3)
- Detection importance (Level 3 vs 4)
- Client-adaptive advantage (Level 4 vs 5)

---

## Implementation Timeline

### Phase 1: Infrastructure (Levels 1-2)
1. Set up Flower environment
2. Implement data loading and partitioning
3. Implement basic models
4. Level 1 experiments (2-3 days)
5. Level 2 experiments (2-3 days)

### Phase 2: Attacks (Levels 3-4)
1. Implement attack infrastructure
2. Level 3 experiments (3-4 days)
3. Implement detection metrics
4. Level 4 experiments (4-5 days)

### Phase 3: CAAC-FL (Level 5)
1. Implement behavioral profiling
2. Implement CAAC-FL aggregation
3. Level 5 experiments (5-7 days)
4. Comprehensive analysis

### Phase 4: Documentation
1. Generate all figures
2. Write analysis documents
3. Create final report
4. Prepare paper submission materials

**Total Estimated Time**: 3-4 weeks

---

## Success Criteria

### Level 1
- ✅ Both methods converge
- ✅ Similar performance (no attacks)
- ✅ ~70-75% accuracy on CIFAR-10

### Level 2
- ✅ Clear visualization of non-IID distribution
- ✅ Performance degradation observed
- ✅ Different methods show different sensitivities

### Level 3
- ✅ FedAvg severely impacted by attacks
- ✅ Robust methods show resilience
- ✅ Attack impact quantified

### Level 4
- ✅ ALIE circumvents static defenses
- ✅ Detection metrics computed
- ✅ Behavioral tracking shows promise

### Level 5
- ✅ CAAC-FL outperforms all baselines under attacks
- ✅ TPR > 90%, FPR < 10%
- ✅ Detection latency < 10 rounds
- ✅ Maintains >70% accuracy with 20-40% Byzantine clients
- ✅ Distinguishes heterogeneity from attacks

---

## Technical Stack

- **FL Framework**: Flower (flwr)
- **Deep Learning**: PyTorch
- **Data**: torchvision (CIFAR-10)
- **Numerical**: NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: scikit-learn
- **Experiment Tracking**: CSV files + custom logging
- **Environment**: Python 3.8+

---

## Repository Structure

```
experiments/
├── EXPERIMENT-PLAN.md (this file)
├── level1_fundamentals/
│   ├── README.md
│   ├── fedavg.py
│   ├── fedmedian.py
│   ├── utils.py
│   ├── run_experiments.sh
│   ├── results/
│   └── analysis.md
├── level2_heterogeneous/
│   ├── README.md
│   ├── fedavg.py
│   ├── fedmedian.py
│   ├── krum.py
│   ├── utils.py
│   ├── run_experiments.sh
│   ├── results/
│   └── analysis.md
├── level3_basic_attacks/
│   ├── README.md
│   ├── attacks.py
│   ├── fedavg.py
│   ├── fedmedian.py
│   ├── krum.py
│   ├── trimmedmean.py
│   ├── utils.py
│   ├── run_experiments.sh
│   ├── results/
│   └── analysis.md
├── level4_advanced_attacks/
│   ├── README.md
│   ├── attacks.py
│   ├── behavioral_filter.py
│   ├── baselines.py
│   ├── utils.py
│   ├── run_experiments.sh
│   ├── results/
│   └── analysis.md
├── level5_caacfl/
│   ├── README.md
│   ├── caacfl.py
│   ├── attacks.py
│   ├── baselines.py
│   ├── utils.py
│   ├── experiments.py
│   ├── run_experiments.sh
│   ├── results/
│   ├── analysis.md
│   └── paper_figures.py
├── shared/
│   ├── models.py (shared model architectures)
│   ├── data_utils.py (shared data loading)
│   └── metrics.py (shared metrics)
└── analysis/
    ├── cross_level_comparison.py
    ├── generate_paper_figures.py
    └── final_report.md
```

---

## Notes

- Each level is self-contained but builds on previous levels
- Shared utilities minimize code duplication
- Results are saved in standardized formats for easy comparison
- All experiments use fixed random seeds for reproducibility
- Hyperparameters documented in each level's README
- Analysis documents interpret results and guide next steps
