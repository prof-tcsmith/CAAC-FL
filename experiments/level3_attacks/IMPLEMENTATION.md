# Level 3: Byzantine Attacks - Implementation Details

## Overview

Level 3 introduces Byzantine attacks to test the robustness of federated learning aggregation methods. This level builds on Level 2's non-IID data distribution while adding malicious clients that attempt to disrupt model convergence.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FL Server (Flower)                       │
│  ┌────────────┬────────────┬────────────┬────────────────┐  │
│  │  FedAvg    │ FedMedian  │    Krum    │ Trimmed Mean  │  │
│  └────────────┴────────────┴────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ▲│
                          ││ Updates / Parameters
                          │▼
┌─────────────────────────────────────────────────────────────┐
│                     15 Clients                               │
│  ┌────────────────┬────────────────────────────────────┐    │
│  │ 12 Honest (80%)│      3 Byzantine (20%)              │    │
│  │                │  ┌──────────────┬──────────────┐    │    │
│  │  Normal        │  │ Random Noise │ Sign Flipping│    │    │
│  │  Training      │  │  Attack      │   Attack     │    │    │
│  └────────────────┴──┴──────────────┴──────────────┘    │    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              Non-IID Data (Dirichlet α=0.5)
```

## Byzantine Attacks

### 1. Random Noise Attack

**Objective**: Disrupt convergence through random perturbations

**Method**:
```python
θ_malicious = θ_trained + N(0, σ²I)
```

Where:
- `θ_trained`: Model parameters after local training
- `N(0, σ²I)`: Gaussian noise with mean 0 and variance σ²
- `σ = 1.0`: Noise scale (configurable)

**Characteristics**:
- Simple but effective
- Difficult to distinguish from natural parameter variation
- Impact increases with noise scale
- All parameters affected equally

**Expected Impact**: Moderate degradation (20-40% accuracy loss)

### 2. Sign Flipping Attack

**Objective**: Push model in opposite direction of convergence

**Method**:
```python
Δθ = θ_trained - θ_global           # Compute update
θ_malicious = θ_global - Δθ          # Flip the sign
```

This is equivalent to:
```python
θ_malicious = 2·θ_global - θ_trained
```

**Characteristics**:
- Most destructive untargeted attack
- Reverses gradient direction
- Easy to detect with distance-based methods
- Highly effective against averaging methods

**Expected Impact**: Severe degradation (40-70% accuracy loss for non-robust methods)

### Attack Implementation

Attacks are modular and follow a common interface:

```python
class ByzantineAttack:
    def apply(self, model: nn.Module, original_model: nn.Module) -> nn.Module:
        """Apply attack to model"""
        raise NotImplementedError()

# Usage in client
if self.is_byzantine:
    self.model = self.attack.apply(self.model, self.original_model)
```

## Aggregation Methods

### 1. FedAvg (Baseline, Non-Robust)

**Algorithm**: Weighted averaging
```python
θ_global = Σ(n_i / N) · θ_i
```

**Byzantine Robustness**: None

**Expected Behavior**:
- Baseline (no attack): ~70-73%
- Random Noise: ~40-50% (significant degradation)
- Sign Flipping: ~10-20% (severe degradation)

**Why Vulnerable**: Equally weights all clients, allowing Byzantine clients to directly influence global model.

### 2. FedMedian (Coordinate-wise Median)

**Algorithm**: Median for each parameter coordinate
```python
θ_global[j] = median(θ_1[j], θ_2[j], ..., θ_n[j])
```

**Byzantine Robustness**: Moderate (up to 50% Byzantine clients theoretically)

**Expected Behavior**:
- Baseline: ~68-71%
- Random Noise: ~60-65% (better than FedAvg)
- Sign Flipping: ~30-40% (still vulnerable)

**Why Partially Robust**: Median is robust to outliers in each dimension, but with only 15 clients and 20% Byzantine (3 clients), attacks can still influence median.

### 3. Krum (Distance-based Selection)

**Algorithm**: Select client with minimum sum of distances to k-nearest neighbors
```python
k = n - f - 2  # f = num_byzantine
scores[i] = Σ_{j in k-nearest} ||θ_i - θ_j||²
selected = argmin(scores)
```

**Byzantine Robustness**: Strong (up to f < n/2 - 1 Byzantine clients)

**Expected Behavior**:
- Baseline: ~65-70% (better than Level 2!)
- Random Noise: ~60-68%
- Sign Flipping: ~55-65%

**Why Robust**: Distance-based selection effectively identifies and excludes Byzantine clients whose updates differ significantly from honest clients.

**Key Insight**: Krum performs better in Level 3 than Level 2 because:
- In Level 2: Natural heterogeneity confused distance-based selection
- In Level 3: Attacks are more extreme and easier to detect than natural variation

### 4. Trimmed Mean (NEW)

**Algorithm**: Remove top/bottom β% of values per coordinate, then average
```python
# For each parameter θ[j]:
sorted_values = sort(θ_1[j], θ_2[j], ..., θ_n[j])
trimmed = sorted_values[⌊n·β⌋ : ⌈n·(1-β)⌉]
θ_global[j] = mean(trimmed)
```

**Byzantine Robustness**: Strong (up to β fraction of Byzantine clients)

**Configuration**: β = 0.2 (matching 20% Byzantine ratio)

**Expected Behavior**:
- Baseline: ~68-72%
- Random Noise: ~65-70% (best under this attack)
- Sign Flipping: ~60-68% (best overall robustness)

**Why Robust**:
- Removes extreme values (both high and low) before averaging
- Works coordinate-wise like FedMedian but uses mean instead of median
- More stable than median with small client counts
- Directly matches Byzantine ratio (β = 20%)

## Configuration

### System Parameters

```python
NUM_CLIENTS = 15              # Increased from 10 (Level 2)
NUM_BYZANTINE = 3             # 20% of total clients
BYZANTINE_RATIO = 0.2

NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

ALPHA = 0.5                   # Dirichlet heterogeneity (same as Level 2)
```

### Attack Parameters

```python
# Random Noise
NOISE_SCALE = 1.0             # σ for Gaussian noise

# Sign Flipping
# (no parameters, deterministic)
```

### Byzantine Client Selection

Byzantine clients are the first `NUM_BYZANTINE` clients (IDs 0, 1, 2):

```python
byzantine_clients = list(range(NUM_BYZANTINE))
```

This is deterministic and known (white-box setting) for evaluation purposes.

## Experimental Design

### Experiment Matrix

| Method | No Attack | Random Noise | Sign Flipping |
|--------|-----------|--------------|---------------|
| FedAvg | ✓ | ✓ | ✓ |
| FedMedian | ✓ | ✓ | ✓ |
| Krum | ✓ | ✓ | ✓ |
| Trimmed Mean | ✓ | ✓ | ✓ |

**Total**: 12 experiments

### Metrics Collected

For each experiment:
- **Round-by-round**: Test accuracy, test loss
- **Metadata**: Heterogeneity (KL divergence), class imbalance, attack type, number of Byzantine clients

### Evaluation Metrics

1. **Final Accuracy**: Test accuracy at round 50
2. **Degradation**: Baseline accuracy - Attack accuracy
3. **Robustness Score**: (Worst attack accuracy) / (Baseline accuracy) × 100%
4. **Convergence Speed**: Rounds to reach target accuracies (40%, 50%, 60%)

## Analysis

### Visualization

The analysis script generates a comprehensive 3×4 grid plot:

**Row 1: Test Accuracy Curves**
- 4 panels (one per method)
- 3 lines per panel (one per attack scenario)
- Shows convergence behavior under different attacks

**Row 2: Test Loss Curves**
- 4 panels (one per method)
- 3 lines per panel
- Shows optimization landscape under attacks

**Row 3: Summary Metrics**
- Panel 1: Final accuracy by attack type (grouped bar chart)
- Panel 2: Degradation from baseline (stacked bar chart)
- Panel 3: Average performance ranking (horizontal bar chart)
- Panel 4: Robustness scores (horizontal bar chart)

### Statistical Summaries

Printed summaries include:
- Final test accuracies for all method-attack combinations
- Attack impact (degradation from baseline)
- Method rankings (by average performance and robustness)
- Key observations (most robust/vulnerable methods, attack severity)

## Expected Results

### Predicted Performance Matrix

| Method | No Attack | Random Noise | Sign Flipping | Robustness |
|--------|-----------|--------------|---------------|------------|
| **FedAvg** | 70-73% | 40-50% | 10-20% | 14-29% |
| **FedMedian** | 68-71% | 60-65% | 30-40% | 44-59% |
| **Krum** | 65-70% | 60-68% | 55-65% | 85-100% |
| **Trimmed Mean** | 68-72% | 65-70% | 60-68% | 88-100% |

### Hypotheses

1. **H1**: FedAvg will show severe degradation under both attacks
   - Rationale: No Byzantine robustness mechanism

2. **H2**: Trimmed Mean will maintain highest accuracy under attacks
   - Rationale: Trim ratio (β=20%) matches Byzantine ratio exactly

3. **H3**: Krum will perform better than in Level 2
   - Rationale: Attacks are more extreme and easier to detect than natural heterogeneity

4. **H4**: Sign Flipping will cause more damage than Random Noise
   - Rationale: Gradient reversal is more destructive than random perturbation

5. **H5**: All methods will show some degradation compared to baseline
   - Rationale: Even robust methods lose information by excluding/trimming Byzantine updates

## Key Differences from Level 2

| Aspect | Level 2 | Level 3 |
|--------|---------|---------|
| **Clients** | 10 | 15 |
| **Byzantine** | 0 | 3 (20%) |
| **Attack Types** | None | Random Noise, Sign Flipping |
| **Aggregation Methods** | 3 | 4 (+ Trimmed Mean) |
| **Total Experiments** | 3 | 12 |
| **Expected Krum Performance** | 8.6% (failure) | 60-68% (success) |
| **Main Focus** | Data heterogeneity | Attack robustness |

## Implementation Notes

### Attack Application

Attacks are applied **after** local training but **before** sending parameters to server:

```python
# In FlowerClient.fit()
1. Receive global parameters from server
2. Store copy as original_model (for sign flipping)
3. Perform local training → trained model
4. If Byzantine: Apply attack to trained model
5. Send (possibly attacked) parameters to server
```

### Strategy Integration

All aggregation strategies follow the Flower Strategy interface:

```python
class CustomStrategy(Strategy):
    def aggregate_fit(self, server_round, results, failures):
        # Extract client updates
        weights_list = [parameters_to_ndarrays(fit_res.parameters)
                       for _, fit_res in results]

        # Apply aggregation (FedAvg, FedMedian, Krum, or Trimmed Mean)
        aggregated = self._aggregate_method(weights_list)

        return ndarrays_to_parameters(aggregated), metrics
```

### Command-line Interface

All run scripts accept attack type as argument:

```bash
# No attack (baseline)
python run_fedavg.py --attack none

# Random noise attack
python run_fedavg.py --attack random_noise

# Sign flipping attack
python run_fedavg.py --attack sign_flipping
```

## Reproducibility

### Random Seeds

- Global seed: 42
- Client-specific seeds: `seed + client_id` (for attack randomness)
- Ensures reproducible Byzantine behavior while maintaining client diversity

### Deterministic Byzantine Selection

- Byzantine clients: Always IDs 0, 1, 2
- Honest clients: Always IDs 3-14
- Known configuration (white-box evaluation)

### Data Partitioning

- Same Dirichlet partitioning as Level 2 (α=0.5)
- Same random seed (42) ensures identical data distribution
- Direct comparison with Level 2 baseline possible

## Future Enhancements (Level 4)

Level 3 establishes Byzantine attack capabilities. Level 4 will build on this foundation:

1. **Advanced Attacks**: Label flipping, model poisoning, adaptive attacks
2. **Detection Mechanisms**: Byzantine client detection and exclusion
3. **Detection Metrics**: True/False Positive Rates, F1 scores
4. **Adaptive Defense**: Dynamic trim ratios, multi-Krum selection

## References

1. **Random Noise Attack**: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning," USENIX Security 2020

2. **Sign Flipping Attack**: Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed Learning," NeurIPS 2019

3. **Trimmed Mean**: Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates," ICML 2018

4. **Krum**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent," NeurIPS 2017

5. **FedMedian**: Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates," ICML 2018

---

**Implementation Status**: ✅ Complete
**Testing Status**: Ready for execution
**Documentation**: Complete
