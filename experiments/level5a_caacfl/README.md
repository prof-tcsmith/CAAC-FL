# Level 5a: CAAC-FL Implementation and Demonstration

This experiment implements and demonstrates the **CAAC-FL (Client-Adaptive Anomaly-Aware Clipping for Federated Learning)** algorithm as described in the WITS 2025 paper.

## Overview

CAAC-FL is a Byzantine-robust aggregation strategy that:
1. Maintains per-client behavioral profiles using EWMA tracking
2. Uses three-dimensional anomaly detection (magnitude, directional, temporal)
3. Applies adaptive thresholds based on client reliability
4. Soft-clips suspicious gradients rather than binary rejection

## Files

- `caacfl.py` - Core CAAC-FL implementation
  - `ClientProfile`: Per-client EWMA behavioral tracking
  - `AnomalyDetector`: Three-dimensional anomaly scoring
  - `CAACFLAggregator`: Full aggregation pipeline with adaptive clipping

- `run_caacfl_experiment.py` - Demonstration experiment on Fashion-MNIST

## Key Formulas Implemented

### Client Profile (EWMA)
```
μ_i^t = α · ||g_i^t||_2 + (1 - α) · μ_i^{t-1}
(σ_i^t)² = α · (||g_i^t||_2 - μ_i^t)² + (1 - α) · (σ_i^{t-1})²
```

### Three-Dimensional Anomaly Detection

1. **Magnitude**: `A_mag = (||g|| - μ) / (σ + ε)` (z-score)
2. **Directional**: `A_dir = 1 - (1/W) Σ cos(g^t, g^k)` (consistency with history)
3. **Temporal**: `A_temp = (σ^t - σ^{t-W}) / (σ^{t-W} + ε)` (variance drift)

### Composite Score
```
A_i^t = w_1 · |A_mag| + w_2 · A_dir + w_3 · |A_temp|
```

### Reliability and Adaptive Threshold
```
R_i^t = γ · 1_{[A < τ]} + (1 - γ) · R_i^{t-1}
τ_i^t = τ_base · (1 + β · R_i^{t-1})
```

### Soft Clipping
```
g̃ = g · (τ / A)  if A > τ
g̃ = g           otherwise
```

## Usage

### Basic Run (No Attacks)
```bash
cd experiments/level5a_caacfl
python run_caacfl_experiment.py
```

### With Sign Flipping Attack
```bash
python run_caacfl_experiment.py --attack sign_flipping --byzantine_ratio 0.2
```

### With ALIE Attack (Stealthy)
```bash
python run_caacfl_experiment.py --attack alie --byzantine_ratio 0.2
```

### Full Options
```bash
python run_caacfl_experiment.py \
    --num_clients 10 \
    --num_rounds 20 \
    --local_epochs 2 \
    --alpha 0.5 \
    --attack sign_flipping \
    --byzantine_ratio 0.2 \
    --seed 42
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_clients` | 10 | Number of FL clients |
| `--num_rounds` | 20 | Number of FL rounds |
| `--local_epochs` | 2 | Local training epochs per round |
| `--alpha` | 0.5 | Dirichlet concentration for non-IID data |
| `--attack` | none | Attack type: none, sign_flipping, random_noise, alie |
| `--byzantine_ratio` | 0.2 | Fraction of Byzantine clients |
| `--seed` | 42 | Random seed |

## CAAC-FL Parameters (in code)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `α` (EWMA) | 0.1 | EWMA smoothing factor |
| `γ` | 0.1 | Reliability update rate |
| `τ_base` | 2.0 | Base anomaly threshold |
| `β` | 0.5 | Threshold flexibility factor |
| `W` | 5 | History window size |
| `w_1, w_2, w_3` | 1/3 each | Anomaly dimension weights |

## Expected Output

```
======================================================================
Level 5a: CAAC-FL Demonstration on Fashion-MNIST
======================================================================

Configuration:
  Clients: 10 (Byzantine: 2)
  Rounds: 20
  ...

1. Loading Fashion-MNIST dataset...
2. Partitioning data (Dirichlet α=0.5)...
3. Byzantine clients: [0, 1]
4. Initializing model and CAAC-FL aggregator...
5. Starting Federated Training...
----------------------------------------------------------------------
Round   1/20: Acc=75.23% | Anomalous= 2 (True:2 FP:0) | MeanRel=0.450 | Time=2.1s
Round   2/20: Acc=78.45% | Anomalous= 2 (True:2 FP:0) | MeanRel=0.433 | Time=2.0s
...
----------------------------------------------------------------------

Final Test Accuracy: 85.67%

======================================================================
CAAC-FL Analysis Summary
======================================================================
  Total Rounds: 20
  Mean Anomalous/Round: 2.00
  Mean Anomaly Score: 1.234
  Final Mean Reliability: 0.567

Detection Performance:
  Byzantine Detection Rate: 95.0% (38/40)
  Total False Positives: 3
```

## What This Demonstrates

1. **CAAC-FL Works**: The implementation correctly maintains client profiles, computes anomaly scores, and applies adaptive clipping.

2. **Byzantine Detection**: Byzantine clients are identified through elevated anomaly scores across multiple dimensions.

3. **Heterogeneity Tolerance**: Legitimate non-IID clients maintain high reliability scores and are not falsely flagged.

4. **Graceful Degradation**: Soft clipping allows partial contribution from borderline cases rather than binary rejection.

## Cold-Start Problem and Mitigations

The **cold-start problem** is a fundamental challenge in profile-based detection: Byzantine
clients attacking from round 1 can establish malicious behavior as their "normal" baseline
before profiles are established.

### Implemented Mitigations with Specific Values

#### 1. Conservative Initial Thresholds
**Parameters:** `warmup_rounds=5`, `warmup_factor=0.5`

During rounds 0-4 (warmup period), thresholds are stricter and gradually relax:
```
τ = τ_base × (warmup_factor + (1 - warmup_factor) × (round / warmup_rounds))

Round 0: τ = 2.0 × 0.5 = 1.0  (50% of base, very strict)
Round 1: τ = 2.0 × 0.6 = 1.2
Round 2: τ = 2.0 × 0.7 = 1.4
Round 3: τ = 2.0 × 0.8 = 1.6
Round 4: τ = 2.0 × 0.9 = 1.8
Round 5+: τ = 2.0 (full base threshold)
```

#### 2. Cross-Client Comparison
**Parameter:** `use_cross_comparison=True`

For each gradient, compute cosine similarity with all other clients:
```
sim_ij = cos(g_i, g_j) for all j ≠ i
A_cross = 1 - median(sim_ij)
```

During cold-start (round_count < 3 or warmup), anomaly weighting shifts:
```
Normal:     composite = 0.33×|A_mag| + 0.33×A_dir + 0.33×|A_temp| + 0.2×A_cross
Cold-start: composite = 0.20×|A_mag| + 0.30×A_dir + 0.50×A_cross
```
The 50% weight on cross-client comparison during cold-start allows detection
of outliers even without individual history.

#### 3. Global Gradient Reference
**Parameter:** `use_global_comparison=True` (in AnomalyDetector)

Previous round's aggregated gradient is stored and used in directional anomaly:
```
cos(g_i, g_global) added with double weight to similarity average
```
Sign-flipping attacks produce cos ≈ -1 with global direction.

#### 4. Delayed Profile Trust
**Parameter:** `min_rounds_for_trust=3`

Reliability bonus only applies after participation threshold:
```python
if round_count >= 3:
    τ = τ × (1 + 0.5 × reliability)  # Can increase threshold up to 50%
else:
    τ = τ  # No bonus, stays at strict base threshold
```
Prevents Byzantine clients from quickly earning trust.

#### 5. Population-Based Initialization
**Parameter:** `use_population_init=True`

After each round, population statistics are updated (EWMA α=0.2):
```
pop_mu = 0.2 × round_mu + 0.8 × pop_mu
pop_sigma = 0.2 × round_sigma + 0.8 × pop_sigma
```
New clients (round_count=0) are initialized with these values:
```
profile.mu = pop_mu
profile.sigma = max(pop_sigma, 0.1)
```

#### 6. New Client Weight Reduction
**Parameter:** `new_client_weight=0.5`

New clients contribute with reduced aggregation weight:
```
weight_factor = 0.5 + 0.5 × (round_count / 3)

Round 0: weight_factor = 0.50 (50% contribution)
Round 1: weight_factor = 0.67 (67% contribution)
Round 2: weight_factor = 0.83 (83% contribution)
Round 3+: weight_factor = 1.00 (full contribution)
```

### Cold-Start Parameters Summary

| Parameter | Default | Type | Effect |
|-----------|---------|------|--------|
| `warmup_rounds` | 5 | int | Rounds 0-4 use stricter thresholds |
| `warmup_factor` | 0.5 | float | Round 0 threshold = 50% of base |
| `min_rounds_for_trust` | 3 | int | No reliability bonus until round 3 |
| `use_cross_comparison` | True | bool | 50% weight on cross-client during warmup |
| `use_population_init` | True | bool | Initialize μ,σ from population stats |
| `new_client_weight` | 0.5 | float | New clients start at 50% contribution |

## Limitations of This Demo

- Small scale (10 clients, 20 rounds) for quick execution
- Simplified attack implementations
- Single dataset (Fashion-MNIST)
- Cold-start detection remains challenging for subtle attacks
- No comparison with baseline methods (see Level 1-3 for baselines)

For comprehensive benchmarking, see the Level 3 attack experiments and the baseline aggregation study.

## References

- Smith, Bhattacherjee, Komara. "Distinguishing Medical Diversity from Byzantine Attacks: Client-Adaptive Anomaly Detection for Healthcare Federated Learning." WITS 2025.
- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
- Baruch et al. "A Little Is Enough: Circumventing Defenses for Distributed Learning." NeurIPS 2019.
