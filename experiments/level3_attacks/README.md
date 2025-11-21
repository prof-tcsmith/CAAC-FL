# Level 3: Basic Byzantine Attacks

## Overview

Level 3 introduces Byzantine attacks to the federated learning system, testing the robustness of different aggregation methods against malicious clients.

## Objectives

1. **Implement Byzantine Attacks**
   - Random Noise Attack: Add Gaussian noise to model updates
   - Sign Flipping Attack: Reverse the sign of gradients

2. **Expand System Scale**
   - Increase from 10 to 15 clients
   - Introduce 20% Byzantine clients (3 attackers)
   - Keep non-IID data distribution (Dirichlet α=0.5)

3. **Evaluate Robustness**
   - Test 4 aggregation methods: FedAvg, FedMedian, Krum, Trimmed Mean
   - Compare performance with vs. without attacks
   - Measure attack impact on convergence

4. **Detection Metrics**
   - True Positive Rate (TPR): Correctly identified Byzantine clients
   - False Positive Rate (FPR): Honest clients misclassified as Byzantine
   - F1 Score: Harmonic mean of precision and recall

## Configuration

### System Setup
- **Total Clients**: 15
- **Byzantine Clients**: 3 (20%)
- **Honest Clients**: 12 (80%)
- **Data Distribution**: Non-IID (Dirichlet α=0.5)
- **Rounds**: 50
- **Local Epochs**: 5

### Attack Types

#### 1. Random Noise Attack
- Adds Gaussian noise to model parameters
- Noise scale: σ = 1.0 (configurable)
- Goal: Disrupt convergence through random perturbations

#### 2. Sign Flipping Attack
- Reverses the sign of all gradients
- Most destructive untargeted attack
- Goal: Push model in opposite direction of convergence

### Aggregation Methods

1. **FedAvg** (Baseline, non-robust)
   - Weighted averaging of all client updates
   - Expected to be vulnerable to attacks

2. **FedMedian** (Coordinate-wise median)
   - Robust to outliers in each parameter
   - Expected to resist noise but struggle with sign flipping

3. **Krum** (Distance-based selection)
   - Selects most representative client
   - Expected to perform better with attacks than with heterogeneity

4. **Trimmed Mean** (NEW)
   - Removes top/bottom β% of updates per parameter
   - β = 20% (matches Byzantine ratio)
   - Theoretically robust to up to β fraction of Byzantine clients

## Expected Results

### Performance Predictions

| Method | No Attack | Random Noise | Sign Flipping |
|--------|-----------|--------------|---------------|
| **FedAvg** | 70-73% | 40-50% | 10-20% |
| **FedMedian** | 68-71% | 60-65% | 30-40% |
| **Krum** | 65-70% | 60-68% | 55-65% |
| **Trimmed Mean** | 68-72% | 65-70% | 60-68% |

### Hypotheses

1. **FedAvg Vulnerability**: Should degrade significantly under both attacks
2. **Trimmed Mean Best**: Should maintain highest accuracy under attacks
3. **Krum Improvement**: Should perform better than Level 2 (attacks more detectable than natural heterogeneity)
4. **Attack Severity**: Sign Flipping > Random Noise > No Attack

## Files

```
level3_attacks/
├── README.md                      # This file
├── IMPLEMENTATION.md              # Technical details
├── attacks.py                     # Byzantine attack implementations
├── trimmed_mean_strategy.py       # Trimmed Mean aggregation
├── client.py                      # Enhanced client with attack support
├── run_fedavg.py                  # FedAvg with attacks
├── run_fedmedian.py               # FedMedian with attacks
├── run_krum.py                    # Krum with attacks
├── run_trimmed_mean.py            # Trimmed Mean with attacks
├── analyze_results.py             # Attack impact analysis
├── run_all.sh                     # Orchestration script
└── results/                       # Output directory
```

## Usage

### Quick Start

```bash
cd level3_attacks
bash run_all.sh
```

This will run all 12 experiments (4 methods × 3 attack scenarios) and generate comparative analysis.

### Individual Experiments

```bash
# FedAvg with no attack (baseline)
python run_fedavg.py --attack none

# FedAvg with random noise
python run_fedavg.py --attack random_noise

# FedAvg with sign flipping
python run_fedavg.py --attack sign_flipping

# Same pattern for other methods
python run_fedmedian.py --attack sign_flipping
python run_krum.py --attack random_noise
python run_trimmed_mean.py --attack none
```

### Analysis Only

```bash
python analyze_results.py
```

## Key Differences from Level 2

| Aspect | Level 2 | Level 3 |
|--------|---------|---------|
| **Clients** | 10 | 15 |
| **Byzantine** | 0 | 3 (20%) |
| **Attacks** | None | Random Noise, Sign Flipping |
| **Aggregation Methods** | 3 | 4 (+ Trimmed Mean) |
| **Experiments** | 3 | 12 (4 methods × 3 scenarios) |
| **Detection Metrics** | No | Yes (TPR, FPR, F1) |
| **Expected Runtime** | 30-45 min (CPU) | 90-120 min (CPU) |

## Next Steps

After Level 3 completion:
- **Level 4**: Advanced attacks (Label Flipping, Model Poisoning) + Detection strategies
- **Level 5**: Full CAAC-FL protocol with clustering and adaptive defense

## References

- **Random Noise Attack**: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning," USENIX Security 2020
- **Sign Flipping**: Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed Learning," NeurIPS 2019
- **Trimmed Mean**: Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates," ICML 2018
- **Krum**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent," NeurIPS 2017

---

**Status**: Ready for implementation
**Level**: 3 of 5
**Progress**: 40% → 60% of experimental framework
