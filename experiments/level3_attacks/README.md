# Level 3: Byzantine Attack Robustness Study

## Overview

Level 3 evaluates the robustness of federated learning aggregation strategies against Byzantine attacks. Building on the baseline study from Level 1-2, this level tests how well different aggregation methods withstand malicious client behavior.

## Research Questions

1. **RQ1**: How do aggregation strategies perform under different attack types?
2. **RQ2**: How does robustness scale with Byzantine ratio?
3. **RQ3**: Do targeted attacks circumvent defenses designed for untargeted attacks?
4. **RQ4**: What is the practical trade-off between baseline performance and attack robustness?

## Experimental Design

### Strategies (from Level 1-2 findings)

| Strategy | Type | Description |
|----------|------|-------------|
| **FedAvg** | Baseline | Weighted average, not Byzantine-robust |
| **FedMedian** | Robust | Coordinate-wise median |
| **FedTrimmedAvg** | Robust | Trimmed mean (β=0.2) |

Note: FedAdam excluded due to catastrophic failure in baseline study.

### Attacks

| Attack | Type | Description |
|--------|------|-------------|
| **None** | Baseline | Honest client behavior |
| **Random Noise** | Untargeted | Add Gaussian noise (σ=1.0) |
| **Sign Flipping** | Untargeted | Reverse gradient direction |
| **ALIE** | Targeted | A Little Is Enough - evade detection |
| **IPM** | Targeted | Inner Product Manipulation |
| **Label Flipping** | Data Poisoning | Simulate training on flipped labels |

### Byzantine Ratios

- 10% (2-3 of 25 clients)
- 20% (5 of 25 clients)
- 30% (7-8 of 25 clients)

### Data Distributions

- **IID**: Uniform random partitioning (homogeneous)
- **Non-IID**: Dirichlet α=0.5 (heterogeneous)

### Total Experiments

3 strategies × 5 attacks × 3 ratios × 2 data distributions = **90 experiments**

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Clients | 25 |
| IID | Uniform random split |
| Non-IID | Dirichlet α=0.5 |
| Rounds | 50 |
| Local Epochs | 5 |
| Learning Rate | 0.01 |

## Files

```
level3_attacks/
├── README.md                    # This file
├── attacks.py                   # Attack implementations
├── client.py                    # FL client with attack support
├── run_experiments.py           # Unified experiment runner
├── analyze_paper_results.py     # Analysis and visualization
├── run_fedavg.py               # Legacy single-strategy runner
├── run_fedmedian.py            # Legacy single-strategy runner
├── run_krum.py                 # Legacy (Krum excluded from paper)
├── run_trimmed_mean.py         # Legacy single-strategy runner
└── results/
    └── paper/                  # Paper experiment results
        ├── *_result.json       # Individual experiment results
        └── figures/            # Generated figures
```

## Usage

### Quick Start: Run Full Study

```bash
cd experiments/level3_attacks

# Run all 45 experiments (takes several hours)
python run_experiments.py --full_study --output_dir ./results/paper
```

### Run Subset of Experiments

```bash
# All attacks for one strategy (non-IID, default)
python run_experiments.py --strategy fedmedian --all_attacks

# All attacks for one strategy (IID)
python run_experiments.py --strategy fedmedian --all_attacks --iid

# Single experiment (non-IID)
python run_experiments.py --strategy fedavg --attack sign_flipping --byzantine_ratio 0.2

# Single experiment (IID)
python run_experiments.py --strategy fedavg --attack sign_flipping --byzantine_ratio 0.2 --iid

# Quick test with fewer rounds
python run_experiments.py --strategy fedavg --attack none --num_rounds 10
```

### Analyze Results

```bash
python analyze_paper_results.py --results_dir ./results/paper --output_dir ./results/paper/figures
```

## Expected Results

### Hypothesis: FedMedian Most Robust

| Strategy | No Attack | Under Attack (avg) | Expected Robustness |
|----------|-----------|-------------------|---------------------|
| FedAvg | ~62% | ~25% | Low |
| FedMedian | ~58% | ~50% | **High** |
| FedTrimmedAvg | ~60% | ~42% | Medium |

### Hypothesis: Targeted > Untargeted

ALIE and IPM should be more effective than Random Noise and Sign Flipping against FedTrimmedAvg because they craft updates that appear statistically normal.

## Paper

The paper is located at:
```
experiments/papers/byzantine-robustness/byzantine_robustness_study.qmd
```

To render:
```bash
cd experiments/papers/byzantine-robustness
quarto render byzantine_robustness_study.qmd
```

## References

- McMahan et al. (2017) - Communication-Efficient Learning of Deep Networks (FedAvg)
- Yin et al. (2018) - Byzantine-Robust Distributed Learning (FedMedian, TrimmedMean)
- Baruch et al. (2019) - A Little Is Enough (ALIE Attack)
- Xie et al. (2020) - Fall of Empires (IPM Attack)
- Fang et al. (2020) - Local Model Poisoning Attacks

---

**Status**: Ready for experiments
**Level**: 3 of 5
**Progress**: Framework complete, experiments pending
