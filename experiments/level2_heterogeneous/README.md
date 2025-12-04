# Level 2: Non-IID (Heterogeneous) Data Distribution

## Overview

Level 2 extends the Level 1 baseline study to non-IID (non-independent and identically distributed) data scenarios. Using the same experimental framework as Level 1, we investigate how label heterogeneity affects the relative performance of FedAvg, FedMean, and FedMedian aggregation strategies.

## Objectives

1. Measure the impact of label heterogeneity on federated learning convergence
2. Compare aggregation strategies under non-IID conditions
3. Determine whether FedAvg's weighting advantage persists with heterogeneous data
4. Quantify performance degradation from IID to non-IID scenarios

## Experimental Setup

### Datasets

Same as Level 1 for direct comparison:

- **MNIST**: Handwritten digits (60,000 train, 10,000 test)
- **Fashion-MNIST**: Fashion items (60,000 train, 10,000 test)
- **CIFAR-10**: Color images (50,000 train, 10,000 test)

### Data Partitioning

Non-IID partitioning using **Dirichlet distribution**:

- **Method**: Each client receives samples based on Dirichlet allocation per class
- **Alpha (α)**: Controls heterogeneity level
  - α = 0.1: Highly non-IID (clients have 1-2 dominant classes)
  - α = 0.5: Moderately non-IID (default, some class imbalance)
  - α = 1.0: Mildly non-IID (approaching IID-like)
- **Clients**: 50 (matching Level 1)

### Model Architecture

Same models as Level 1:

- **MNIST/Fashion-MNIST**: SimpleMLP (784→512→256→10)
- **CIFAR-10**: SimpleCNN (conv layers → FC layers)

### Training Configuration (Matching Level 1)

| Parameter | Value |
|-----------|-------|
| Rounds | 50 |
| Local epochs | 1 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Optimizer | SGD (momentum=0.9) |
| Client participation | 100% |

### Aggregation Strategies

1. **FedAvg**: Weighted averaging by client dataset size
2. **FedMean**: Unweighted averaging (equal weight per client)
3. **FedMedian**: Coordinate-wise median

## Experiment Matrix

For α = 0.5 (primary study):

- 3 datasets × 3 strategies × 5 seeds = **45 experiments**

For extended alpha study (optional):

- 3 datasets × 3 strategies × 3 alpha values × 5 seeds = **135 experiments**

## Key Differences from Level 1

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Label Distribution | IID | Non-IID (Dirichlet) |
| Data Quantity | Equal or Unequal | Variable (Dirichlet) |
| Heterogeneity (KL div) | ~0.001 | ~0.5-1.5 |
| Primary Research Question | Quantity imbalance | Label heterogeneity |

## Files

### New Unified Framework

- `run_noniid_experiments.py`: Unified experiment runner for all non-IID experiments
- `run_all_noniid.sh`: Shell script to run complete study
- `analyze_noniid_results.py`: Results aggregation and analysis

### Legacy Files (Original Implementation)

- `run_fedavg.py`, `run_fedmedian.py`, `run_krum.py`: Original single-dataset runners
- `krum_strategy.py`: Krum aggregation strategy
- `client.py`: Original Flower client
- `analyze_results.py`: Original analysis script

## Running Experiments

```bash
# Run all experiments (45 total, ~6-8 hours with GPU)
bash run_all_noniid.sh

# Or run specific configurations
python run_noniid_experiments.py --dataset mnist --strategy fedavg --seed 42

# Run all seeds/strategies for one dataset
python run_noniid_experiments.py --dataset mnist --all

# Run with different alpha
python run_noniid_experiments.py --dataset mnist --all --alpha 0.1

# Analyze results
python analyze_noniid_results.py
```

## Expected Outcomes

Based on federated learning literature:

1. **Performance degradation** from IID baseline (5-15% accuracy drop)
2. **FedAvg may struggle** when large clients have skewed distributions
3. **FedMedian may be more robust** to heterogeneous updates
4. **FedMean** behavior under heterogeneity is less documented

## Metrics Collected

- Test accuracy per round (full trajectory)
- Test loss per round
- Client data sizes (from Dirichlet allocation)
- Heterogeneity metrics (mean KL divergence from uniform)
- Per-client class distributions
- Comparison with Level 1 IID baselines

## Analysis Outputs

- `analysis/summary_statistics.md`: Markdown summary tables
- `analysis/detailed_statistics.csv`: Full statistics CSV
- `analysis/level1_comparison.csv`: IID vs Non-IID comparison
- Visualization plots comparing strategies and datasets
