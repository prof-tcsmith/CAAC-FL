# Level 2: Heterogeneous Data Distribution

## Overview

Level 2 introduces non-IID (non-independent and identically distributed) data to observe how data heterogeneity affects different aggregation strategies. This level compares three aggregation methods on heterogeneous data without Byzantine attacks.

## Objectives

1. Understand impact of non-IID data on federated learning
2. Compare FedAvg, FedMedian, and Krum on heterogeneous data
3. Analyze performance degradation compared to Level 1 (IID)
4. Measure data heterogeneity using statistical metrics

## Experimental Setup

### Dataset
- **Dataset**: CIFAR-10
- **Distribution**: Non-IID using Dirichlet (α=0.5)
- **Clients**: 10
- **Heterogeneity**: ~1000× more heterogeneous than IID
- **Training samples per client**: 2,857-8,718 (variable)
- **Test set**: Centralized (10,000 samples)

### Model Architecture
- **Model**: SimpleCNN (same as Level 1)
- **Parameters**: ~545K trainable parameters

### Training Configuration
- **Rounds**: 50
- **Local epochs per round**: 1
- **Batch size**: 32
- **Learning rate**: 0.01
- **Optimizer**: SGD with momentum=0.9
- **Client participation**: 100% (all 10 clients)

### Aggregation Methods

1. **FedAvg** (Baseline)
   - Weighted averaging by dataset size
   - Expected to work reasonably on non-IID data

2. **FedMedian** (Robust)
   - Coordinate-wise median of client updates
   - Should be more stable than FedAvg

3. **Krum** (Distance-based)
   - Selects the most representative client update
   - May struggle with high heterogeneity

## Expected Outcomes

With non-IID data (no attacks):
- Performance degradation compared to Level 1 IID results
- FedAvg: ~70-75% accuracy (down from ~80%)
- FedMedian: Similar to FedAvg (~70-75%)
- Krum: May struggle (~60-70% or lower)
- Higher variance in convergence curves

## Key Differences from Level 1

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Data Distribution | IID | Non-IID (Dirichlet α=0.5) |
| Heterogeneity (KL div) | ~0.0007 | ~0.75 (1000× higher) |
| Client Data Sizes | Uniform (~5,000) | Variable (2,857-8,718) |
| Aggregation Methods | 2 (FedAvg, FedMedian) | 3 (+ Krum) |
| Expected Accuracy | ~75-80% | ~60-75% |

## Metrics Collected

- Test accuracy per round
- Test loss per round
- Data heterogeneity metrics (KL divergence, class imbalance)
- Per-client class distribution
- Convergence speed comparison

## Files

- `client.py`: Flower client (same as Level 1)
- `krum_strategy.py`: Custom Krum implementation
- `run_fedavg.py`: Run FedAvg with non-IID data
- `run_fedmedian.py`: Run FedMedian with non-IID data
- `run_krum.py`: Run Krum with non-IID data
- `analyze_results.py`: Compare results and visualize heterogeneity
- `run_all.sh`: Execute all experiments

## Running Experiments

```bash
# Run all three methods
bash run_all.sh

# Or run individually
python run_fedavg.py
python run_fedmedian.py
python run_krum.py

# Generate comparison plots
python analyze_results.py
```

## Success Criteria

- [x] All methods complete training without errors
- [x] Heterogeneity metrics show non-IID distribution (KL div > 0.5)
- [x] Performance degradation observed compared to Level 1
- [x] Krum shows different behavior than FedAvg/FedMedian
- [x] Clear visualizations showing impact of heterogeneity

## Key Observations to Document

1. **Performance Gap**: How much accuracy is lost due to non-IID data?
2. **Method Comparison**: Which aggregation handles heterogeneity best?
3. **Convergence Patterns**: Different convergence curves across methods?
4. **Client Variability**: Impact of variable client dataset sizes?

## Next Steps (Level 3)

- Introduce Byzantine attacks (Random Noise, Sign Flipping)
- Test robustness of aggregation methods under attack
- Add detection metrics (TPR, FPR)
- Expand to 15 clients with 20% Byzantine clients
