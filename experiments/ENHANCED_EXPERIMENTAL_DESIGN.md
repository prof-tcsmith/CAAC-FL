# Enhanced Experimental Design for Federated Learning Paper

## Motivation

The current Level 1 experiments use **equal-sized IID partitions**, which makes FedAvg mathematically equivalent to FedMean. To properly test our hypotheses and provide comprehensive insights, we need to vary:

1. **Dataset size distribution** (equal vs unequal)
2. **Data heterogeneity** (IID vs non-IID)
3. **Number of clients** (scalability)

## Current Gap

**Hypothesis 1**: "FedAvg achieves highest accuracy due to optimal weighting"
- **Current result**: REJECTED (FedMean wins 77.47% vs 77.04%)
- **Problem**: With equal dataset sizes, FedAvg weights = FedMean weights
- **Solution**: Need unequal dataset sizes to test weighting advantage

## Proposed Experimental Matrix

### Dimension 1: Data Distribution (3 scenarios)

| Scenario | Description | Dataset Sizes | Purpose |
|----------|-------------|---------------|---------|
| **IID-Equal** | Current setup | All 1000 samples | Baseline |
| **IID-Unequal** | Random sizes | 500-1500 range | Test FedAvg weighting |
| **Non-IID** | Dirichlet α=0.5 | Equal sizes | Test heterogeneity robustness |

### Dimension 2: Aggregation Methods (3 methods)

- FedAvg (weighted averaging)
- FedMean (unweighted averaging)
- FedMedian (robust median)

### Dimension 3: Client Count (3 levels)

- 10 clients (low parallelism, high communication efficiency)
- 25 clients (medium)
- 50 clients (high parallelism, current)

## Experimental Design Matrix

### **Level 1: IID Experiments** (9 experiments)

| Exp | Scenario | Clients | Aggregation | Purpose |
|-----|----------|---------|-------------|---------|
| 1A | IID-Equal | 50 | FedAvg | ✅ DONE (baseline) |
| 1B | IID-Equal | 50 | FedMean | ✅ DONE |
| 1C | IID-Equal | 50 | FedMedian | ✅ DONE |
| 1D | IID-Unequal | 50 | FedAvg | 🔴 NEW - Test weighting advantage |
| 1E | IID-Unequal | 50 | FedMean | 🔴 NEW - Compare vs weighted |
| 1F | IID-Unequal | 50 | FedMedian | 🔴 NEW - Robustness baseline |
| 1G | IID-Equal | 10 | All 3 | 🔴 NEW - Low client count |
| 1H | IID-Equal | 25 | All 3 | 🔴 NEW - Medium client count |
| 1I | IID-Equal | 100 | All 3 | 🔴 NEW - High scalability test |

### **Level 2: Non-IID Heterogeneous** (9 experiments)

| Exp | Dirichlet α | Clients | Aggregation | Purpose |
|-----|-------------|---------|-------------|---------|
| 2A | 0.1 (high skew) | 50 | FedAvg | Non-IID baseline |
| 2B | 0.1 | 50 | FedMean | Compare unweighted |
| 2C | 0.1 | 50 | FedMedian | Robustness test |
| 2D | 0.5 (medium) | 50 | All 3 | Moderate heterogeneity |
| 2E | 1.0 (low skew) | 50 | All 3 | Nearly IID comparison |
| 2F | 0.5 | 10 | All 3 | Few clients + non-IID |
| 2G | 0.5 | 25 | All 3 | Medium clients |
| 2H | 0.5 + Unequal sizes | 50 | All 3 | Worst-case heterogeneity |

### **Level 3: Byzantine Attacks** (keep current + enhancements)

Already configured, but ensure:
- Run with non-IID data (more realistic)
- Test with different attacker ratios (10%, 20%, 30%)
- Include Krum and Trimmed Mean

## Expected Insights from Enhanced Design

### 1. **IID-Unequal (Experiments 1D-1F)**

**Expected Results**:
- FedAvg > FedMean (finally proves H1 correctly)
- Gap widens as size variation increases
- FedMedian remains ~1% behind

**Why This Matters**:
- Shows FedAvg's true advantage
- Demonstrates when weighting is important
- Provides actionable guidance: "Use FedAvg when dataset sizes vary"

### 2. **Client Count Scaling (Experiments 1G-1I)**

**Expected Results**:
- Fewer clients = slower convergence but less communication overhead
- More clients = faster convergence per-round but more total communication
- Optimal client count depends on communication cost vs convergence speed trade-off

**Why This Matters**:
- Practical deployment guidance
- Understand parallelism vs efficiency trade-off
- Inform resource allocation decisions

### 3. **IID vs Non-IID Comparison**

**Expected Results**:
- All methods degrade with non-IID data
- FedMedian gap may narrow (robustness advantage)
- FedAvg advantage more pronounced with non-IID

**Why This Matters**:
- Most realistic FL scenario is non-IID
- Shows which methods are robust to heterogeneity
- Validates (or invalidates) IID assumptions

## Revised Paper Contributions

With enhanced experiments, the paper can claim:

1. ✅ **Comprehensive baseline comparison** under IID conditions
2. 🆕 **First systematic study** of aggregation strategies across:
   - Equal vs unequal dataset sizes
   - IID vs non-IID data distributions
   - 10-100 client scalability
3. 🆕 **Actionable guidance** on aggregation strategy selection:
   - When to use FedAvg vs FedMean
   - When robustness (FedMedian) is worth the cost
   - How client count affects strategy choice

## Implementation Priority

### Phase 1: Critical for Paper (HIGH PRIORITY)
1. **IID-Unequal experiments (1D-1F)** - Fixes H1 test
2. **Non-IID comparison (2A-2E)** - Shows robustness
3. **Update paper with new results**

### Phase 2: Enhances Paper (MEDIUM PRIORITY)
4. **Client scaling (1G-1I)** - Adds scalability insights
5. **Extended non-IID experiments (2F-2H)**

### Phase 3: Future Work (OPTIONAL)
6. More aggregation methods (Krum, Trimmed Mean standalone)
7. Different model architectures
8. Real-world datasets beyond CIFAR-10

## Code Changes Needed

### 1. Add IID-Unequal Partitioning Function

```python
# In shared/data_utils.py
def partition_data_iid_unequal(dataset, num_clients, size_variation=0.5, seed=42):
    """
    Partition dataset into IID subsets with unequal sizes.

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        size_variation: Controls size heterogeneity (0.5 = ±50% from average)
        seed: Random seed

    Returns:
        dict: {client_id: list of indices}
    """
    np.random.seed(seed)
    num_items = len(dataset)

    # Generate random sizes using Dirichlet distribution
    alpha = np.ones(num_clients) * (1.0 / size_variation)
    size_proportions = np.random.dirichlet(alpha)
    client_sizes = (size_proportions * num_items).astype(int)

    # Adjust to ensure sum equals total
    client_sizes[-1] = num_items - client_sizes[:-1].sum()

    # Randomly shuffle indices (IID)
    indices = np.random.permutation(num_items)

    # Partition based on sizes
    client_dict = {}
    start = 0
    for i in range(num_clients):
        end = start + client_sizes[i]
        client_dict[i] = indices[start:end].tolist()
        start = end

    return client_dict
```

### 2. Create Experiment Runner Script

```bash
# run_enhanced_level1.sh
#!/bin/bash

echo "=========================================="
echo "Level 1 Enhanced Experiments"
echo "=========================================="

# Phase 1: IID-Unequal
echo "Running IID-Unequal experiments..."
python run_fedavg.py --partition iid-unequal --variation 0.5
python run_fedmean.py --partition iid-unequal --variation 0.5
python run_fedmedian.py --partition iid-unequal --variation 0.5

# Phase 2: Client scaling
for num_clients in 10 25 100; do
  echo "Running with $num_clients clients..."
  python run_fedavg.py --num_clients $num_clients
  python run_fedmean.py --num_clients $num_clients
  python run_fedmedian.py --num_clients $num_clients
done
```

### 3. Update Analysis Scripts

Add comparison plots:
- IID-Equal vs IID-Unequal (show FedAvg advantage)
- IID vs Non-IID (show robustness)
- Client count vs convergence speed

## Timeline Estimate

| Phase | Tasks | Time Estimate |
|-------|-------|---------------|
| **Phase 1** | Add IID-unequal function, run 3 experiments, analyze | 4-6 hours |
| **Phase 2** | Client scaling experiments (9 runs), analysis | 6-8 hours |
| **Phase 3** | Non-IID comparison (already implemented, need runs) | 4-6 hours |
| **Phase 4** | Update paper with new results, revise conclusions | 4-6 hours |
| **Total** | Complete enhanced study | 18-26 hours |

## Expected Paper Impact

### Before Enhancement:
- "We compared 3 aggregation strategies under IID conditions"
- Hypothesis 1 was rejected (confusing, since FedAvg should win)
- Limited practical guidance

### After Enhancement:
- "We provide the first comprehensive comparison across IID, non-IID, equal, unequal, and various client counts"
- Clear guidance: "Use FedAvg when sizes vary, FedMean when equal, FedMedian when Byzantine threats exist"
- Scalability insights for deployment planning

## Next Steps

1. **Decide on priority**: Which experiments are most critical?
2. **Implement IID-unequal partitioning**: Most important for fixing H1
3. **Run Phase 1 experiments**: IID-unequal for all 3 methods
4. **Update paper**: Revise hypotheses and results sections
5. **Expand to Phase 2/3**: If time allows and results are promising
