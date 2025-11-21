# Level 2 Implementation Details

## Overview

Level 2 introduces non-IID (non-independent and identically distributed) data distribution using the Dirichlet distribution. This level evaluates how data heterogeneity affects different aggregation strategies without Byzantine attacks.

## Key Differences from Level 1

### Data Distribution

**Level 1 (IID)**:
- Uniform random partitioning
- Each client has ~5,000 samples
- Balanced class distribution per client
- KL divergence: ~0.0007

**Level 2 (Non-IID)**:
- Dirichlet distribution (α=0.5)
- Variable client sizes (2,857-8,718 samples)
- Imbalanced class distribution per client
- KL divergence: ~0.75 (1000× more heterogeneous)

### Aggregation Methods

Level 2 adds a third method:
1. **FedAvg** (baseline)
2. **FedMedian** (robust)
3. **Krum** (distance-based) ← NEW

## Dirichlet Partitioning

### What is Dirichlet Distribution?

The Dirichlet distribution controls class heterogeneity across clients. The concentration parameter α determines the degree of non-IID:

- **α → 0**: Extreme non-IID (each client has few classes)
- **α = 0.1**: Highly non-IID (as used in federated learning papers)
- **α = 0.5**: Moderately non-IID ← **Level 2 choice**
- **α = 1.0**: Slightly non-IID
- **α → ∞**: Approaches IID

### Implementation

```python
from shared.data_utils import partition_data_dirichlet

client_dict = partition_data_dirichlet(
    dataset=train_dataset,
    num_clients=10,
    alpha=0.5,  # Moderate heterogeneity
    seed=42     # Reproducibility
)
```

### Example Distribution

With α=0.5, client class distributions might look like:
- Client 0: 41% trucks, 26% horses, 10% dogs, ...
- Client 1: 54% deer, 19% frogs, 8% ships, ...
- Client 2: 39% automobiles, 32% deer, 18% frogs, ...

## Krum Aggregation

### Algorithm

Krum (Blanchard et al., 2017) selects the single "most representative" client update:

1. **Compute pairwise distances** between all client updates
2. For each client i, compute score = sum of distances to k nearest neighbors
   - k = n - f - 2 (n = total clients, f = expected Byzantine clients)
3. **Select client with minimum score** (most central update)
4. Use that client's update as the global model

### Why Krum?

**Advantages**:
- Theoretically robust to Byzantine attacks (up to f < n/2 - 1)
- Simple distance-based selection
- No averaging needed

**Disadvantages**:
- May struggle with high data heterogeneity (non-IID)
- Discards all other client updates (information loss)
- Sensitive to the choice of k

### Implementation

Custom implementation in `krum_strategy.py`:

```python
class Krum(Strategy):
    def _krum_selection(self, weights_list):
        # Flatten weights
        flattened = [np.concatenate([w.flatten() for w in weights])
                     for weights in weights_list]

        # Compute pairwise distances
        distances = compute_pairwise_distances(flattened)

        # For each client, sum distances to k nearest neighbors
        k = n - f - 2
        scores = []
        for i in range(n):
            k_nearest = get_k_nearest(distances[i], k)
            scores.append(sum(k_nearest))

        # Select client with minimum score
        return argmin(scores)
```

## Expected Behavior

### Performance Expectations

Based on federated learning literature:

| Method | Expected Accuracy | Rationale |
|--------|------------------|-----------|
| FedAvg | 70-75% | Handles moderate heterogeneity reasonably |
| FedMedian | 70-75% | Similar to FedAvg, slightly more stable |
| Krum | 60-70% | May struggle due to high heterogeneity |

### Why Krum May Underperform

With non-IID data (no attacks):
- Client updates are naturally diverse
- "Central" update may not exist
- Discarding other clients loses information
- High heterogeneity confuses distance-based selection

## Heterogeneity Metrics

### KL Divergence

Measures how much client distributions differ from uniform:

```python
KL(client_dist || uniform) = Σ p_i * log(p_i / q_i)
```

- IID: ~0.0007
- Non-IID (α=0.5): ~0.75
- Higher = more heterogeneous

### Class Imbalance

Standard deviation of class proportions across clients:
- IID: ~0.01
- Non-IID: ~0.3-0.4

## File Structure

```
level2_heterogeneous/
├── README.md              # Quick reference
├── IMPLEMENTATION.md      # This file
├── client.py              # Flower client (same as Level 1)
├── krum_strategy.py       # Custom Krum implementation
├── run_fedavg.py          # FedAvg with non-IID data
├── run_fedmedian.py       # FedMedian with non-IID data
├── run_krum.py            # Krum with non-IID data
├── analyze_results.py     # Multi-method comparison + Level 1 comparison
├── run_all.sh             # Execute all experiments
└── results/               # Generated results
```

## Code Walkthrough

### Non-IID Data Loading

```python
# Partition with Dirichlet
client_dict = partition_data_dirichlet(
    train_dataset,
    num_clients=10,
    alpha=0.5,
    seed=42
)

# Analyze heterogeneity
stats = analyze_data_distribution(train_dataset, client_dict)
print(f"KL divergence: {stats['heterogeneity_metrics']['mean_kl_divergence']}")
```

### Krum Strategy Usage

```python
from krum_strategy import Krum

strategy = Krum(
    fraction_fit=1.0,
    evaluate_fn=evaluate_fn,
    initial_parameters=initial_params,
    num_byzantine=0  # No attacks in Level 2
)
```

## Comparison with Level 1

The analysis script compares:
1. **Same method, different data**: FedAvg (IID) vs FedAvg (non-IID)
2. **Different methods, same data**: FedAvg vs FedMedian vs Krum (all non-IID)

Expected findings:
- Performance degradation: 5-10% accuracy loss due to heterogeneity
- Krum underperforms: Distance-based methods struggle with non-IID
- Averaging methods more robust: FedAvg/FedMedian handle heterogeneity better

## Key Insights

### Observation 1: Heterogeneity Impact

Non-IID data causes:
- Lower peak accuracy
- Slower convergence
- Higher variance across rounds
- Method-dependent degradation

### Observation 2: Krum's Challenge

Krum may fail because:
- Non-IID creates naturally diverse updates
- No single "representative" client exists
- Information loss from discarding all but one update

### Observation 3: Averaging Robustness

FedAvg/FedMedian more robust because:
- Incorporate information from all clients
- Natural averaging smooths heterogeneity
- Less sensitive to outliers

## Next Steps (Level 3)

Building on Level 2:
1. Keep non-IID data (Dirichlet α=0.5)
2. Add Byzantine attacks (Random Noise, Sign Flipping)
3. Expand to 15 clients with 20% Byzantine
4. Add detection metrics
5. Evaluate robustness vs. heterogeneity tradeoff

## References

- **Dirichlet Partitioning**: Hsu et al., "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification," 2019
- **Krum**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent," NeurIPS 2017
- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017
- **Non-IID Impact**: Li et al., "Federated Learning on Non-IID Data Silos: An Experimental Study," ICDE 2022
