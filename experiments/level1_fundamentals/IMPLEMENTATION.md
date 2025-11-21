# Level 1 Implementation Details

## Overview

Level 1 implements basic federated learning with IID data distribution to establish baseline behavior. This serves as the foundation for understanding FL dynamics before introducing heterogeneity and attacks in later levels.

## Architecture

### Components

1. **Client Implementation** (`client.py`)
   - Extends `flwr.client.NumPyClient`
   - Handles local training on client data
   - Implements `fit()` for training and `evaluate()` for evaluation
   - Uses SimpleCNN model from shared utilities

2. **Server Strategies**
   - **FedAvg** (`run_fedavg.py`): Weighted averaging by dataset size
   - **FedMedian** (`run_fedmedian.py`): Coordinate-wise median aggregation

3. **Shared Utilities** (from `../shared/`)
   - `models.py`: SimpleCNN architecture
   - `data_utils.py`: Data loading and IID partitioning
   - `metrics.py`: Evaluation metrics and logging

### Data Flow

```
1. Server initializes global model
2. For each round:
   a. Server sends global model to all clients
   b. Clients train locally on their data
   c. Clients send updated parameters to server
   d. Server aggregates updates (FedAvg or FedMedian)
   e. Server evaluates global model on test set
3. Repeat for specified number of rounds
```

## Implementation Decisions

### Why Flower Framework?

- **Simulation mode**: Easy to run multiple clients on single machine
- **Built-in strategies**: FedAvg, FedMedian, and others available
- **Extensibility**: Easy to implement custom strategies for later levels
- **Production-ready**: Can scale to real distributed settings if needed

### Why IID Distribution for Level 1?

- Establishes baseline without data heterogeneity complications
- Both FedAvg and FedMedian should perform similarly
- Validates implementation correctness
- Provides comparison point for Level 2 (non-IID)

### Model Choice: SimpleCNN

- Lightweight (~122K parameters)
- Fast training for quick iterations
- Sufficient complexity for CIFAR-10
- Consistent with literature (similar architectures in reference papers)

### Configuration Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Clients | 10 | Standard small-scale FL setting |
| Rounds | 50 | Sufficient for convergence |
| Batch size | 32 | Balance between memory and convergence |
| Learning rate | 0.01 | Standard SGD rate for CIFAR-10 |
| Local epochs | 1 | Follows FedSGD approach |
| Client participation | 100% | All clients in each round |

## File Descriptions

### Core Files

- **`client.py`**: Flower client implementation with local training logic
- **`run_fedavg.py`**: Execute FedAvg experiment
- **`run_fedmedian.py`**: Execute FedMedian experiment
- **`analyze_results.py`**: Compare results and generate visualizations
- **`run_all.sh`**: Orchestrate all experiments

### Supporting Files

- **`test_setup.py`**: Verify setup before running experiments
- **`README.md`**: Quick reference guide
- **`IMPLEMENTATION.md`**: This file (detailed implementation notes)

## Metrics Collected

For each round:
- **Test Accuracy**: % correct predictions on test set
- **Test Loss**: Cross-entropy loss on test set

Saved formats:
- CSV: For easy import into spreadsheets
- JSON: For programmatic access
- PNG: Visualization plots

## Expected Results

### FedAvg
- Should converge to ~75-80% test accuracy
- Smooth convergence curve
- Typical FL baseline behavior

### FedMedian
- Similar accuracy to FedAvg (~75-80%)
- Possibly slightly slower convergence
- More stable (robust to outliers, though none present)

### Key Observation
With IID data and no attacks, both methods should perform similarly. This validates:
1. Implementation correctness
2. Data distribution is truly IID
3. Baseline for comparison with later levels

## Code Walkthrough

### Client Training Loop

```python
def fit(self, parameters, config):
    # 1. Set global model parameters
    self.set_parameters(parameters)

    # 2. Train locally
    stats = train_model(
        self.model,
        self.train_loader,
        self.optimizer,
        self.criterion,
        device=self.device,
        epochs=self.local_epochs
    )

    # 3. Return updated parameters
    return (
        self.get_parameters(config={}),
        len(self.train_loader.dataset),
        {'train_loss': stats['train_loss']}
    )
```

### Server Aggregation

```python
# FedAvg: Weighted average
strategy = FedAvg(
    fraction_fit=1.0,  # All clients
    evaluate_fn=evaluate_fn,  # Centralized evaluation
    initial_parameters=initial_params
)

# FedMedian: Coordinate-wise median
strategy = FedMedian(
    fraction_fit=1.0,
    evaluate_fn=evaluate_fn,
    initial_parameters=initial_params
)
```

## Extending to Level 2

To prepare for Level 2 (non-IID data):
1. Replace `partition_data_iid()` with `partition_data_dirichlet(alpha=0.5)`
2. Add Krum strategy as third aggregation method
3. Analyze performance degradation compared to Level 1
4. Document heterogeneity metrics (KL divergence, class imbalance)

## Debugging Tips

### If accuracy is very low (<40%):
- Check data augmentation is correct
- Verify model architecture
- Check learning rate isn't too high
- Ensure labels are correctly loaded

### If experiments crash:
- Check GPU memory (reduce batch size if needed)
- Verify all dependencies installed
- Check file paths are correct
- Run `test_setup.py` first

### If results differ from expected:
- Check random seed is set
- Verify data partitioning is correct
- Compare with shared/models.py test run
- Check CIFAR-10 normalization values

## References

- Flower documentation: https://flower.dev/docs/
- FedAvg paper: McMahan et al. (2017)
- FedMedian: Yin et al. (2018) - coordinate-wise median
