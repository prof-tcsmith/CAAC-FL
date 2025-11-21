# Level 1: Federated Learning Fundamentals

## Overview

This level establishes baseline federated learning behavior with IID data distribution and no attacks. The goal is to understand basic FL dynamics and compare standard aggregation (FedAvg) with a robust aggregation method (FedMedian/Trimmed Mean).

## Objectives

1. Implement basic FL simulation using Flower framework
2. Compare FedAvg vs FedMedian aggregation strategies
3. Establish baseline performance metrics
4. Understand convergence behavior with IID data

## Experimental Setup

### Dataset
- **Dataset**: CIFAR-10
- **Distribution**: IID (equal random partitioning)
- **Clients**: 10
- **Training samples per client**: ~5,000
- **Test set**: Centralized (10,000 samples)

### Model Architecture
- **Model**: SimpleCNN
- **Parameters**: ~122K trainable parameters
- **Architecture**:
  - Conv1: 3→32 channels
  - Conv2: 32→64 channels
  - FC1: 64×8×8→128
  - FC2: 128→10

### Training Configuration
- **Rounds**: 50
- **Local epochs per round**: 1
- **Batch size**: 32
- **Learning rate**: 0.01
- **Optimizer**: SGD with momentum=0.9
- **Client participation**: 100% (all 10 clients)

### Aggregation Methods

1. **FedAvg** (Baseline)
   - Simple weighted averaging by dataset size
   - Standard federated learning aggregation

2. **FedMedian** (Robust)
   - Coordinate-wise median of client updates
   - Robust to outliers (though no attacks in Level 1)
   - Provides comparison point for robust aggregation

## Expected Outcomes

Since this is IID data with no attacks:
- Both methods should converge to similar accuracy (~75-80% on CIFAR-10)
- FedAvg might converge slightly faster
- FedMedian may be more stable but potentially slower
- No significant performance difference expected (validates implementation)

## Metrics Collected

- Test accuracy per round
- Test loss per round
- Training time per round
- Convergence speed (rounds to reach target accuracy)

## Files

- `client.py`: Flower client implementation
- `server.py`: Flower server with aggregation strategies
- `run_fedavg.py`: Run FedAvg experiment
- `run_fedmedian.py`: Run FedMedian experiment
- `analyze_results.py`: Compare and visualize results
- `run_all.sh`: Execute all experiments

## Running Experiments

```bash
# Run both experiments
bash run_all.sh

# Or run individually
python run_fedavg.py
python run_fedmedian.py

# Generate comparison plots
python analyze_results.py
```

## Success Criteria

- [x] Both methods reach >70% test accuracy
- [x] Similar convergence patterns (validates IID setup)
- [x] Clean metrics logging
- [x] Reproducible results

## Next Steps (Level 2)

- Introduce non-IID data (Dirichlet α=0.5)
- Add third aggregation method (Krum)
- Observe performance degradation patterns
