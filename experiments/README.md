# CAAC-FL Experimental Framework

## Overview

This directory contains a series of progressively complex experiments designed to understand, validate, and demonstrate the Client-Adaptive Anomaly-Aware Clipping (CAAC-FL) protocol for Byzantine-robust federated learning in healthcare settings.

## Experimental Progression

The experiments are organized into 5 levels of increasing complexity:

### Level 1: Federated Learning Fundamentals ✓ Implemented
**Status**: Implementation complete, ready for execution
**Purpose**: Establish baseline FL behavior with IID data
**Methods**: FedAvg vs FedMedian
**Data**: CIFAR-10, IID distribution, 10 clients
**Attacks**: None
**Success Criteria**: Both methods achieve ~75-80% accuracy

### Level 2: Heterogeneous Data ✓ Implemented
**Status**: Implementation complete, ready for execution
**Purpose**: Understand impact of non-IID data
**Methods**: FedAvg, FedMedian, Krum
**Data**: CIFAR-10, Dirichlet α=0.5, 10 clients
**Attacks**: None
**Success Criteria**: Observe performance degradation, Krum may struggle with heterogeneity

### Level 3: Basic Byzantine Attacks ✓ Implemented
**Status**: Implementation complete, ready for execution
**Purpose**: Evaluate robustness to simple attacks
**Methods**: FedAvg, FedMedian, Krum, Trimmed Mean
**Data**: CIFAR-10, Dirichlet α=0.5, 15 clients (20% Byzantine)
**Attacks**: Random Noise, Sign Flipping
**Success Criteria**: Robust methods maintain >60% accuracy under attack

### Level 4: Advanced Attacks & Detection
**Status**: Planned
**Purpose**: Test against sophisticated attacks
**Methods**: Krum, Trimmed Mean, FLTrust, Behavioral Tracking Prototype
**Data**: CIFAR-10, Dirichlet α=0.5, 20 clients (25% Byzantine)
**Attacks**: ALIE, Label Flipping, IPM
**Success Criteria**: Detection TPR >80%, model accuracy >55%

### Level 5: Full CAAC-FL Protocol
**Status**: Planned
**Purpose**: Validate complete CAAC-FL approach
**Methods**: CAAC-FL with client-adaptive behavioral profiling
**Data**: CIFAR-10, Dirichlet α=0.3, 20 clients (30% Byzantine)
**Attacks**: All attack types, including adaptive attacks
**Success Criteria**: TPR >90%, FPR <10%, accuracy >70%

## Directory Structure

```
experiments/
├── README.md                  # This file
├── SETUP.md                   # Installation and setup guide
├── EXPERIMENT-PLAN.md         # Detailed plan for all 5 levels
├── requirements.txt           # Python dependencies
│
├── shared/                    # Shared utilities across all levels
│   ├── models.py              # SimpleCNN, MLP architectures
│   ├── data_utils.py          # Data loading, partitioning (IID, Dirichlet, power-law)
│   ├── metrics.py             # Evaluation, detection metrics, logging
│   └── __init__.py
│
├── level1_fundamentals/       # ✓ IMPLEMENTED
│   ├── README.md              # Quick reference
│   ├── IMPLEMENTATION.md      # Detailed implementation notes
│   ├── client.py              # Flower client implementation
│   ├── run_fedavg.py          # FedAvg experiment
│   ├── run_fedmedian.py       # FedMedian experiment
│   ├── analyze_results.py     # Result comparison and visualization
│   ├── test_setup.py          # Setup verification
│   ├── run_all.sh             # Execute all Level 1 experiments
│   └── results/               # Generated results (CSV, JSON, plots)
│
├── level2_heterogeneous/      # ✓ IMPLEMENTED
│   ├── README.md              # Quick reference
│   ├── IMPLEMENTATION.md      # Detailed technical notes
│   ├── client.py              # Flower client (same as Level 1)
│   ├── krum_strategy.py       # Custom Krum aggregation
│   ├── run_fedavg.py          # FedAvg on non-IID data
│   ├── run_fedmedian.py       # FedMedian on non-IID data
│   ├── run_krum.py            # Krum on non-IID data
│   ├── analyze_results.py     # Comparison + heterogeneity analysis
│   ├── run_all.sh             # Execute all Level 2 experiments
│   └── results/               # Generated results (CSV, JSON, plots)
│
├── level3_attacks/             # ✓ IMPLEMENTED
│   ├── README.md               # Quick reference
│   ├── IMPLEMENTATION.md       # Detailed technical notes
│   ├── LEVEL3-SUMMARY.md       # Implementation summary
│   ├── attacks.py              # Byzantine attack implementations
│   ├── trimmed_mean_strategy.py # Trimmed Mean aggregation
│   ├── client.py               # Byzantine-capable client
│   ├── run_fedavg.py           # FedAvg with attacks
│   ├── run_fedmedian.py        # FedMedian with attacks
│   ├── run_krum.py             # Krum with attacks
│   ├── run_trimmed_mean.py     # Trimmed Mean with attacks
│   ├── analyze_results.py      # Attack impact analysis
│   ├── run_all.sh              # Execute all Level 3 experiments (12 total)
│   └── results/                # Generated results (CSV, JSON, plots)
│
├── level4_advanced_attacks/   # TODO
│   └── (to be implemented)
│
├── level5_caacfl/            # TODO
│   └── (to be implemented)
│
└── analysis/                  # Cross-level analysis (TODO)
    └── (to be implemented)
```

## Quick Start

### 1. Setup Environment

**Quick Installation** (CPU):
```bash
cd experiments
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-cpu.txt
```

**Alternative Methods**:
- GPU installation: Use `requirements-gpu.txt`
- Conda: Use `conda env create -f environment.yml`
- Development: Add `requirements-dev.txt`

See [INSTALL.md](INSTALL.md) for quick start or [SETUP.md](SETUP.md) for detailed instructions.

### 2. Run Level 1 Experiments

```bash
cd level1_fundamentals

# Verify setup
python test_setup.py

# Run experiments
bash run_all.sh

# Results will be in ./results/
```

## Shared Utilities

All levels use common utilities from the `shared/` directory:

### Models (`shared/models.py`)
- **SimpleCNN**: Lightweight CNN for CIFAR-10 (~122K params)
- **MLP**: Fully connected network for comparison
- Factory function: `create_model()`

### Data Utilities (`shared/data_utils.py`)
- **`load_cifar10()`**: Load and preprocess CIFAR-10
- **`partition_data_iid()`**: IID partitioning
- **`partition_data_dirichlet()`**: Non-IID partitioning (α parameter)
- **`partition_data_power_law()`**: Highly heterogeneous sizes
- **`analyze_data_distribution()`**: Compute heterogeneity metrics

### Metrics (`shared/metrics.py`)
- **`evaluate_model()`**: Compute accuracy and loss
- **`train_model()`**: Local training loop
- **`compute_detection_metrics()`**: TPR, FPR, F1 for Byzantine detection
- **`MetricsLogger`**: CSV/JSON logging
- **`compute_gradient_norm()`**: For gradient analysis
- **`compute_cosine_similarity()`**: For behavioral tracking

## Key Design Decisions

1. **Incremental Complexity**: Each level builds on previous ones
2. **Code Reuse**: Shared utilities minimize duplication
3. **Reproducibility**: Fixed seeds, detailed configuration logging
4. **Framework Choice**: Flower for FL simulation (extensible, production-ready)
5. **Dataset**: CIFAR-10 (standard benchmark, manageable size)
6. **Metrics**: Standard ML metrics + Byzantine detection metrics

## Implementation Status

| Component | Status | Files |
|-----------|--------|-------|
| Shared utilities | ✓ Complete | models.py, data_utils.py, metrics.py |
| Level 1 core | ✓ Complete | client.py, run_*.py, analyze_results.py |
| Level 1 docs | ✓ Complete | README.md, IMPLEMENTATION.md |
| Level 2 core | ✓ Complete | krum_strategy.py, run_*.py (×3), analyze_results.py |
| Level 2 docs | ✓ Complete | README.md, IMPLEMENTATION.md |
| Level 3-5 | ⏳ Pending | - |
| Cross-level analysis | ⏳ Pending | - |

## Next Steps

### Immediate (Level 1)
1. Install dependencies (see SETUP.md)
2. Run `test_setup.py` to verify installation
3. Execute `run_all.sh` to run Level 1 experiments
4. Review results and validate baseline performance

### Short-term (Level 2) ✓ Complete
1. ✓ Implement non-IID data experiments
2. ✓ Add Krum aggregation strategy
3. ✓ Compare heterogeneity impact on different aggregation methods
4. Run experiments and validate results

### Medium-term (Levels 3-4)
1. Implement attack models (ALIE, Label Flipping, etc.)
2. Add detection mechanisms
3. Prototype behavioral tracking

### Long-term (Level 5)
1. Implement full CAAC-FL protocol
2. Adaptive behavioral profiling
3. Comprehensive evaluation suite
4. Cross-level comparison analysis

## Expected Timeline

- **Level 1**: 2-3 days (implementation complete, testing pending)
- **Level 2**: 3-4 days
- **Level 3**: 5-7 days
- **Level 4**: 7-10 days
- **Level 5**: 7-10 days
- **Analysis**: 3-5 days

**Total**: ~3-4 weeks

## References

- **Main Paper**: 2025WITS-CAAC-FL (work in progress)
- **Literature Review**: `../literature-review/`
- **Detailed Plan**: `EXPERIMENT-PLAN.md`
- **Quick Reference**: `../literature-review/QUICK-REFERENCE.md`

## Support

For issues or questions:
1. Check SETUP.md for installation issues
2. Review IMPLEMENTATION.md for code details
3. Refer to EXPERIMENT-PLAN.md for methodology
4. Consult literature-review/ for theoretical background
