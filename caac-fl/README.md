# CAAC-FL: Client-Adaptive Anomaly-Aware Clipping for Byzantine-Robust Federated Learning

Implementation of CAAC-FL algorithm for heterogeneous healthcare federated learning with Byzantine robustness.

## Project Structure

```
caac-fl/
├── src/                      # Source code
│   ├── aggregators/         # Aggregation methods (CAAC-FL, FedAvg, Krum, etc.)
│   ├── attacks/            # Byzantine attack implementations
│   ├── datasets/           # Dataset loaders and partitioners
│   ├── fl_core/            # Flower-based FL server and client
│   ├── models/             # Model architectures
│   ├── profiles/           # Client profile management
│   └── utils/              # Utilities (metrics, logging, visualization)
├── configs/                 # Experiment configurations
├── scripts/                 # Execution scripts
├── tests/                   # Unit and integration tests
├── notebooks/              # Analysis notebooks
├── data/                   # Dataset storage
└── results/                # Experiment results
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/prof-tcsmith/CACL-WITS.git
cd CACL-WITS/caac-fl
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running a Single Experiment

```bash
python scripts/run_experiment.py \
    --single \
    --dataset mimic \
    --aggregator caac_fl \
    --heterogeneity extreme \
    --byzantine-fraction 0.2 \
    --attack sign_flip \
    --seed 42
```

### Running Core Experiments

Run the full experimental matrix from the protocol:

```bash
python scripts/run_experiment.py --config configs/core_experiments.yaml
```

## Key Components

### 1. CAAC-FL Aggregator (`src/aggregators/caac_fl.py`)

Implements the canonical CAAC-FL algorithm with:
- Multi-dimensional anomaly detection (magnitude, direction, temporal)
- Client-specific behavioral profiling
- Adaptive clipping thresholds
- Reliability-weighted aggregation

### 2. FL Server (`src/fl_core/server.py`)

Flower-based federated learning server with:
- Custom strategy for CAAC-FL
- Profile management
- Checkpoint saving
- Metrics tracking

### 3. FL Client (`src/fl_core/client.py`)

Flower client implementation with:
- Local training
- Byzantine attack capabilities
- Gradient computation
- Model evaluation

### 4. Byzantine Attacks (`src/attacks/`)

Implemented attacks:
- Sign-flipping
- ALIE (A Little Is Enough)
- Slow-drift poisoning
- Random noise
- Inner Product Manipulation (IPM)

### 5. Baseline Aggregators

- **FedAvg**: Standard federated averaging
- **Krum**: Distance-based robust aggregation
- **FLTrust**: Trust bootstrapping with root dataset
- **Trimmed Mean**: Statistical filtering
- **Median**: Coordinate-wise median

## Configuration

Experiments are configured via YAML files in `configs/`. Key parameters:

```yaml
experiment:
  num_rounds: 100
  seed: 42
  
federation:
  num_clients: 20
  heterogeneity:
    alpha: 0.1  # Dirichlet parameter
    
aggregators:
  caac_fl:
    beta: 0.9        # EWMA decay
    gamma: 0.1       # Reliability smoothing
    lambda_mag: 0.4  # Magnitude weight
    lambda_dir: 0.4  # Direction weight
    lambda_temp: 0.2 # Temporal weight
    tau_anomaly: 2.0 # Anomaly threshold
    bootstrap_rounds: 10
```

## Datasets

The implementation supports:
- **MIMIC-III**: ICU mortality prediction
- **ISIC 2019**: Melanoma detection
- **ChestX-ray8**: Multi-label disease classification

### Data Preparation

Place preprocessed datasets in `data/` directory:
```
data/
├── mimic/
│   ├── train/
│   ├── val/
│   └── test/
├── isic/
│   ├── train/
│   ├── val/
│   └── test/
└── chestxray/
    ├── train/
    ├── val/
    └── test/
```

## Experimental Protocol

The implementation follows the experimental design from `Protocol-CAAC-FL-Revised.md`:

### Core Experiments

1. **H1 - Heterogeneity Preservation**: Clean training with varying heterogeneity
2. **H2 - Multi-Dimensional Robustness**: Byzantine attacks under extreme heterogeneity
3. **H3 - Temporal Discrimination**: Slow-drift attacks to test temporal detection

### Metrics

- **Performance**: AUROC, AUPRC, accuracy, F1
- **Robustness**: Performance degradation under attacks
- **Detection**: Benign FPR, malicious TPR, detection latency
- **Computational**: Aggregation time, memory usage

## Testing

Run unit tests:
```bash
pytest tests/unit/
```

Run integration tests:
```bash
pytest tests/integration/
```

## Results Analysis

Analyze experiment results using provided notebooks:
```bash
jupyter lab notebooks/analysis_core.ipynb
```

Generate plots:
```bash
python scripts/analyze_results.py --results-dir results/
```

## Implementation Status

### Completed ✅
- Core CAAC-FL algorithm
- Flower framework integration
- Byzantine attack implementations
- Experiment configuration system
- Client profile management

### TODO 📝
- Dataset loading and partitioning
- Model architectures (MLP, ResNet18)
- Metrics computation
- Visualization tools
- Statistical analysis
- FLTrust and other baselines
- Comprehensive testing

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{smith2025caacfl,
  title={Distinguishing Medical Diversity from Byzantine Attacks: 
         Client-Adaptive Anomaly Detection for Healthcare Federated Learning},
  author={Smith, Timothy and Bhattacherjee, Anol and Komara, Raghu Ram},
  booktitle={WITS 2025},
  year={2025}
}
```

## License

MIT License - see LICENSE file

## Contact

- Timothy Smith - [email]
- Anol Bhattacherjee - [email]
- Raghu Ram Komara - [email]

University of South Florida