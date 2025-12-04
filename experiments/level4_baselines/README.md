# Level 4: Baseline Byzantine-Robust FL Experiments

Comprehensive baseline experiments for comparing CAAC-FL against state-of-the-art Byzantine-robust aggregation schemes.

## Reference

Based on:
> Li et al. (2024), "An Experimental Study of Byzantine-Robust Aggregation Schemes in Federated Learning", IEEE Transactions on Big Data.

## Novel Contributions Beyond Li et al. 2024

This implementation extends Li et al.'s experimental framework with three novel dimensions that were not studied in the original paper:

### 1. Compromise Timing Study (Delayed Attacks)
Li et al. only tested **immediate attacks** (starting at round 0). We introduce the **delayed compromise threat model** where attacks begin at various points during training:

| Compromise Round | Description |
|------------------|-------------|
| 0 | Immediate (Li et al. baseline) |
| 10 | Early delayed |
| 20 | Mid-training |
| 30 | Late delayed |
| 40 | Very late |

**Research Question**: Does partial model convergence provide inherent robustness against Byzantine attacks?

### 2. Attack Window Study (Transient Attacks)
We introduce **transient attack windows** where attacks have both a start AND end round:

| Window | Description |
|--------|-------------|
| [0, end) | Full attack (Li et al. baseline) |
| [0, 25) | Early attack, then honest |
| [25, end) | Late compromise only |
| [10, 40) | Mid-training window |
| [0, 10) | Brief initial attack |

**Research Question**: Can models recover after transient attacks? How does attack duration affect final accuracy?

### 3. Non-IID Severity Study
Li et al. only tested α=0.1 (severe) and IID. We test across the heterogeneity spectrum:

| Dirichlet α | Heterogeneity Level |
|-------------|---------------------|
| 0.1 | Severe (some classes missing per client) |
| 0.3 | High |
| 0.5 | Moderate (our default) |
| 1.0 | Mild |

**Research Question**: How does data heterogeneity interact with Byzantine robustness?

## Aggregation Strategies

| Strategy | Description | Detection | Reference |
|----------|-------------|-----------|-----------|
| FedAvg | Weighted average (baseline, non-robust) | No | McMahan et al., 2017 |
| FedMedian | Coordinate-wise median | No | Yin et al., 2018 |
| TrimmedMean | Remove extreme values, then average | No | Yin et al., 2018 |
| Krum | Select single client closest to others | Yes | Blanchard et al., 2017 |
| Multi-Krum | Select top-k clients closest to others | Yes | Blanchard et al., 2017 |
| GeoMed | Geometric median (Weiszfeld algorithm) | No | Chen et al., 2017 |
| CC | Centered Clipping (iterative, Eq. 8) | No | Karimireddy et al., 2021 |
| Clustering | Agglomerative clustering (average linkage) | Yes | Sattler et al., 2020 |
| ClippedClustering | Clip BEFORE clustering (historical median τ) | Yes | Li et al., 2024 |

## Attack Types

| Attack | Description | Parameters |
|--------|-------------|------------|
| none | No attack (baseline) | - |
| sign_flipping | Negate gradients | - |
| random_noise | Gaussian noise | noise_scale=5.0 |
| alie | A Little Is Enough | Based on client count |
| ipm_small | Inner Product Manipulation | ε=0.5 (reduces magnitude) |
| ipm_large | Inner Product Manipulation | ε=100 (reverses direction) |
| label_flipping | Train on flipped labels | - |

## Usage

### Novel Study Modes

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate caac-fl

# NOVEL: Compromise Timing Study
# Tests when attacks start (delayed compromise threat model)
python run_flower_experiments.py --timing_study --all_strategies

# NOVEL: Attack Window Study
# Tests transient attacks with start AND end rounds
python run_flower_experiments.py --window_study --all_strategies

# NOVEL: Non-IID Severity Study
# Tests varying Dirichlet alpha values
python run_flower_experiments.py --alpha_study --all_strategies
```

### Single Experiment with Attack Window

```bash
# Attack window from round 10 to 30
python run_flower_experiments.py \
    --strategy krum \
    --attack sign_flipping \
    --byzantine_ratio 0.2 \
    --compromise_round 10 \
    --attack_end_round 30 \
    --output_dir ./results/flower
```

### Standard Mode (Li et al. Compatible)

```bash
# Run all attacks for a strategy (immediate scenario)
python run_flower_experiments.py \
    --strategy multikrum \
    --all_attacks \
    --output_dir ./results/multikrum

# Run with delayed scenario
python run_flower_experiments.py \
    --strategy clippedclustering \
    --all_attacks \
    --all_scenarios \
    --output_dir ./results/clippedclustering
```

### Comprehensive Experiments

```bash
# Run all 9 strategies with all attacks
./run_all_baselines.sh

# Quick test (5 rounds)
./run_all_baselines.sh --quick

# With specific GPU
./run_all_baselines.sh --gpu 0
```

## Experimental Setup

| Parameter | Default | Li et al. 2024 |
|-----------|---------|----------------|
| Clients | 25 | 20 |
| Byzantine ratios | 10%, 20%, 30% | 25% |
| Rounds | 50 | 600 (FedAvg) |
| Local epochs | 5 | 50 SGD steps |
| Dataset | CIFAR-10 | CIFAR-10, MNIST |
| Dirichlet α | 0.5 | 0.1 or IID |

## Output

Results are saved as JSON files with:
- Experiment configuration (including attack window)
- Final accuracy and loss
- Per-round metrics history
- Detection statistics (for strategies with client selection)

File naming convention:
- `{strategy}_{attack}_byz{ratio}_seed{seed}_imm.json` - Immediate attack
- `{strategy}_{attack}_byz{ratio}_seed{seed}_d{round}.json` - Delayed attack
- `{strategy}_{attack}_byz{ratio}_seed{seed}_w{start}-{end}.json` - Attack window
- `{strategy}_{attack}_byz{ratio}_seed{seed}_a{alpha}.json` - Non-default alpha

## Detection Metrics

For strategies with client selection (Krum, Multi-Krum, Clustering, ClippedClustering):

- **TP (True Positive)**: Byzantine clients correctly rejected
- **FP (False Positive)**: Honest clients incorrectly rejected
- **TN (True Negative)**: Honest clients correctly selected
- **FN (False Negative)**: Byzantine clients incorrectly selected

Metrics:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * Precision * Recall / (Precision + Recall)

## Comparison with CAAC-FL

Results from this level can be compared against Level 5 (CAAC-FL) experiments using the same:
- Attack configurations
- Client setup (25 clients)
- Data distribution (Non-IID)
- Novel threat models (delayed compromise, attack windows)

The key difference is that CAAC-FL uses adaptive context-aware trust weighting rather than geometric selection or clipping.

## Files

- `run_flower_experiments.py`: Main experiment runner with all strategies and novel study modes
- `run_all_baselines.sh`: Script to run comprehensive experiments
