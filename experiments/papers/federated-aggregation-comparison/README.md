# Federated Learning Aggregation Comparison

**A Comprehensive Experimental Study of Aggregation Strategies in Federated Learning**

## Paper Information

- **Document**: `federated_learning_aggregation_comparison.qmd`
- **Format**: Quarto QMD (renders to HTML)
- **Date**: November 2025
- **Experimental Framework**: Flower (flwr) v1.x with PyTorch

## Experimental Design

This paper presents results from a comprehensive experimental study across three dimensional axes:

### 1. Data Distribution (Priority 1)
- **IID-Equal**: Uniform data distribution, equal dataset sizes across clients
- **IID-Unequal**: Uniform data distribution, heterogeneous dataset sizes (tests FedAvg weighting advantage)

### 2. Aggregation Methods
- **FedAvg** (Federated Averaging): Weighted averaging by dataset size
- **FedMean** (Federated Mean): Unweighted averaging (equal client weights)
- **FedMedian**: Coordinate-wise median aggregation (Byzantine-robust)
- **Krum**: Byzantine-robust aggregation (Level 2 experiments)

### 3. Client Scaling (Priority 3)
- 10 clients
- 25 clients
- 50 clients

### Additional Experiments (Level 2)
- **Non-IID Data**: Dirichlet distribution with α ∈ {0.1, 0.5, 1.0}
- **Byzantine-Robust Aggregation**: Krum and FedMedian

## Evidence Files

### Generated Figures

1. **`comparison_iid_equal_vs_unequal.png`** (250 KB)
   - Tests **Hypothesis H1**: FedAvg weighting advantage with unequal client sizes
   - Shows convergence curves for IID-Equal vs IID-Unequal across all aggregation methods
   - Key Finding: FedAvg (72.98%) outperforms FedMean (70.97%) with unequal sizes (+2.01%)

2. **`comparison_client_scaling.png`** (494 KB)
   - Tests **Hypothesis H3**: Scalability across different client counts
   - Left panel: Convergence curves for 10, 25, 50 clients
   - Right panel: Final accuracy vs client count
   - Key Finding: Accuracy decreases with more clients (78.8% → 75.6% → 73.0% for FedAvg)

3. **`heatmap_grand_comparison.png`** (121 KB)
   - Grand comparison across all experimental conditions
   - Final test accuracy heatmap (partition × aggregation)
   - Visual summary of all Level 1 experimental results

### Data Files

4. **`comprehensive_summary.csv`** (297 bytes)
   - Summary table of all experiments
   - Columns: Partition, Aggregation, Clients, Final Accuracy
   - 9 Level 1 experiments recorded

### Supporting Files

5. **`references.bib`** (3.4 KB)
   - BibTeX references for academic citations
   - Includes: McMahan 2017 (FedAvg), Yin 2018 (Byzantine-robust), Krizhevsky 2009 (CIFAR-10)

## How Evidence Was Generated

### Experimental Configuration

```bash
# Dataset
Dataset: CIFAR-10 (50,000 train, 10,000 test, 10 classes)
Model: SimpleCNN (3-layer CNN)

# Training Parameters
- Local Epochs: 5
- Communication Rounds: 20
- Batch Size: 32
- Learning Rate: 0.01
- Optimizer: SGD
- Device: CUDA (GPU)
```

### Execution Scripts

The experiments were run using automated scripts:

```bash
# Level 1: IID experiments (Equal and Unequal)
cd /path/to/experiments
./run_comprehensive_experiments.sh    # Initial run (Priority 1, 2, 3)
./rerun_level1.sh                      # Rerun with fixed logging

# Level 2: Non-IID experiments
cd level2_heterogeneous
python run_fedavg.py --alpha 0.1 --num_clients 50 --num_rounds 20
python run_fedmedian.py --alpha 0.1 --num_clients 50 --num_rounds 20
python run_krum.py --alpha 0.1 --num_clients 50 --num_rounds 20
# (repeated for α = 0.5, 1.0)
```

### Analysis Generation

```bash
# Generate plots and summary
cd level1_fundamentals
python analyze_comprehensive_results.py

# Outputs:
# - comparison_iid_equal_vs_unequal.png
# - comparison_client_scaling.png
# - heatmap_grand_comparison.png
# - comprehensive_summary.csv
```

### Analysis Script Details

The `analyze_comprehensive_results.py` script:
1. Loads all JSON experiment results from `./results/comprehensive/`
2. Parses experiment metadata (partition type, aggregation method, client count)
3. Extracts final accuracy and convergence trajectories
4. Generates comparison plots using matplotlib and seaborn
5. Performs hypothesis testing with verdict assignment
6. Exports summary CSV for tabular analysis

## Key Experimental Findings

### Hypothesis H1: FedAvg Weighting Advantage (IID-Unequal)
**Result**: ✅ **CONFIRMED**
- FedAvg: 72.98%
- FedMean: 70.97%
- Difference: +2.01%
- **Conclusion**: FedAvg's dataset-size weighting provides measurable advantage with heterogeneous client sizes

### Hypothesis H2: Non-IID Robustness
**Status**: ⏳ Partial analysis (Level 2 data not included in current plots)
- Experiments completed for α ∈ {0.1, 0.5, 1.0}
- FedAvg, FedMedian, and Krum tested
- Full analysis requires integration of Level 2 results

### Hypothesis H3: Scalability
**Result**: ⚠️ **PERFORMANCE DEGRADES WITH SCALE**
- 10 clients: 78.84% (FedAvg)
- 25 clients: 75.58% (FedAvg)
- 50 clients: 72.98% (FedAvg)
- **Conclusion**: Accuracy decreases as client count increases, likely due to increased gradient noise and slower convergence

## Reproducing Results

### Prerequisites
```bash
conda env create -f environment.yml
conda activate caac-fl
```

### Running Experiments
```bash
# Full comprehensive suite (18 experiments, ~2-3 hours)
cd /path/to/CAAC-FL/experiments
./run_comprehensive_experiments.sh

# Resume from failure
./resume_experiments.sh

# Rerun specific experiments
./rerun_level1.sh
```

### Generating Figures
```bash
cd level1_fundamentals
python analyze_comprehensive_results.py
```

### Rendering Paper
```bash
# Install Quarto: https://quarto.org/docs/get-started/
cd papers/federated-aggregation-comparison
quarto render federated_learning_aggregation_comparison.qmd
# Output: federated_learning_aggregation_comparison.html
```

## Paper Structure

1. **Abstract**: Summary of motivation, methods, and key findings
2. **Introduction**: Background on federated learning and aggregation challenges
3. **Problem Statement**: Research questions and experimental hypotheses
4. **Methodology**: Dataset, model architecture, aggregation strategies, experimental design
5. **Results**: Quantitative analysis with figures and tables
6. **Discussion**: Interpretation of findings, hypothesis evaluation
7. **Limitations**: Scope constraints and potential confounding factors
8. **Future Work**: Extensions and open questions
9. **Conclusions**: Summary of contributions
10. **References**: Academic citations

## File Manifest

```
papers/federated-aggregation-comparison/
├── README.md                                    (this file)
├── federated_learning_aggregation_comparison.qmd  (main paper)
├── references.bib                                (citations)
├── comparison_iid_equal_vs_unequal.png           (Figure 1)
├── comparison_client_scaling.png                 (Figure 2)
├── heatmap_grand_comparison.png                  (Figure 3)
└── comprehensive_summary.csv                     (Table data)
```

## Notes

- The current analysis includes **Level 1 experiments** (IID-Equal, IID-Unequal, Client Scaling)
- **Level 2 experiments** (Non-IID with varying α) were completed but are not yet integrated into the comprehensive analysis plots
- Future work should integrate Level 2 results for complete multi-dimensional analysis
- All experiments used 20 communication rounds (reduced from original 50 for faster iteration)

## Contact

For questions about the experimental setup or to request raw data, please refer to the main CAAC-FL repository.
