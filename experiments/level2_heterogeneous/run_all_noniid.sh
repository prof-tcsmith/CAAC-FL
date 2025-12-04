#!/bin/bash
# Run all Level 2 Non-IID experiments
#
# This runs 45 experiments:
# - 3 datasets (MNIST, Fashion-MNIST, CIFAR-10)
# - 3 strategies (FedAvg, FedMean, FedMedian)
# - 5 seeds for statistical analysis
# - Default alpha=0.5 (moderately non-IID)
#
# Estimated time: ~6-8 hours with GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Level 2: Non-IID Aggregation Study"
echo "========================================"
echo "Started: $(date)"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate caac-fl

# Run all experiments
python run_noniid_experiments.py --all-datasets

echo ""
echo "========================================"
echo "All Level 2 experiments complete!"
echo "Finished: $(date)"
echo "========================================"
