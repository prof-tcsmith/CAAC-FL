#!/bin/bash

# Run all Level 1 experiments
# This script runs FedAvg and FedMedian experiments, then compares results

set -e  # Exit on error

echo "========================================"
echo "Level 1: Federated Learning Fundamentals"
echo "========================================"
echo ""

# Create results directory
mkdir -p results

# Run FedAvg experiment
echo "Step 1/3: Running FedAvg experiment..."
echo "----------------------------------------"
python run_fedavg.py
echo ""

# Run FedMedian experiment
echo "Step 2/3: Running FedMedian experiment..."
echo "----------------------------------------"
python run_fedmedian.py
echo ""

# Analyze and compare results
echo "Step 3/3: Analyzing results..."
echo "----------------------------------------"
python analyze_results.py
echo ""

echo "========================================"
echo "All Level 1 experiments complete!"
echo "========================================"
echo ""
echo "Results saved in ./results/"
echo "  - level1_fedavg_metrics.csv/json"
echo "  - level1_fedmedian_metrics.csv/json"
echo "  - level1_comparison.csv"
echo "  - level1_comparison.png"
echo ""
