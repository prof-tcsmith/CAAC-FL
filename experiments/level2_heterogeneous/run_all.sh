#!/bin/bash

# Run all Level 2 experiments
# This script runs FedAvg, FedMedian, and Krum experiments on non-IID data

set -e  # Exit on error

echo "========================================"
echo "Level 2: Heterogeneous Data Distribution"
echo "Non-IID (Dirichlet α=0.5)"
echo "========================================"
echo ""

# Create results directory
mkdir -p results

# Run FedAvg experiment
echo "Step 1/4: Running FedAvg experiment..."
echo "----------------------------------------"
python run_fedavg.py
echo ""

# Run FedMedian experiment
echo "Step 2/4: Running FedMedian experiment..."
echo "----------------------------------------"
python run_fedmedian.py
echo ""

# Run Krum experiment
echo "Step 3/4: Running Krum experiment..."
echo "----------------------------------------"
python run_krum.py
echo ""

# Analyze and compare results
echo "Step 4/4: Analyzing results..."
echo "----------------------------------------"
python analyze_results.py
echo ""

echo "========================================"
echo "All Level 2 experiments complete!"
echo "========================================"
echo ""
echo "Results saved in ./results/"
echo "  - level2_fedavg_metrics.csv/json"
echo "  - level2_fedmedian_metrics.csv/json"
echo "  - level2_krum_metrics.csv/json"
echo "  - level2_comparison.csv"
echo "  - level2_comparison.png"
echo "  - level1_vs_level2.png"
echo ""
