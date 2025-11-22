#!/bin/bash

# Run all Level 1 experiments
# This script runs FedAvg, FedMean, and FedMedian experiments, then compares results

set -e  # Exit on error

echo "========================================"
echo "Level 1: Federated Learning Fundamentals"
echo "========================================"
echo ""

# Create results directory
mkdir -p results

# Run FedAvg experiment
echo "Step 1/4: Running FedAvg experiment..."
echo "----------------------------------------"
python run_fedavg.py
echo ""

# Run FedMean experiment
echo "Step 2/4: Running FedMean experiment..."
echo "----------------------------------------"
python run_fedmean.py
echo ""

# Run FedMedian experiment
echo "Step 3/4: Running FedMedian experiment..."
echo "----------------------------------------"
python run_fedmedian.py
echo ""

# Analyze and compare results
echo "Step 4/4: Analyzing results..."
echo "----------------------------------------"
python analyze_results.py
echo ""

echo "========================================"
echo "All Level 1 experiments complete!"
echo "========================================"
echo ""
echo "Results saved in ./results/"
echo "  - level1_fedavg_metrics.csv/json"
echo "  - level1_fedmean_metrics.csv/json"
echo "  - level1_fedmedian_metrics.csv/json"
echo "  - level1_comparison.csv"
echo "  - level1_comparison.png"
echo ""
