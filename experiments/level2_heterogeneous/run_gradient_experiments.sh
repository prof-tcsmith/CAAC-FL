#!/bin/bash
# Run all gradient-based experiments through the same protocol as weight-sharing experiments
# Total: 3 strategies × 2 conditions × 3 client counts × 3 datasets × 5 seeds = 270 experiments

set -e

echo "========================================"
echo "GRADIENT STRATEGIES EXPERIMENT SUITE"
echo "Strategies: FedSGD, FedAdam, FedTrimmed"
echo "========================================"

# Run gradient experiments for each client count
for num_clients in 10 25 50; do
    echo ""
    echo "========================================"
    echo "Running ${num_clients}-client gradient experiments"
    echo "========================================"

    conda run -n caac-fl python run_noniid_experiments.py \
        --all-conditions \
        --num_clients ${num_clients} \
        --gradient-strategies

    echo "Completed ${num_clients}-client gradient experiments"
done

echo ""
echo "========================================"
echo "ALL GRADIENT EXPERIMENTS COMPLETE"
echo "========================================"
