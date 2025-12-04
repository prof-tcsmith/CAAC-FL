#!/bin/bash
#
# Run Comprehensive Baseline Byzantine-Robust FL Experiments
#
# Based on Li et al. 2024 paper configuration:
# - 25 clients total
# - 50 rounds
# - CIFAR-10, Dirichlet α=0.5 (Non-IID)
# - All 9 aggregation strategies
# - All attack types from the paper
# - Both immediate (round 0) and delayed (round 15) compromise scenarios
#
# Usage:
#   ./run_all_baselines.sh               # Run all experiments
#   ./run_all_baselines.sh --gpu 0       # Use specific GPU
#   ./run_all_baselines.sh --quick       # Quick test (5 rounds)
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
GPU_ARG=""
NUM_ROUNDS=50
OUTPUT_DIR="./results/comprehensive"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ARG="--gpu $2"
            shift 2
            ;;
        --quick)
            NUM_ROUNDS=5
            OUTPUT_DIR="./results/quick_test"
            shift
            ;;
        --rounds)
            NUM_ROUNDS=$2
            shift 2
            ;;
        --output)
            OUTPUT_DIR=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate caac-fl

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/../level3_attacks:${SCRIPT_DIR}/..:$PYTHONPATH"

# Clean up Ray
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Comprehensive Baseline Byzantine-Robust FL Experiments"
echo "========================================================================"
echo "Configuration:"
echo "  Rounds: $NUM_ROUNDS"
echo "  Clients: 25"
echo "  Output: $OUTPUT_DIR"
echo "  GPU: ${GPU_ARG:-auto}"
echo ""
echo "Strategies: fedavg, fedmedian, trimmed, krum, multikrum, geomed, cc, clustering, clippedclustering"
echo "Attacks: none, sign_flipping, random_noise, alie, ipm_small, ipm_large, label_flipping"
echo "Byzantine ratios: 10%, 20%, 30%"
echo "Scenarios: immediate (round 0), delayed (round 15)"
echo "========================================================================"
echo ""

# Count total experiments: 9 strategies × (1 baseline + 6 attacks × 3 ratios × 2 scenarios)
# = 9 × (1 + 36) = 9 × 37 = 333 (but baseline has no delayed)
# More precisely: 9 strategies × (1 baseline + 6 attacks × 3 ratios × 2 scenarios - 6×3 skipped no-attack delayed)
# Actually: 9 × (1 + 18×2) = 9 × 37... let me just count

# Run experiments for each strategy
STRATEGIES="fedavg fedmedian trimmed krum multikrum geomed cc clustering clippedclustering"

start_time=$(date +%s)

for strategy in $STRATEGIES; do
    echo ""
    echo "========================================================================"
    echo "RUNNING STRATEGY: $strategy"
    echo "========================================================================"

    # Run with all attacks and both scenarios
    python run_flower_experiments.py \
        --strategy "$strategy" \
        --all_attacks \
        --all_scenarios \
        --num_rounds "$NUM_ROUNDS" \
        --num_clients 25 \
        --local_epochs 5 \
        --alpha 0.5 \
        --output_dir "$OUTPUT_DIR" \
        $GPU_ARG \
        2>&1 | tee -a "${OUTPUT_DIR}/log_${strategy}.txt"

    # Clean up Ray between strategies to avoid memory issues
    ray stop --force 2>/dev/null || true
    sleep 5
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
echo "Total time: $(printf '%02d:%02d:%02d' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate summary
echo "Generating summary..."
python - << 'EOF'
import json
import csv
import os
from pathlib import Path

output_dir = os.environ.get('OUTPUT_DIR', './results/comprehensive')
results_dir = Path(output_dir)

summary = []
for json_file in sorted(results_dir.glob("*.json")):
    with open(json_file) as f:
        data = json.load(f)

    summary.append({
        'strategy': data.get('strategy', ''),
        'attack': data.get('attack', ''),
        'byzantine_ratio': data.get('byzantine_ratio', 0),
        'scenario': data.get('scenario', ''),
        'final_accuracy': data.get('final_accuracy', 0),
        'final_loss': data.get('final_loss', 0),
        'detection_precision': data.get('detection_stats', {}).get('precision'),
        'detection_recall': data.get('detection_stats', {}).get('recall'),
        'detection_f1': data.get('detection_stats', {}).get('f1'),
    })

if summary:
    csv_path = results_dir / 'all_results_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary saved to: {csv_path}")
else:
    print("No results found to summarize")
EOF

echo "Done!"
