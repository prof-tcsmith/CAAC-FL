#!/bin/bash

###############################################################################
# Level 3: Byzantine Attacks - Run All Experiments
#
# This script runs all 12 experiments (4 methods × 3 attack scenarios):
# - FedAvg, FedMedian, Krum, Trimmed Mean
# - No Attack, Random Noise, Sign Flipping
#
# Expected runtime:
#   - CPU: ~90-120 minutes (12 experiments × 7-10 min each)
#   - GPU: ~30-45 minutes (12 experiments × 2.5-4 min each)
###############################################################################

# Exit on error
set -e

# Set PYTHONPATH to include parent directory for shared modules
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p ./results

echo "================================================================================"
echo "Level 3: Byzantine Attacks - Running All Experiments"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Methods: FedAvg, FedMedian, Krum, Trimmed Mean"
echo "  Attacks: No Attack, Random Noise, Sign Flipping"
echo "  Total Experiments: 12"
echo "  Rounds per Experiment: 50"
echo "  Clients: 15 (3 Byzantine @ 20%)"
echo ""
echo "================================================================================"
echo ""

# Track timing
START_TIME=$(date +%s)
EXPERIMENT_NUM=0
TOTAL_EXPERIMENTS=12

# Function to run a single experiment
run_experiment() {
    local METHOD=$1
    local ATTACK=$2
    local SCRIPT=$3

    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))

    echo ""
    echo -e "${BLUE}[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS]${NC} Running ${GREEN}$METHOD${NC} with ${YELLOW}$ATTACK${NC}..."
    echo "--------------------------------------------------------------------------------"

    EXP_START=$(date +%s)

    # Run the experiment
    python "$SCRIPT" --attack "$ATTACK"

    EXP_END=$(date +%s)
    EXP_DURATION=$((EXP_END - EXP_START))

    echo -e "${GREEN}✓${NC} Completed in ${EXP_DURATION}s"
    echo ""
}

# Run FedAvg experiments
echo -e "${BLUE}=== Running FedAvg Experiments ===${NC}"
run_experiment "FedAvg" "none" "run_fedavg.py"
run_experiment "FedAvg" "random_noise" "run_fedavg.py"
run_experiment "FedAvg" "sign_flipping" "run_fedavg.py"

# Run FedMedian experiments
echo -e "${BLUE}=== Running FedMedian Experiments ===${NC}"
run_experiment "FedMedian" "none" "run_fedmedian.py"
run_experiment "FedMedian" "random_noise" "run_fedmedian.py"
run_experiment "FedMedian" "sign_flipping" "run_fedmedian.py"

# Run Krum experiments
echo -e "${BLUE}=== Running Krum Experiments ===${NC}"
run_experiment "Krum" "none" "run_krum.py"
run_experiment "Krum" "random_noise" "run_krum.py"
run_experiment "Krum" "sign_flipping" "run_krum.py"

# Run Trimmed Mean experiments
echo -e "${BLUE}=== Running Trimmed Mean Experiments ===${NC}"
run_experiment "Trimmed Mean" "none" "run_trimmed_mean.py"
run_experiment "Trimmed Mean" "random_noise" "run_trimmed_mean.py"
run_experiment "Trimmed Mean" "sign_flipping" "run_trimmed_mean.py"

# Run analysis
echo "================================================================================"
echo -e "${BLUE}Running Analysis...${NC}"
echo "================================================================================"
python analyze_results.py

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "================================================================================"
echo -e "${GREEN}All Experiments Completed!${NC}"
echo "================================================================================"
echo ""
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in: ./results/"
echo "  - Individual metrics: level3_*_metrics.csv/json"
echo "  - Summary: level3_summary.csv"
echo "  - Visualization: level3_attack_impact.png"
echo ""
echo "================================================================================"
