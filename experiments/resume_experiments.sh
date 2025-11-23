#!/bin/bash
#
# Resume Comprehensive Experimental Suite
# Continues from where the previous run left off
#

set -e  # Exit on error

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="caac-fl"
ROUNDS=20

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  RESUME COMPREHENSIVE FEDERATED LEARNING EXPERIMENTAL SUITE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Conda Environment: ${CONDA_ENV}"
echo "  Rounds: ${ROUNDS}"
echo "  Start Time: $(date)"
echo ""

echo -e "${GREEN}Completed Experiments (skipping):${NC}"
echo "  ✓ Priority 1: All 3 IID-Unequal experiments"
echo "  ✓ Priority 2: FedAvg (α=0.1, 0.5, 1.0)"
echo "  ✓ Priority 2: FedMedian (α=0.1, 0.5, 1.0)"
echo ""
echo -e "${YELLOW}Remaining Experiments:${NC}"
echo "  → Priority 2: Krum (α=0.1, 0.5, 1.0) - 3 experiments"
echo "  → Priority 3: Client Scaling (10, 25 clients) - 6 experiments"
echo "  → Total: 9 experiments"
echo ""

#############################################################################
# PRIORITY 2 (CONTINUED): Non-IID Experiments - Krum only
#############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  PRIORITY 2 (CONTINUED): NON-IID - KRUM EXPERIMENTS${NC}"
echo -e "${GREEN}  Purpose: Complete Byzantine-robust aggregation tests${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

cd level2_heterogeneous

for alpha in 0.1 0.5 1.0; do
    echo -e "${YELLOW}Running Non-IID (α=${alpha}) with krum...${NC}"

    conda run -n ${CONDA_ENV} python run_krum.py \
        --num_clients 50 \
        --num_rounds ${ROUNDS} \
        --alpha ${alpha} \
        --output_dir ./results/comprehensive

    echo -e "${GREEN}✓ Completed Non-IID α=${alpha} krum${NC}"
    echo ""
done

cd ..

#############################################################################
# PRIORITY 3: Client Scaling Experiments
#############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  PRIORITY 3: CLIENT SCALING EXPERIMENTS${NC}"
echo -e "${GREEN}  Purpose: Test scalability (10, 25 clients)${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

cd level1_fundamentals

for num_clients in 10 25; do
    for agg in fedavg fedmean fedmedian; do
        echo -e "${YELLOW}Running IID-Equal with ${num_clients} clients, ${agg}...${NC}"
        conda run -n ${CONDA_ENV} python run_experiment.py \
            --partition iid-equal \
            --aggregation ${agg} \
            --num_clients ${num_clients} \
            --num_rounds ${ROUNDS} \
            --output_dir ./results/comprehensive
        echo -e "${GREEN}✓ Completed ${num_clients} clients ${agg}${NC}"
        echo ""
    done
done

cd ..

#############################################################################
# SUMMARY
#############################################################################

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  EXPERIMENTAL SUITE COMPLETED${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "End Time: $(date)"
echo ""
echo -e "${GREEN}All remaining experiments completed successfully!${NC}"
echo ""
echo "Results Location:"
echo "  Level 1 (IID): ./level1_fundamentals/results/comprehensive/"
echo "  Level 2 (Non-IID): ./level2_heterogeneous/results/comprehensive/"
echo ""
echo "Total Experiments Completed:"
echo "  Priority 1 (IID-Unequal): 3 experiments"
echo "  Priority 2 (Non-IID): 9 experiments"
echo "  Priority 3 (Client Scaling): 6 experiments"
echo ""
echo -e "${YELLOW}Total: 18 experiments${NC}"
echo ""
echo "Next Steps:"
echo "  1. Run analysis: cd level1_fundamentals && python analyze_comprehensive_results.py"
echo "  2. Generate comparison plots across all dimensions"
echo "  3. Update paper with new findings"
echo ""
