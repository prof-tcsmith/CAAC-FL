#!/bin/bash
#
# Comprehensive Experimental Suite Runner
# Runs all experiments for Priority 1, 2, and 3
#
# Priority 1: IID-Unequal (test FedAvg weighting advantage)
# Priority 2: Non-IID (test heterogeneity robustness)
# Priority 3: Client Scaling (test scalability)
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
ROUNDS=20  # Reduced from 50 for faster iteration

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  COMPREHENSIVE FEDERATED LEARNING EXPERIMENTAL SUITE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Conda Environment: ${CONDA_ENV}"
echo "  Rounds: ${ROUNDS}"
echo "  Start Time: $(date)"
echo ""

#############################################################################
# PRIORITY 1: IID-Unequal Experiments
#############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  PRIORITY 1: IID-UNEQUAL EXPERIMENTS${NC}"
echo -e "${GREEN}  Purpose: Test FedAvg weighting advantage vs FedMean${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

cd level1_fundamentals

for agg in fedavg fedmean fedmedian; do
    echo -e "${YELLOW}Running IID-Unequal with ${agg}...${NC}"
    conda run -n ${CONDA_ENV} python run_experiment.py \
        --partition iid-unequal \
        --aggregation ${agg} \
        --num_clients 50 \
        --num_rounds ${ROUNDS} \
        --size_variation 0.5 \
        --output_dir ./results/comprehensive
    echo -e "${GREEN}✓ Completed IID-Unequal ${agg}${NC}"
    echo ""
done

cd ..

#############################################################################
# PRIORITY 2: Non-IID Experiments (Level 2)
#############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  PRIORITY 2: NON-IID EXPERIMENTS (Level 2)${NC}"
echo -e "${GREEN}  Purpose: Test heterogeneity robustness${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

cd level2_heterogeneous

for agg in fedavg fedmedian krum; do
    for alpha in 0.1 0.5 1.0; do
        echo -e "${YELLOW}Running Non-IID (α=${alpha}) with ${agg}...${NC}"

        # Run based on which script exists
        if [ "${agg}" = "fedavg" ]; then
            script="run_fedavg.py"
        elif [ "${agg}" = "fedmedian" ]; then
            script="run_fedmedian.py"
        elif [ "${agg}" = "krum" ]; then
            script="run_krum.py"
        fi

        conda run -n ${CONDA_ENV} python ${script} \
            --num_clients 50 \
            --num_rounds ${ROUNDS} \
            --alpha ${alpha} \
            --output_dir ./results/comprehensive

        echo -e "${GREEN}✓ Completed Non-IID α=${alpha} ${agg}${NC}"
        echo ""
    done
done

cd ..

#############################################################################
# PRIORITY 3: Client Scaling Experiments
#############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  PRIORITY 3: CLIENT SCALING EXPERIMENTS${NC}"
echo -e "${GREEN}  Purpose: Test scalability (10, 25, 50 clients)${NC}"
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

# Note: 50 clients with IID-equal already done in original experiments

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
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo ""
echo "Results Location:"
echo "  Level 1 (IID): ./level1_fundamentals/results/comprehensive/"
echo "  Level 2 (Non-IID): ./level2_heterogeneous/results/comprehensive/"
echo ""
echo "Summary of Experiments Run:"
echo ""
echo "  Priority 1 - IID-Unequal (3 experiments):"
echo "    - FedAvg, FedMean, FedMedian with unequal client dataset sizes"
echo ""
echo "  Priority 2 - Non-IID (9 experiments):"
echo "    - FedAvg, FedMedian, Krum × Dirichlet α ∈ {0.1, 0.5, 1.0}"
echo ""
echo "  Priority 3 - Client Scaling (6 experiments):"
echo "    - FedAvg, FedMean, FedMedian × clients ∈ {10, 25}"
echo "    (Note: 50 clients already completed in original experiments)"
echo ""
echo -e "${YELLOW}Total New Experiments: 18${NC}"
echo ""
echo "Next Steps:"
echo "  1. Run analysis: python level1_fundamentals/analyze_comprehensive_results.py"
echo "  2. Generate comparison plots across all dimensions"
echo "  3. Update paper with new findings"
echo ""
