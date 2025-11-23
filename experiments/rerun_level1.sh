#!/bin/bash
#
# Rerun Level 1 experiments with fixed logging
# These experiments completed but had empty test_accuracy data due to logging bug
#

set -e  # Exit on error

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CONDA_ENV="caac-fl"
ROUNDS=20

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  RERUNNING LEVEL 1 EXPERIMENTS (Fixed Logging)${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Rerunning 9 Level 1 experiments with corrected logging code"
echo "  - 3 IID-Unequal (50 clients)"
echo "  - 6 IID-Equal Client Scaling (10, 25 clients)"
echo ""

cd level1_fundamentals

# Rerun Priority 1: IID-Unequal experiments
echo -e "${GREEN}Priority 1: IID-Unequal (50 clients)${NC}"
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

# Rerun Priority 3: Client Scaling experiments
echo -e "${GREEN}Priority 3: Client Scaling (10, 25 clients)${NC}"
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

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  LEVEL 1 RERUN COMPLETED${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "All 9 Level 1 experiments rerun with corrected logging!"
echo ""
echo "Next step: Run analysis"
echo "  cd level1_fundamentals && python analyze_comprehensive_results.py"
echo ""
