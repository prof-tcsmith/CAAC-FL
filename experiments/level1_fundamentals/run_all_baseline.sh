#!/bin/bash
# Run all baseline aggregation comparison experiments
#
# Experiment matrix:
# - 3 datasets: mnist, fashion_mnist, cifar10
# - 3 strategies: fedavg, fedmean, fedmedian
# - 2 conditions: iid_equal, iid_unequal
# - 5 seeds: 42, 123, 456, 789, 1011
# Total: 3 × 3 × 2 × 5 = 90 experiments

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESULTS_DIR="./level1_fundamentals/results/baseline"
LOG_DIR="./level1_fundamentals/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Configuration
DATASETS=("mnist" "fashion_mnist" "cifar10")
STRATEGIES=("fedavg" "fedmean" "fedmedian")
CONDITIONS=("iid_equal" "iid_unequal")
SEEDS=(42 123 456 789 1011)

NUM_CLIENTS=50
NUM_ROUNDS=50
LOCAL_EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=0.01

# Count total experiments
TOTAL=$((${#DATASETS[@]} * ${#STRATEGIES[@]} * ${#CONDITIONS[@]} * ${#SEEDS[@]}))
CURRENT=0

echo "=============================================="
echo "Baseline Aggregation Comparison Experiments"
echo "=============================================="
echo "Total experiments: $TOTAL"
echo "Results directory: $RESULTS_DIR"
echo ""

START_TIME=$(date +%s)

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "####################################################"
    echo "# Dataset: $DATASET"
    echo "####################################################"

    for STRATEGY in "${STRATEGIES[@]}"; do
        for CONDITION in "${CONDITIONS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                CURRENT=$((CURRENT + 1))

                # Check if result already exists
                RESULT_FILE="$RESULTS_DIR/$DATASET/$CONDITION/${STRATEGY}_seed${SEED}.json"
                if [ -f "$RESULT_FILE" ]; then
                    echo "[$CURRENT/$TOTAL] Skipping (exists): $DATASET | $STRATEGY | $CONDITION | seed=$SEED"
                    continue
                fi

                echo ""
                echo "[$CURRENT/$TOTAL] Running: $DATASET | $STRATEGY | $CONDITION | seed=$SEED"

                LOG_FILE="$LOG_DIR/${DATASET}_${STRATEGY}_${CONDITION}_seed${SEED}.log"

                conda run -n caac-fl --no-capture-output \
                    python level1_fundamentals/run_baseline_experiments.py \
                    --dataset "$DATASET" \
                    --strategy "$STRATEGY" \
                    --condition "$CONDITION" \
                    --seed "$SEED" \
                    --num_clients "$NUM_CLIENTS" \
                    --num_rounds "$NUM_ROUNDS" \
                    --local_epochs "$LOCAL_EPOCHS" \
                    --batch_size "$BATCH_SIZE" \
                    --learning_rate "$LEARNING_RATE" \
                    --results_dir "$RESULTS_DIR" \
                    2>&1 | tee "$LOG_FILE"

                # Brief pause between experiments
                sleep 2
            done
        done
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Total time: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="
