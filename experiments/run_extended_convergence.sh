#!/bin/bash
#
# Batch runner for extended convergence experiments (50 rounds)
# Runs Priority 1 and Priority 2 experiments from implementation plan
#

set -e  # Exit on error

# Configuration
ROUNDS=50
OUTPUT_DIR="./level1_fundamentals/results/extended"
LOG_DIR="./logs/extended"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/batch_run.log"
}

# Track experiment count
TOTAL_EXPERIMENTS=12
CURRENT=0
START_TIME=$(date +%s)

log "========================================="
log "Extended Convergence Experiments"
log "Total experiments: $TOTAL_EXPERIMENTS"
log "Rounds per experiment: $ROUNDS"
log "Output directory: $OUTPUT_DIR"
log "========================================="

# Function to run single experiment
run_experiment() {
    local partition=$1
    local aggregation=$2
    local num_clients=$3

    CURRENT=$((CURRENT + 1))

    log ""
    log "Experiment $CURRENT/$TOTAL_EXPERIMENTS: $partition $aggregation c$num_clients"
    log "-----------------------------------------"

    local exp_start=$(date +%s)
    local log_file="$LOG_DIR/${partition}_${aggregation}_c${num_clients}_50rounds.log"

    if conda run -n caac-fl python level1_fundamentals/run_experiment.py \
        --partition "$partition" \
        --aggregation "$aggregation" \
        --num_clients "$num_clients" \
        --num_rounds "$ROUNDS" \
        --output_dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1; then

        local exp_end=$(date +%s)
        local exp_duration=$((exp_end - exp_start))
        log "✓ Completed in ${exp_duration}s"

        # Calculate ETA
        local elapsed=$((exp_end - START_TIME))
        local avg_per_exp=$((elapsed / CURRENT))
        local remaining=$((TOTAL_EXPERIMENTS - CURRENT))
        local eta=$((avg_per_exp * remaining))
        local eta_min=$((eta / 60))

        log "  Progress: $CURRENT/$TOTAL_EXPERIMENTS ($((CURRENT * 100 / TOTAL_EXPERIMENTS))%)"
        log "  Estimated time remaining: ${eta_min} minutes"
    else
        log "✗ FAILED - check $log_file for details"
        return 1
    fi
}

log ""
log "=== PRIORITY 1: IID-Unequal (3 experiments) ==="
log ""

run_experiment "iid-unequal" "fedavg" 50
run_experiment "iid-unequal" "fedmean" 50
run_experiment "iid-unequal" "fedmedian" 50

log ""
log "=== PRIORITY 2: Client Scaling IID-Equal (9 experiments) ==="
log ""

# 10 clients
run_experiment "iid-equal" "fedavg" 10
run_experiment "iid-equal" "fedmean" 10
run_experiment "iid-equal" "fedmedian" 10

# 25 clients
run_experiment "iid-equal" "fedavg" 25
run_experiment "iid-equal" "fedmean" 25
run_experiment "iid-equal" "fedmedian" 25

# 50 clients
run_experiment "iid-equal" "fedavg" 50
run_experiment "iid-equal" "fedmean" 50
run_experiment "iid-equal" "fedmedian" 50

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

log ""
log "========================================="
log "All experiments completed!"
log "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
log "Results saved to: $OUTPUT_DIR"
log "Logs saved to: $LOG_DIR"
log "========================================="
