#!/bin/bash
#
# Periodic Progress Monitor for Comprehensive Experiments
#

LOG_FILE="/home/tim/Workspace/_RESEARCH/CAAC-FL/experiments/comprehensive_experiments.log"
LEVEL1_RESULTS="/home/tim/Workspace/_RESEARCH/CAAC-FL/experiments/level1_fundamentals/results/comprehensive"
LEVEL2_RESULTS="/home/tim/Workspace/_RESEARCH/CAAC-FL/experiments/level2_heterogeneous/results/comprehensive"
MONITOR_LOG="/home/tim/Workspace/_RESEARCH/CAAC-FL/experiments/progress_monitor.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================" | tee -a "$MONITOR_LOG"
echo "COMPREHENSIVE EXPERIMENT PROGRESS MONITOR" | tee -a "$MONITOR_LOG"
echo "Started: $(date)" | tee -a "$MONITOR_LOG"
echo "======================================================================" | tee -a "$MONITOR_LOG"
echo "" | tee -a "$MONITOR_LOG"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    echo "" | tee -a "$MONITOR_LOG"
    echo "======================================================================" | tee -a "$MONITOR_LOG"
    echo "Progress Check: $TIMESTAMP" | tee -a "$MONITOR_LOG"
    echo "======================================================================" | tee -a "$MONITOR_LOG"

    # Check if main script is still running
    if pgrep -f "run_comprehensive_experiments.sh" > /dev/null; then
        echo -e "${GREEN}✓ Comprehensive suite is RUNNING${NC}" | tee -a "$MONITOR_LOG"
    else
        echo -e "${RED}✗ Comprehensive suite has STOPPED or COMPLETED${NC}" | tee -a "$MONITOR_LOG"
        echo "Final check at: $TIMESTAMP" | tee -a "$MONITOR_LOG"
        break
    fi

    # Current experiment
    echo "" | tee -a "$MONITOR_LOG"
    echo "Current Activity:" | tee -a "$MONITOR_LOG"
    if [ -f "$LOG_FILE" ]; then
        CURRENT=$(tail -20 "$LOG_FILE" | grep -E "Running|Completed" | tail -1)
        if [ -n "$CURRENT" ]; then
            echo "  $CURRENT" | tee -a "$MONITOR_LOG"
        fi

        # Latest accuracy (if available)
        LATEST_ACC=$(tail -50 "$LOG_FILE" | grep -E "Round [0-9]+/[0-9]+:" | tail -1)
        if [ -n "$LATEST_ACC" ]; then
            echo "  Latest: $LATEST_ACC" | tee -a "$MONITOR_LOG"
        fi
    fi

    # Count completed experiments
    echo "" | tee -a "$MONITOR_LOG"
    echo "Completed Experiments:" | tee -a "$MONITOR_LOG"

    LEVEL1_COUNT=0
    LEVEL2_COUNT=0

    if [ -d "$LEVEL1_RESULTS" ]; then
        LEVEL1_COUNT=$(ls "$LEVEL1_RESULTS"/*.json 2>/dev/null | wc -l)
    fi

    if [ -d "$LEVEL2_RESULTS" ]; then
        LEVEL2_COUNT=$(ls "$LEVEL2_RESULTS"/*.json 2>/dev/null | wc -l)
    fi

    TOTAL_COMPLETED=$((LEVEL1_COUNT + LEVEL2_COUNT))

    echo "  Level 1 (IID): $LEVEL1_COUNT files" | tee -a "$MONITOR_LOG"
    echo "  Level 2 (Non-IID): $LEVEL2_COUNT files" | tee -a "$MONITOR_LOG"
    echo -e "  ${BLUE}Total: $TOTAL_COMPLETED / 18 experiments${NC}" | tee -a "$MONITOR_LOG"

    # Progress percentage
    PROGRESS=$((TOTAL_COMPLETED * 100 / 18))
    echo "  Progress: ${PROGRESS}%" | tee -a "$MONITOR_LOG"

    # Resource usage
    echo "" | tee -a "$MONITOR_LOG"
    echo "Resource Usage:" | tee -a "$MONITOR_LOG"

    PYTHON_PROCS=$(ps aux | grep "run_experiment.py" | grep -v grep)
    if [ -n "$PYTHON_PROCS" ]; then
        echo "$PYTHON_PROCS" | awk '{printf "  CPU: %s%%, MEM: %s%%, Runtime: %s\n", $3, $4, $10}' | tee -a "$MONITOR_LOG"
    else
        echo "  No active Python processes" | tee -a "$MONITOR_LOG"
    fi

    # Estimate completion time
    if [ $TOTAL_COMPLETED -gt 0 ]; then
        echo "" | tee -a "$MONITOR_LOG"
        REMAINING=$((18 - TOTAL_COMPLETED))
        echo "  Remaining: $REMAINING experiments" | tee -a "$MONITOR_LOG"

        # Rough estimate: 15-20 minutes per experiment
        EST_MINUTES=$((REMAINING * 17))
        EST_HOURS=$((EST_MINUTES / 60))
        EST_MINS=$((EST_MINUTES % 60))
        echo "  Estimated time remaining: ~${EST_HOURS}h ${EST_MINS}m" | tee -a "$MONITOR_LOG"
    fi

    echo "" | tee -a "$MONITOR_LOG"
    echo "Next check in 5 minutes..." | tee -a "$MONITOR_LOG"

    # Wait 5 minutes
    sleep 300
done

echo "" | tee -a "$MONITOR_LOG"
echo "======================================================================" | tee -a "$MONITOR_LOG"
echo "MONITORING COMPLETED" | tee -a "$MONITOR_LOG"
echo "Finished: $(date)" | tee -a "$MONITOR_LOG"
echo "======================================================================" | tee -a "$MONITOR_LOG"

# Final summary
echo "" | tee -a "$MONITOR_LOG"
echo "Final Results:" | tee -a "$MONITOR_LOG"
echo "  Level 1 files: $(ls "$LEVEL1_RESULTS"/*.json 2>/dev/null | wc -l)" | tee -a "$MONITOR_LOG"
echo "  Level 2 files: $(ls "$LEVEL2_RESULTS"/*.json 2>/dev/null | wc -l)" | tee -a "$MONITOR_LOG"
echo "" | tee -a "$MONITOR_LOG"
echo "Next step: Run analysis script" | tee -a "$MONITOR_LOG"
echo "  cd level1_fundamentals && python analyze_comprehensive_results.py" | tee -a "$MONITOR_LOG"
