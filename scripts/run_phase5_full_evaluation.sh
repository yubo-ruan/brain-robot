#!/bin/bash
# Phase 5: Full Suite Evaluation
# Evaluates all 10 libero_spatial tasks with both oracle and learned perception
# 20 episodes per task = 200 total episodes per perception mode

set -e

OUTPUT_DIR="logs/phase5_full_evaluation"
N_EPISODES=20
SEED=42

echo "=============================================="
echo "PHASE 5: FULL SUITE EVALUATION"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo "Episodes per task: $N_EPISODES"
echo "Seed: $SEED"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# Results file
RESULTS_FILE="$OUTPUT_DIR/results_summary.txt"
echo "Phase 5 Full Suite Evaluation Results" > "$RESULTS_FILE"
echo "======================================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Track success rates
declare -A ORACLE_RESULTS
declare -A LEARNED_RESULTS

# Run evaluation for all 10 tasks
for TASK_ID in 0 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo "=========================================="
    echo "Task $TASK_ID - Oracle Perception"
    echo "=========================================="

    MUJOCO_GL=egl python scripts/run_evaluation.py \
        --mode hardcoded \
        --perception oracle \
        --task-suite libero_spatial \
        --task-id $TASK_ID \
        --n-episodes $N_EPISODES \
        --seed $SEED \
        --output-dir "$OUTPUT_DIR/oracle_task${TASK_ID}" 2>&1 | tee "$OUTPUT_DIR/oracle_task${TASK_ID}.log"

    # Extract success rate from log
    ORACLE_RATE=$(grep "Success Rate:" "$OUTPUT_DIR/oracle_task${TASK_ID}.log" | tail -1 | awk '{print $3}')
    ORACLE_RESULTS[$TASK_ID]=$ORACLE_RATE

    echo ""
    echo "=========================================="
    echo "Task $TASK_ID - Learned Perception (Cold Bootstrap)"
    echo "=========================================="

    MUJOCO_GL=egl python scripts/run_evaluation.py \
        --mode hardcoded \
        --perception learned \
        --bootstrap cold \
        --task-suite libero_spatial \
        --task-id $TASK_ID \
        --n-episodes $N_EPISODES \
        --seed $SEED \
        --output-dir "$OUTPUT_DIR/learned_cold_task${TASK_ID}" 2>&1 | tee "$OUTPUT_DIR/learned_cold_task${TASK_ID}.log"

    # Extract success rate from log
    LEARNED_RATE=$(grep "Success Rate:" "$OUTPUT_DIR/learned_cold_task${TASK_ID}.log" | tail -1 | awk '{print $3}')
    LEARNED_RESULTS[$TASK_ID]=$LEARNED_RATE

    echo "" >> "$RESULTS_FILE"
    echo "Task $TASK_ID: Oracle=${ORACLE_RATE}, Learned=${LEARNED_RATE}" >> "$RESULTS_FILE"
done

echo ""
echo "=============================================="
echo "PHASE 5 EVALUATION COMPLETE"
echo "=============================================="

# Print summary table
echo ""
echo "| Task | Oracle | Learned (Cold) | Delta |"
echo "|------|--------|----------------|-------|"

ORACLE_TOTAL=0
LEARNED_TOTAL=0

for TASK_ID in 0 1 2 3 4 5 6 7 8 9; do
    O=${ORACLE_RESULTS[$TASK_ID]:-"N/A"}
    L=${LEARNED_RESULTS[$TASK_ID]:-"N/A"}

    # Calculate delta if both are numeric
    if [[ "$O" =~ ^[0-9]+\.?[0-9]*% ]] && [[ "$L" =~ ^[0-9]+\.?[0-9]*% ]]; then
        O_NUM=$(echo "$O" | tr -d '%')
        L_NUM=$(echo "$L" | tr -d '%')
        DELTA=$(echo "$L_NUM - $O_NUM" | bc)
        ORACLE_TOTAL=$(echo "$ORACLE_TOTAL + $O_NUM" | bc)
        LEARNED_TOTAL=$(echo "$LEARNED_TOTAL + $L_NUM" | bc)
        echo "| $TASK_ID    | $O   | $L          | ${DELTA}%  |"
    else
        echo "| $TASK_ID    | $O   | $L          | N/A   |"
    fi
done

ORACLE_AVG=$(echo "scale=1; $ORACLE_TOTAL / 10" | bc)
LEARNED_AVG=$(echo "scale=1; $LEARNED_TOTAL / 10" | bc)
AVG_DELTA=$(echo "scale=1; $LEARNED_AVG - $ORACLE_AVG" | bc)

echo "|------|--------|----------------|-------|"
echo "| AVG  | ${ORACLE_AVG}% | ${LEARNED_AVG}%        | ${AVG_DELTA}%  |"

echo ""
echo "Results saved to: $OUTPUT_DIR"

# Save summary to file
echo "" >> "$RESULTS_FILE"
echo "Summary" >> "$RESULTS_FILE"
echo "=======" >> "$RESULTS_FILE"
echo "Oracle Average: ${ORACLE_AVG}%" >> "$RESULTS_FILE"
echo "Learned Average: ${LEARNED_AVG}%" >> "$RESULTS_FILE"
echo "Delta: ${AVG_DELTA}%" >> "$RESULTS_FILE"
