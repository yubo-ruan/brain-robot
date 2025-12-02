#!/bin/bash
# Bootstrap Ablation Study: Oracle vs Cold-Start
# This runs the same evaluation with both bootstrap modes to measure
# the impact of oracle bootstrap on learned perception performance.

set -e
cd /workspace/brain_robot

export MUJOCO_GL=egl

EPISODES=20
TASKS="0 1 2 3 4"
SEEDS="42 123"

echo "=============================================="
echo "BOOTSTRAP ABLATION STUDY"
echo "=============================================="
echo "Episodes per task: ${EPISODES}"
echo "Tasks: ${TASKS}"
echo "Seeds: ${SEEDS}"
echo "=============================================="

# Run oracle bootstrap across all tasks and seeds
for seed in ${SEEDS}; do
    for task_id in ${TASKS}; do
        echo ""
        echo "=========================================="
        echo "Oracle Bootstrap - Task ${task_id} - Seed ${seed}"
        echo "=========================================="
        python scripts/run_evaluation.py \
            --mode hardcoded \
            --perception learned \
            --bootstrap oracle \
            --task-suite libero_spatial \
            --task-id "${task_id}" \
            --n-episodes "${EPISODES}" \
            --seed "${seed}" \
            --output-dir logs/bootstrap_ablation
    done
done

# Run cold bootstrap across all tasks and seeds
for seed in ${SEEDS}; do
    for task_id in ${TASKS}; do
        echo ""
        echo "=========================================="
        echo "Cold Bootstrap - Task ${task_id} - Seed ${seed}"
        echo "=========================================="
        python scripts/run_evaluation.py \
            --mode hardcoded \
            --perception learned \
            --bootstrap cold \
            --task-suite libero_spatial \
            --task-id "${task_id}" \
            --n-episodes "${EPISODES}" \
            --seed "${seed}" \
            --output-dir logs/bootstrap_ablation
    done
done

echo ""
echo "=============================================="
echo "BOOTSTRAP ABLATION COMPLETE"
echo "=============================================="
echo "Results saved to logs/bootstrap_ablation/"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_bootstrap_ablation.py"
