#!/bin/bash
# Compare oracle vs learned perception across tasks

cd /workspace/brain_robot

export MUJOCO_GL=egl

echo "=============================================="
echo "PERCEPTION COMPARISON: ORACLE vs LEARNED"
echo "=============================================="

# Run oracle perception on tasks 0-4
for task_id in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "Task ${task_id} - Oracle Perception"
    echo "=========================================="
    python scripts/run_evaluation.py \
        --mode hardcoded \
        --perception oracle \
        --task-suite libero_spatial \
        --task-id "${task_id}" \
        --n-episodes 10 \
        --output-dir logs/phase4_comparison
done

# Run learned perception on tasks 0-4
for task_id in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "Task ${task_id} - Learned Perception"
    echo "=========================================="
    python scripts/run_evaluation.py \
        --mode hardcoded \
        --perception learned \
        --task-suite libero_spatial \
        --task-id "${task_id}" \
        --n-episodes 10 \
        --output-dir logs/phase4_comparison
done

echo ""
echo "=============================================="
echo "COMPARISON COMPLETE"
echo "=============================================="
echo "Results saved to logs/phase4_comparison/"
