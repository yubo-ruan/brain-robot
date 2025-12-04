#!/usr/bin/env python3
"""Run evaluation on all tasks in a suite."""
import subprocess
import sys
import os
import re

def run_task(task_suite, task_id, n_episodes, detector):
    cmd = [
        sys.executable, "scripts/run_evaluation.py",
        "--task-suite", task_suite,
        "--task-id", str(task_id),
        "--n-episodes", str(n_episodes),
        "--perception", "learned",
        "--detector", detector,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr

def parse_results(output):
    """Extract success rate and LIBERO success from output."""
    skill_match = re.search(r'Skill Success Rate:\s+(\d+\.?\d*)%', output)
    libero_match = re.search(r'LIBERO Success Rate:\s+(\d+\.?\d*)%', output)
    
    skill_rate = float(skill_match.group(1)) if skill_match else 0.0
    libero_rate = float(libero_match.group(1)) if libero_match else 0.0
    return skill_rate, libero_rate

if __name__ == "__main__":
    task_suite = sys.argv[1] if len(sys.argv) > 1 else "libero_object"
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    detector = sys.argv[4] if len(sys.argv) > 4 else "gsam"
    
    print(f"Running {task_suite} suite ({n_tasks} tasks, {n_episodes} episodes each)")
    print(f"Detector: {detector}")
    print("=" * 60)
    
    results = []
    for task_id in range(n_tasks):
        print(f"\n--- Task {task_id} ---")
        output = run_task(task_suite, task_id, n_episodes, detector)
        skill_rate, libero_rate = parse_results(output)
        results.append((task_id, skill_rate, libero_rate))
        print(f"  Skill: {skill_rate:.1f}%, LIBERO: {libero_rate:.1f}%")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    avg_skill = sum(r[1] for r in results) / len(results)
    avg_libero = sum(r[2] for r in results) / len(results)
    
    print(f"\nAvg Skill Success:  {avg_skill:.1f}%")
    print(f"Avg LIBERO Success: {avg_libero:.1f}%")
    
    print("\nPer-task results:")
    for task_id, skill, libero in results:
        print(f"  Task {task_id}: Skill {skill:.0f}%, LIBERO {libero:.0f}%")
