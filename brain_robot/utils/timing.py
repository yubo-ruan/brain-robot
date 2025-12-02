"""Performance profiling utilities.

Simple timing utilities for measuring execution performance.
"""

from contextlib import contextmanager
from collections import defaultdict
import time
from typing import Dict, Any


class Timer:
    """Simple performance profiler for measuring code execution times.
    
    Usage:
        timer = Timer()
        with timer.measure("perception"):
            perception = perceive(env)
        with timer.measure("skill_execution"):
            result = skill.run(env, world_state, args)
        
        print(timer.summary())
    """
    
    def __init__(self):
        self.times: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def measure(self, name: str):
        """Context manager to measure execution time of a code block.
        
        Args:
            name: Label for this timing measurement.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.times[name] += elapsed
            self.counts[name] += 1
    
    def start(self, name: str):
        """Start a named timer (for non-context-manager usage).
        
        Args:
            name: Label for this timing measurement.
        """
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop a named timer and record the elapsed time.
        
        Args:
            name: Label for the timer to stop.
            
        Returns:
            Elapsed time in seconds.
        """
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.perf_counter() - self._start_times[name]
        self.times[name] += elapsed
        self.counts[name] += 1
        del self._start_times[name]
        return elapsed
    
    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all timing measurements.
        
        Returns:
            Dictionary mapping names to {total, count, avg} dicts.
        """
        return {
            name: {
                "total_sec": self.times[name],
                "count": self.counts[name],
                "avg_sec": self.times[name] / self.counts[name] if self.counts[name] > 0 else 0,
                "avg_ms": (self.times[name] / self.counts[name] * 1000) if self.counts[name] > 0 else 0,
            }
            for name in sorted(self.times.keys())
        }
    
    def reset(self):
        """Reset all timing measurements."""
        self.times.clear()
        self.counts.clear()
        self._start_times.clear()
    
    def __str__(self) -> str:
        """String representation of timing summary."""
        lines = ["Timing Summary:"]
        for name, stats in self.summary().items():
            lines.append(
                f"  {name}: {stats['total_sec']:.3f}s total, "
                f"{stats['count']} calls, {stats['avg_ms']:.2f}ms avg"
            )
        return "\n".join(lines)
