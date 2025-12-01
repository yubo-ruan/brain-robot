"""
Asynchronous VLM Planner for real-time robot control.

Runs the VLM in a background thread to avoid blocking the control loop.
"""

import threading
import queue
import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PlanRequest:
    """Request for VLM planning."""
    image: np.ndarray
    task_description: str
    gripper_state: str
    timestamp: float
    request_id: int


@dataclass
class PlanResult:
    """Result from VLM planning."""
    plan: Dict[str, Any]
    request_id: int
    timestamp: float
    inference_time: float


class AsyncVLMPlanner:
    """
    Asynchronous wrapper for VLM planner.

    Runs VLM inference in a background thread, allowing the main
    control loop to continue without blocking.
    """

    def __init__(
        self,
        vlm_planner,  # QwenVLPlanner instance
        max_queue_size: int = 2,
        timeout: float = 5.0,
    ):
        """
        Initialize async planner.

        Args:
            vlm_planner: Synchronous VLM planner instance
            max_queue_size: Maximum pending requests
            timeout: Timeout for waiting for results
        """
        self.vlm = vlm_planner
        self.timeout = timeout

        # Request/result queues
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)

        # State
        self.latest_plan = self._default_plan()
        self.request_counter = 0
        self.pending_request_id = None

        # Thread control
        self._stop_event = threading.Event()
        self._thread = None

        # Stats
        self.total_requests = 0
        self.total_completions = 0
        self.total_timeouts = 0
        self.inference_times = []

    def _default_plan(self) -> Dict[str, Any]:
        """Return default plan."""
        return {
            "observation": {
                "target_object": "unknown",
                "gripper_position": "unknown",
                "distance_to_target": "far",
            },
            "plan": {
                "phase": "approach",
                "movements": [{"direction": "forward", "speed": "slow", "steps": 1}],
                "gripper": "maintain",
                "confidence": 0.3,
            },
            "reasoning": "Default async plan",
        }

    def start(self):
        """Start the background planning thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._planning_loop, daemon=True)
        self._thread.start()
        print("[AsyncVLM] Background thread started")

    def stop(self):
        """Stop the background planning thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        print("[AsyncVLM] Background thread stopped")

    def _planning_loop(self):
        """Main loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                # Wait for request with timeout
                request = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Process request
            start_time = time.time()
            try:
                plan = self.vlm.plan(
                    image=request.image,
                    task_description=request.task_description,
                    gripper_state=request.gripper_state,
                )
            except Exception as e:
                print(f"[AsyncVLM] Error: {e}")
                plan = self._default_plan()

            inference_time = time.time() - start_time

            # Put result
            result = PlanResult(
                plan=plan,
                request_id=request.request_id,
                timestamp=time.time(),
                inference_time=inference_time,
            )

            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                # Drop oldest result
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result)
                except:
                    pass

            self.total_completions += 1
            self.inference_times.append(inference_time)

    def request_plan(
        self,
        image: np.ndarray,
        task_description: str,
        gripper_state: str = "open",
    ) -> bool:
        """
        Submit a plan request (non-blocking).

        Returns True if request was submitted, False if queue is full.
        """
        self.request_counter += 1
        request = PlanRequest(
            image=image.copy(),
            task_description=task_description,
            gripper_state=gripper_state,
            timestamp=time.time(),
            request_id=self.request_counter,
        )

        try:
            self.request_queue.put_nowait(request)
            self.pending_request_id = request.request_id
            self.total_requests += 1
            return True
        except queue.Full:
            return False

    def get_plan(self, blocking: bool = False) -> Dict[str, Any]:
        """
        Get the latest plan.

        Args:
            blocking: If True, wait for a new result

        Returns:
            Latest plan dictionary
        """
        # Check for new results
        try:
            if blocking:
                result = self.result_queue.get(timeout=self.timeout)
            else:
                result = self.result_queue.get_nowait()

            self.latest_plan = result.plan
            self.pending_request_id = None

        except queue.Empty:
            if blocking:
                self.total_timeouts += 1

        return self.latest_plan

    def has_pending_request(self) -> bool:
        """Check if there's a pending request."""
        return self.pending_request_id is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get planning statistics."""
        avg_time = np.mean(self.inference_times) if self.inference_times else 0
        return {
            "total_requests": self.total_requests,
            "total_completions": self.total_completions,
            "total_timeouts": self.total_timeouts,
            "avg_inference_time": avg_time,
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
            "queue_size": self.request_queue.qsize(),
        }

    def reset(self):
        """Reset state for new episode."""
        # Clear queues
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except:
                pass
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                pass

        self.latest_plan = self._default_plan()
        self.pending_request_id = None

        if self.vlm is not None:
            self.vlm.reset()


class SmartReplanTrigger:
    """
    Intelligent re-planning trigger based on various signals.
    """

    def __init__(
        self,
        min_interval: int = 10,
        max_interval: int = 50,
        confidence_threshold: float = 0.5,
        phase_change_replan: bool = True,
    ):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.confidence_threshold = confidence_threshold
        self.phase_change_replan = phase_change_replan

        self.steps_since_plan = 0
        self.previous_phase = None
        self.previous_confidence = 1.0

    def should_replan(
        self,
        current_plan: Dict[str, Any],
        gripper_state_changed: bool = False,
    ) -> bool:
        """
        Determine if we should request a new plan.

        Args:
            current_plan: Current cached plan
            gripper_state_changed: Whether gripper state just changed

        Returns:
            True if should replan
        """
        self.steps_since_plan += 1

        # Always replan after max interval
        if self.steps_since_plan >= self.max_interval:
            self.steps_since_plan = 0
            return True

        # Don't replan before min interval
        if self.steps_since_plan < self.min_interval:
            return False

        # Check confidence
        confidence = current_plan.get("plan", {}).get("confidence", 0.5)
        if confidence < self.confidence_threshold:
            self.steps_since_plan = 0
            return True

        # Check phase change
        if self.phase_change_replan:
            current_phase = current_plan.get("plan", {}).get("phase", "unknown")
            if current_phase != self.previous_phase:
                self.previous_phase = current_phase
                # Don't reset counter - just note the change
                # Could add logic to replan on certain phase transitions

        # Replan on gripper state change
        if gripper_state_changed:
            self.steps_since_plan = 0
            return True

        return False

    def reset(self):
        """Reset trigger state."""
        self.steps_since_plan = 0
        self.previous_phase = None
        self.previous_confidence = 1.0
