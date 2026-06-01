"""
BackgroundTaskManager — manages concurrent background tasks with lifecycle.

Allows the agent to offload non-critical work to background threads while
keeping the main brain loop light. Tracks active tasks, handles cleanup,
and provides synchronization primitives.

Typical usage:
    manager = BackgroundTaskManager()
    
    def long_task():
        # do work without blocking brain
        pass
    
    task_id = manager.start_task("research_topic", long_task)
    # ... later ...
    manager.wait_task(task_id)  # <- kills the thread
    manager.cleanup()
"""

import threading
import time
import logging
from uuid import uuid4
from typing import Callable, Any, Optional

log = logging.getLogger("background_tasks")


class BackgroundTask:
    """Wrapper around a background thread with status tracking."""

    def __init__(self, task_id: str, name: str, fn: Callable[[], Any]):
        self.task_id = task_id
        self.name = name
        self.fn = fn
        self.thread = None
        self.result = None
        self.exception = None
        self.start_time = 0
        self.end_time = 0
        self._cancel_event = threading.Event()

    def start(self) -> None:
        """Start the task thread."""
        self.start_time = time.time()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"bg-{self.name}",
        )
        self.thread.start()

    def _run(self) -> None:
        """Execute the task function."""
        try:
            self.result = self.fn()
        except Exception as e:
            self.exception = e
            log.error("Background task %s failed: %s", self.name, e, exc_info=True)
        finally:
            self.end_time = time.time()

    def cancel(self, timeout: float = 2.0) -> bool:
        """Request task cancellation and wait for thread to finish.
        
        Args:
            timeout: Max seconds to wait for thread to finish.
        
        Returns:
            True if thread finished, False if timeout.
        """
        self._cancel_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            return not self.thread.is_alive()
        return True

    def is_running(self) -> bool:
        """Check if task is currently executing."""
        return self.thread is not None and self.thread.is_alive()

    def duration(self) -> float:
        """Return elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time if self.start_time else 0.0

    def should_cancel(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_event.is_set()


class BackgroundTaskManager:
    """Manages a pool of background tasks with lifetime tracking."""

    def __init__(self):
        self._tasks = {}  # task_id -> BackgroundTask
        self._lock = threading.Lock()

    def start_task(self, name: str, fn: Callable[[], Any]) -> str:
        """Start a new background task.
        
        Args:
            name: Human-readable task name.
            fn: Callable that performs the work.
        
        Returns:
            task_id (str) — use this to track/cancel the task.
        """
        task_id = str(uuid4())[:8]
        task = BackgroundTask(task_id, name, fn)

        with self._lock:
            self._tasks[task_id] = task

        task.start()
        log.info("Started background task [%s]: %s", task_id, name)
        return task_id

    def wait_task(self, task_id: str, timeout: float = 10.0) -> bool:
        """Wait for a background task to complete.
        
        Args:
            task_id: Task ID returned by start_task.
            timeout: Max seconds to wait.
        
        Returns:
            True if task finished cleanly, False if timeout or not found.
        """
        with self._lock:
            task = self._tasks.get(task_id)

        if not task:
            log.warning("Task not found: %s", task_id)
            return False

        success = task.cancel(timeout=timeout)
        if not success:
            log.warning("Task %s timed out after %.1f seconds", task_id, timeout)
        return success

    def cancel_all(self, timeout: float = 3.0) -> int:
        """Cancel all running tasks at once.
        
        Args:
            timeout: Max seconds to wait per task.
        
        Returns:
            Number of tasks cancelled.
        """
        with self._lock:
            task_ids = list(self._tasks.keys())

        cancelled = 0
        for task_id in task_ids:
            if self.wait_task(task_id, timeout=timeout):
                cancelled += 1
        return cancelled

    def cleanup(self) -> None:
        """Remove finished tasks from tracking (optional, for memory)."""
        with self._lock:
            finished = [
                task_id for task_id, task in self._tasks.items()
                if not task.is_running()
            ]
            for task_id in finished:
                del self._tasks[task_id]

        if finished:
            log.info("Cleaned up %d finished background tasks", len(finished))

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task object for inspection."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_active(self) -> list[dict]:
        """List all currently active tasks."""
        with self._lock:
            return [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "duration_seconds": task.duration(),
                    "running": task.is_running(),
                }
                for task in self._tasks.values()
            ]

    def stats(self) -> dict:
        """Return diagnostics."""
        with self._lock:
            running = sum(1 for t in self._tasks.values() if t.is_running())
            total = len(self._tasks)

        return {
            "active_tasks": running,
            "total_tracked": total,
            "tasks": self.list_active(),
        }
