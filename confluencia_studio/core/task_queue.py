"""Background Task Queue for ConfluenciaStudio.

Provides async task execution with progress tracking, cancellation support,
and status monitoring. Used for long-running operations like model training,
data processing, and pipeline execution.
"""

from __future__ import annotations

import threading
import uuid
import time
import json
import traceback
from enum import Enum
from typing import Callable, Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

try:
    from PyQt6.QtCore import QObject, pyqtSignal
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QObject = object
    # Create dummy signal class for non-Qt environments
    class pyqtSignal:
        def __init__(self, *args):
            pass
        def emit(self, *args):
            pass


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a background task."""
    id: str
    label: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Task':
        """Create from dictionary."""
        d['status'] = TaskStatus(d['status'])
        return cls(**d)


class StudioTaskQueue(QObject if PYQT_AVAILABLE else object):
    """Manages background task execution with progress tracking.

    Features:
    - Thread pool execution
    - Progress callbacks
    - Cancellation support
    - Status signals (Qt) or callbacks (non-Qt)
    - Task persistence and recovery

    Usage:
        queue = StudioTaskQueue()
        task_id = queue.submit("Training model", train_model, data_path, epochs=100)
        queue.get_status(task_id)
    """

    # Qt signals (only used when PyQt6 is available)
    task_started = pyqtSignal(str)      # task_id
    task_progress = pyqtSignal(str, float, str)  # task_id, progress, message
    task_finished = pyqtSignal(str, bool, object)  # task_id, success, result
    task_failed = pyqtSignal(str, str)  # task_id, error_message

    def __init__(self, max_workers: int = 4, persistence_dir: Optional[str] = None):
        if PYQT_AVAILABLE:
            super().__init__()

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, Task] = {}
        self._futures: Dict[str, Future] = {}
        self._cancellation_flags: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._persistence_dir = Path(persistence_dir) if persistence_dir else Path.home() / ".confluencia" / "tasks"
        self._persistence_dir.mkdir(parents=True, exist_ok=True)

        # Non-Qt callback storage
        self._callbacks: Dict[str, List[Callable]] = {
            'started': [],
            'progress': [],
            'finished': [],
            'failed': [],
        }

        # Load persisted tasks on startup
        self._load_persisted_tasks()

    def submit(self, label: str, fn: Callable, *args, **kwargs) -> str:
        """Submit a new task for execution.

        Args:
            label: Human-readable task description
            fn: Function to execute
            *args, **kwargs: Arguments to pass to fn

        Returns:
            task_id: Unique identifier for the task
        """
        task_id = str(uuid.uuid4())[:8]
        task = Task(id=task_id, label=label)

        with self._lock:
            self._tasks[task_id] = task
            self._cancellation_flags[task_id] = threading.Event()

        # Wrap the function to track progress and handle cancellation
        def wrapped():
            return self._run_task(task_id, fn, args, kwargs)

        future = self._executor.submit(wrapped)
        with self._lock:
            self._futures[task_id] = future

        self._persist_task(task)
        return task_id

    def _run_task(self, task_id: str, fn: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute a task with progress tracking."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        cancel_event = self._cancellation_flags.get(task_id)

        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._emit_started(task_id)

            # Create progress callback
            def progress_callback(progress: float, message: str = ""):
                task.progress = progress
                task.message = message
                self._emit_progress(task_id, progress, message)
                self._persist_task(task)

            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()
                self._persist_task(task)
                return None

            # Inject progress callback if function accepts it
            import inspect
            sig = inspect.signature(fn)
            if 'progress_callback' in sig.parameters:
                kwargs['progress_callback'] = progress_callback
            if 'cancel_event' in sig.parameters:
                kwargs['cancel_event'] = cancel_event

            # Execute the function
            result = fn(*args, **kwargs)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 1.0
            task.completed_at = time.time()
            self._persist_task(task)
            self._emit_finished(task_id, True, result)
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            task.completed_at = time.time()
            self._persist_task(task)
            self._emit_failed(task_id, task.error)
            return None

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancellation was requested, False if task not found or already done
        """
        with self._lock:
            task = self._tasks.get(task_id)
            cancel_event = self._cancellation_flags.get(task_id)

            if not task or not cancel_event:
                return False

            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False

            cancel_event.set()
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self._persist_task(task)
            return True

    def get_status(self, task_id: str) -> Optional[Task]:
        """Get current task status."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List all tasks, optionally filtered by status."""
        with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def clear_completed(self) -> int:
        """Remove completed/failed/cancelled tasks.

        Returns:
            Number of tasks removed
        """
        with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            for tid in to_remove:
                del self._tasks[tid]
                self._cancellation_flags.pop(tid, None)
                self._futures.pop(tid, None)
                # Remove persisted file
                task_file = self._persistence_dir / f"{tid}.json"
                if task_file.exists():
                    task_file.unlink()
            return len(to_remove)

    # --- Callback registration (for non-Qt environments) ---

    def on_started(self, callback: Callable[[str], None]):
        """Register callback for task start events."""
        self._callbacks['started'].append(callback)

    def on_progress(self, callback: Callable[[str, float, str], None]):
        """Register callback for progress updates."""
        self._callbacks['progress'].append(callback)

    def on_finished(self, callback: Callable[[str, bool, Any], None]):
        """Register callback for task completion."""
        self._callbacks['finished'].append(callback)

    def on_failed(self, callback: Callable[[str, str], None]):
        """Register callback for task failure."""
        self._callbacks['failed'].append(callback)

    # --- Internal emit methods ---

    def _emit_started(self, task_id: str):
        if PYQT_AVAILABLE:
            self.task_started.emit(task_id)
        for cb in self._callbacks['started']:
            try: cb(task_id)
            except: pass

    def _emit_progress(self, task_id: str, progress: float, message: str):
        if PYQT_AVAILABLE:
            self.task_progress.emit(task_id, progress, message)
        for cb in self._callbacks['progress']:
            try: cb(task_id, progress, message)
            except: pass

    def _emit_finished(self, task_id: str, success: bool, result: Any):
        if PYQT_AVAILABLE:
            self.task_finished.emit(task_id, success, result)
        for cb in self._callbacks['finished']:
            try: cb(task_id, success, result)
            except: pass

    def _emit_failed(self, task_id: str, error: str):
        if PYQT_AVAILABLE:
            self.task_failed.emit(task_id, error)
        for cb in self._callbacks['failed']:
            try: cb(task_id, error)
            except: pass

    # --- Persistence ---

    def _persist_task(self, task: Task):
        """Save task state to disk."""
        task_file = self._persistence_dir / f"{task.id}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task.to_dict(), f, indent=2)

    def _load_persisted_tasks(self):
        """Load persisted tasks on startup."""
        for task_file in self._persistence_dir.glob("*.json"):
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task = Task.from_dict(json.load(f))
                self._tasks[task.id] = task
                self._cancellation_flags[task.id] = threading.Event()
            except Exception:
                pass

    def shutdown(self, wait: bool = True):
        """Shutdown the executor.

        Args:
            wait: If True, wait for running tasks to complete
        """
        self._executor.shutdown(wait=wait)


# Global singleton instance
_task_queue: Optional[StudioTaskQueue] = None

def get_task_queue() -> StudioTaskQueue:
    """Get the global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = StudioTaskQueue()
    return _task_queue
