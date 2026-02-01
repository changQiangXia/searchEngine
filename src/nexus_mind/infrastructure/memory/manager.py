"""GPU Memory Manager with 3-level circuit breaker for 3080ti safety.

This module provides comprehensive GPU memory management with automatic
fallback mechanisms to prevent OOM errors on GPUs with limited VRAM (e.g., 12GB).
"""

from __future__ import annotations

import gc
import threading
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, TypeVar

import psutil
import torch


class MemoryPressureLevel(Enum):
    """GPU memory pressure levels for circuit breaker.

    NORMAL: <60% - Safe operating range
    WARNING: 60-80% - Start monitoring, prepare for cleanup
    CRITICAL: 80-90% - Aggressive cleanup, offload non-essential models
    EMERGENCY: >90% - Emergency measures, fallback to CPU
    """

    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass(frozen=True)
class MemoryStats:
    """GPU and system memory statistics.

    Attributes:
        gpu_total: Total GPU memory in bytes
        gpu_used: Currently allocated GPU memory in bytes
        gpu_cached: Cached GPU memory in bytes
        ram_available: Available system RAM in bytes
        gpu_usage_pct: Percentage of GPU memory used (0-100)
    """

    gpu_total: int
    gpu_used: int
    gpu_cached: int
    ram_available: int

    @property
    def gpu_usage_pct(self) -> float:
        """Calculate GPU memory usage percentage."""
        if self.gpu_total == 0:
            return 0.0
        return (self.gpu_used / self.gpu_total) * 100

    @property
    def gpu_available(self) -> int:
        """Calculate available GPU memory."""
        return self.gpu_total - self.gpu_used

    def __str__(self) -> str:
        """Human-readable memory stats."""
        if self.gpu_total == 0:
            return f"MemoryStats(GPU: N/A, RAM: {self.ram_available / 1e9:.2f}GB available)"
        return (
            f"MemoryStats("
            f"GPU: {self.gpu_used/1e9:.2f}/{self.gpu_total/1e9:.2f}GB "
            f"({self.gpu_usage_pct:.1f}%), "
            f"RAM: {self.ram_available/1e9:.2f}GB available)"
        )


class ModelProtocol(Protocol):
    """Protocol for models that can be moved between devices."""

    def cpu(self) -> Any: ...
    def cuda(self, device: int | None = None) -> Any: ...
    def to(self, device: str | torch.device) -> Any: ...
    def state_dict(self) -> dict[str, Any]: ...


T = TypeVar("T")


class GPUMemoryManager:
    """Singleton GPU memory manager with circuit breaker pattern.

    This class manages GPU memory usage, automatically handling memory pressure
    through a three-level circuit breaker system:

    1. WARNING (>60%): Clean cache and trigger warnings
    2. CRITICAL (>80%): Offload non-persistent models to CPU
    3. EMERGENCY (>90%): Emergency cleanup and fallback to CPU

    The manager maintains a pool of registered models and can automatically
    offload them based on persistence flags.

    Example:
        >>> manager = GPUMemoryManager()
        >>> manager.register_model("clip", model, persistent=True)
        >>> if manager.check_pressure() == MemoryPressureLevel.CRITICAL:
        ...     manager.auto_clean(aggressive=True)

    Attributes:
        gpu_id: GPU device ID to manage
        safety_margin: Safety margin as fraction (default 0.15 = 15%)
        safe_limit: Calculated safe memory limit in bytes
    """

    _instance: GPUMemoryManager | None = None
    _lock = threading.Lock()

    def __new__(cls, *_args: Any, **_kwargs: Any) -> GPUMemoryManager:
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        gpu_id: int = 0,
        safety_margin: float = 0.15,
        enable_monitoring: bool = True,
    ) -> None:
        """Initialize the memory manager.

        Args:
            gpu_id: GPU device ID to manage
            safety_margin: Fraction of memory to reserve as safety margin
            enable_monitoring: Whether to enable background monitoring
        """
        if self._initialized:
            return

        self.gpu_id = gpu_id
        self.safety_margin = safety_margin
        self.enable_monitoring = enable_monitoring

        # Initialize CUDA properties
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            self.safe_limit = int(self.total_memory * (1 - safety_margin))
        else:
            self.total_memory = 0
            self.safe_limit = 0
            warnings.warn(
                "CUDA not available. Memory manager operating in CPU-only mode.", stacklevel=2
            )

        # Model pool: name -> (model, persistent, cpu_copy)
        self._model_pool: dict[str, tuple] = {}
        self._lock = threading.RLock()
        self._initialized = True

        # Monitoring
        self._monitor_thread: threading.Thread | None = None
        if enable_monitoring and torch.cuda.is_available():
            self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Start background memory monitoring thread."""

        def monitor():
            while self.enable_monitoring:
                pressure = self.check_pressure()
                if pressure == MemoryPressureLevel.EMERGENCY:
                    self.auto_clean(aggressive=True)
                threading.Event().wait(5.0)  # Check every 5 seconds

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics.

        Returns:
            MemoryStats object with current GPU and RAM information
        """
        if not torch.cuda.is_available():
            return MemoryStats(
                gpu_total=0,
                gpu_used=0,
                gpu_cached=0,
                ram_available=psutil.virtual_memory().available,
            )

        allocated = torch.cuda.memory_allocated(self.gpu_id)
        reserved = torch.cuda.memory_reserved(self.gpu_id)

        return MemoryStats(
            gpu_total=self.total_memory,
            gpu_used=allocated,
            gpu_cached=reserved,
            ram_available=psutil.virtual_memory().available,
        )

    def check_pressure(self) -> MemoryPressureLevel:
        """Check current memory pressure level.

        Returns:
            MemoryPressureLevel indicating current stress level
        """
        if not torch.cuda.is_available():
            return MemoryPressureLevel.NORMAL

        stats = self.get_stats()
        usage = stats.gpu_usage_pct

        if usage > 90:
            return MemoryPressureLevel.EMERGENCY
        elif usage > 80:
            return MemoryPressureLevel.CRITICAL
        elif usage > 60:
            return MemoryPressureLevel.WARNING
        return MemoryPressureLevel.NORMAL

    def auto_clean(self, aggressive: bool = False, emergency: bool = False) -> None:
        """Automatically clean up GPU memory.

        Args:
            aggressive: If True, offload non-persistent models to CPU
            emergency: If True, emergency measures including clearing all caches
        """
        # Level 1: Python garbage collection
        gc.collect()

        # Level 2: PyTorch cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.gpu_id)

        # Level 3: Offload non-persistent models
        if aggressive:
            with self._lock:
                offloaded = []
                for name, (_model, persistent, _) in list(self._model_pool.items()):
                    if not persistent:
                        self._offload_to_cpu(name)
                        offloaded.append(name)
                if offloaded:
                    warnings.warn(
                        f"Memory pressure high. Offloaded models: {offloaded}",
                        ResourceWarning,
                        stacklevel=2,
                    )

        # Level 4: Emergency - clear everything possible
        if emergency:
            # Force garbage collection with all generations
            gc.collect(2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def register_model(
        self,
        name: str,
        model: ModelProtocol,
        persistent: bool = False,
    ) -> None:
        """Register a model for memory management.

        Args:
            name: Unique identifier for the model
            model: The model to manage
            persistent: If True, model won't be auto-offloaded
        """
        with self._lock:
            # Store reference and CPU backup
            cpu_copy = None
            if hasattr(model, "state_dict"):
                cpu_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            self._model_pool[name] = (model, persistent, cpu_copy)

    def unregister_model(self, name: str) -> None:
        """Unregister a model from management.

        Args:
            name: Model identifier to remove
        """
        with self._lock:
            if name in self._model_pool:
                del self._model_pool[name]

    def _offload_to_cpu(self, name: str) -> None:
        """Offload a specific model to CPU.

        Args:
            name: Model identifier to offload
        """
        if name not in self._model_pool:
            return

        model, persistent, cpu_copy = self._model_pool[name]

        # Move to CPU
        try:
            model.cpu()
            # Remove from pool but keep reference to prevent GC
            self._model_pool[name] = (model, persistent, cpu_copy)
        except Exception as e:
            warnings.warn(f"Failed to offload {name}: {e}", stacklevel=2)

    def load_safe(
        self,
        loader_fn: Callable[[], T],
        model_name: str,
        estimated_memory: int | None = None,
        persistent: bool = False,
    ) -> T:
        """Safely load a model with memory checks.

        Attempts to load model to GPU, with automatic fallback to CPU if
        memory is insufficient.

        Args:
            loader_fn: Function that loads and returns the model
            model_name: Name to register the model under
            estimated_memory: Estimated VRAM needed in bytes
            persistent: Whether model should persist in GPU memory

        Returns:
            Loaded model (on GPU if possible, CPU otherwise)
        """
        stats = self.get_stats()
        available = self.safe_limit - stats.gpu_used

        # Check if we need to clean up first
        if estimated_memory and estimated_memory > available * 0.8:
            self.auto_clean(aggressive=True)
            stats = self.get_stats()
            available = self.safe_limit - stats.gpu_used

        # Try GPU loading
        if torch.cuda.is_available() and (estimated_memory is None or estimated_memory < available):
            try:
                model = loader_fn()
                if hasattr(model, "cuda"):
                    model = model.cuda(self.gpu_id)
                self.register_model(model_name, model, persistent=persistent)
                return model
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.auto_clean(aggressive=True)
                    # Try once more
                    try:
                        model = loader_fn()
                        if hasattr(model, "cuda"):
                            model = model.cuda(self.gpu_id)
                        self.register_model(model_name, model, persistent=persistent)
                        return model
                    except RuntimeError:
                        pass
                raise

        # Fallback to CPU
        warnings.warn(
            f"Insufficient GPU memory for {model_name}. Loading on CPU.",
            ResourceWarning,
            stacklevel=2,
        )
        model = loader_fn()
        if hasattr(model, "cpu"):
            model = model.cpu()
        self.register_model(model_name, model, persistent=False)
        return model

    def ensure_available(self, required_bytes: int, aggressive_cleanup: bool = True) -> bool:
        """Ensure required GPU memory is available.

        Args:
            required_bytes: Memory required in bytes
            aggressive_cleanup: Whether to perform aggressive cleanup if needed

        Returns:
            True if memory is available, False otherwise
        """
        stats = self.get_stats()
        available = self.safe_limit - stats.gpu_used

        if required_bytes < available:
            return True

        if aggressive_cleanup:
            self.auto_clean(aggressive=True)
            stats = self.get_stats()
            available = self.safe_limit - stats.gpu_used
            return required_bytes < available

        return False

    def get_model(self, name: str) -> Any | None:
        """Get a registered model.

        Args:
            name: Model identifier

        Returns:
            The model if found, None otherwise
        """
        with self._lock:
            if name in self._model_pool:
                return self._model_pool[name][0]
            return None

    def list_models(self) -> dict[str, dict[str, Any]]:
        """List all registered models and their status.

        Returns:
            Dictionary mapping model names to their info
        """
        with self._lock:
            return {
                name: {
                    "persistent": persistent,
                    "device": (
                        str(next(model.parameters()).device)
                        if hasattr(model, "parameters")
                        else "unknown"
                    ),
                }
                for name, (model, persistent, _) in self._model_pool.items()
            }

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.enable_monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)


# Global singleton instance
_memory_manager: GPUMemoryManager | None = None


def get_memory_manager() -> GPUMemoryManager:
    """Get the global memory manager instance.

    Returns:
        Singleton GPUMemoryManager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = GPUMemoryManager()
    return _memory_manager


@contextmanager
def memory_safe_context(
    fallback_to_cpu: bool = True,
    aggressive_cleanup: bool = True,
):
    """Context manager for memory-safe operations.

    Automatically handles OOM errors and performs cleanup.

    Args:
        fallback_to_cpu: Whether to fallback to CPU on OOM
        aggressive_cleanup: Whether to perform aggressive cleanup

    Example:
        >>> with memory_safe_context():
        ...     result = model(inputs)  # Will handle OOM gracefully
    """
    manager = get_memory_manager()

    try:
        yield manager
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and fallback_to_cpu:
            manager.auto_clean(aggressive=aggressive_cleanup, emergency=True)
            warnings.warn(
                "OOM detected, performed emergency cleanup", ResourceWarning, stacklevel=2
            )
            raise
        raise


def memory_safe(
    fallback_strategy: str = "cpu",
    aggressive_cleanup: bool = True,
) -> Callable:
    """Decorator for memory-safe function execution.

    Args:
        fallback_strategy: "cpu", "skip", "retry", or "raise"
        aggressive_cleanup: Whether to clean aggressively on OOM

    Returns:
        Decorator function

    Example:
        >>> @memory_safe(fallback_strategy="cpu")
        ... def process_batch(images):
        ...     return model(images)
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_memory_manager()

            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise

                # OOM handling
                if fallback_strategy == "cpu":
                    manager.auto_clean(aggressive=aggressive_cleanup, emergency=True)
                    # Move tensors to CPU
                    cpu_args = [arg.cpu() if torch.is_tensor(arg) else arg for arg in args]
                    cpu_kwargs = {
                        k: v.cpu() if torch.is_tensor(v) else v for k, v in kwargs.items()
                    }
                    return func(*cpu_args, **cpu_kwargs)

                elif fallback_strategy == "retry":
                    manager.auto_clean(aggressive=aggressive_cleanup)
                    return func(*args, **kwargs)

                elif fallback_strategy == "skip":
                    manager.auto_clean(aggressive=aggressive_cleanup)
                    warnings.warn(
                        f"Skipping {func.__name__} due to OOM", ResourceWarning, stacklevel=2
                    )
                    return None

                else:  # "raise"
                    raise

        return wrapper

    return decorator
