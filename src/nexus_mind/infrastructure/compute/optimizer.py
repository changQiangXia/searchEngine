"""Compute Optimizer - Batch processing and async execution.

Provides optimized batch processing, async execution, and
performance tuning for 3080ti.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        batch_size: Target batch size
        max_batch_size: Maximum batch size
        min_batch_size: Minimum batch size
        dynamic_batching: Adjust batch size based on memory
        prefetch_factor: Number of batches to prefetch
        num_workers: Number of worker threads/processes
        pin_memory: Pin memory for faster GPU transfer
    """

    batch_size: int = 32
    max_batch_size: int = 128
    min_batch_size: int = 4
    dynamic_batching: bool = True
    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True


class DynamicBatcher:
    """Dynamic batch size adjustment based on GPU memory.

    Automatically adjusts batch size to maximize throughput
    while avoiding OOM errors.
    """

    def __init__(
        self,
        config: BatchConfig,
        memory_safety_margin: float = 0.8,
    ):
        """Initialize dynamic batcher.

        Args:
            config: Batch configuration
            memory_safety_margin: Max memory usage (0-1)
        """
        self.config = config
        self.memory_safety_margin = memory_safety_margin
        self.current_batch_size = config.batch_size

        # Performance tracking
        self.batch_times: deque = deque(maxlen=10)
        self.batch_sizes: deque = deque(maxlen=10)

    def get_batch_size(self) -> int:
        """Get current optimal batch size."""
        if not self.config.dynamic_batching or not torch.cuda.is_available():
            return self.current_batch_size

        # Check current GPU memory
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory

        usage = allocated / total

        # Adjust batch size based on memory pressure
        if usage > self.memory_safety_margin:
            # Memory pressure - reduce batch size
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8),
            )
        elif usage < self.memory_safety_margin * 0.5:
            # Low memory usage - can increase
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.1),
            )

        return self.current_batch_size

    def record_batch(self, batch_size: int, duration: float):
        """Record batch performance."""
        self.batch_times.append(duration)
        self.batch_sizes.append(batch_size)

    def get_throughput(self) -> float:
        """Get average throughput (items/second)."""
        if not self.batch_times or not self.batch_sizes:
            return 0.0

        total_items = sum(self.batch_sizes)
        total_time = sum(self.batch_times)

        return total_items / total_time if total_time > 0 else 0.0


class AsyncProcessor:
    """Asynchronous processing with worker pools.

    Provides async execution for I/O bound and CPU bound tasks.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
    ):
        """Initialize async processor.

        Args:
            max_workers: Maximum number of workers
            use_processes: Use process pool (for CPU-bound) instead of thread pool
        """
        self.max_workers = max_workers
        self.use_processes = use_processes

        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_batch(
        self,
        items: list[Any],
        process_fn: Callable[[Any], Any],
        show_progress: bool = True,
    ) -> list[Any]:
        """Process items asynchronously.

        Args:
            items: Items to process
            process_fn: Processing function
            show_progress: Show progress bar

        Returns:
            Processed results
        """
        loop = asyncio.get_event_loop()

        # Create futures
        futures = [loop.run_in_executor(self.executor, process_fn, item) for item in items]

        # Gather results with progress
        if show_progress:
            results = []
            for f in tqdm(asyncio.as_completed(futures), total=len(futures)):
                results.append(await f)
        else:
            results = await asyncio.gather(*futures)

        return results

    async def map_reduce(
        self,
        items: list[Any],
        map_fn: Callable[[Any], Any],
        reduce_fn: Callable[[list[Any]], Any],
        chunk_size: int = 100,
    ) -> Any:
        """Map-reduce pattern for large datasets.

        Args:
            items: Input items
            map_fn: Map function
            reduce_fn: Reduce function
            chunk_size: Items per chunk

        Returns:
            Reduced result
        """
        # Split into chunks
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

        # Map phase
        async def process_chunk(chunk):
            return [map_fn(item) for item in chunk]

        chunk_results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

        # Flatten results
        flat_results = [item for sublist in chunk_results for item in sublist]

        # Reduce phase
        return reduce_fn(flat_results)

    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


class StreamingBatcher:
    """Streaming batch processor for large datasets.

    Processes data in streaming fashion to handle datasets
    larger than memory.
    """

    def __init__(
        self,
        batch_size: int = 32,
        buffer_size: int = 1000,
    ):
        """Initialize streaming batcher.

        Args:
            batch_size: Batch size
            buffer_size: Internal buffer size
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def stream_batches(
        self,
        items: Iterator[Any],
    ) -> Generator[list[Any], None, None]:
        """Stream items in batches.

        Args:
            items: Iterator of items

        Yields:
            Batches of items
        """
        batch = []

        for item in items:
            batch.append(item)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    def parallel_stream_process(
        self,
        items: Iterator[Any],
        process_fn: Callable[[list[Any]], list[Any]],
        num_workers: int = 4,
    ) -> Generator[Any, None, None]:
        """Process streaming data in parallel.

        Args:
            items: Input iterator
            process_fn: Batch processing function
            num_workers: Number of parallel workers

        Yields:
            Processed items
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit initial batches
            futures = {}
            batch_iter = self.stream_batches(items)

            # Prefetch initial batches
            for _ in range(num_workers * 2):
                try:
                    batch = next(batch_iter)
                    future = executor.submit(process_fn, batch)
                    futures[future] = batch
                except StopIteration:
                    break

            # Process as they complete and submit new ones
            while futures:
                for future in list(futures.keys()):
                    if future.done():
                        result = future.result()
                        yield from result
                        del futures[future]

                        # Submit new batch
                        try:
                            batch = next(batch_iter)
                            new_future = executor.submit(process_fn, batch)
                            futures[new_future] = batch
                        except StopIteration:
                            pass

                        break


class PerformanceMonitor:
    """Monitor and log performance metrics.

    Tracks throughput, latency, memory usage, etc.
    """

    def __init__(self):
        """Initialize monitor."""
        self.metrics: dict[str, list[float]] = {
            "throughput": [],
            "latency": [],
            "memory_used": [],
            "gpu_memory": [],
        }
        self.start_times: dict[str, float] = {}

    def start_timer(self, name: str):
        """Start timing an operation."""
        self.start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing and record."""
        if name not in self.start_times:
            return 0.0

        duration = time.time() - self.start_times[name]
        self.metrics["latency"].append(duration)
        del self.start_times[name]

        return duration

    def record_throughput(self, items: int, duration: float):
        """Record throughput."""
        throughput = items / duration if duration > 0 else 0
        self.metrics["throughput"].append(throughput)

    def record_memory(self):
        """Record current memory usage."""
        import psutil

        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        self.metrics["memory_used"].append(mem_mb)

        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.metrics["gpu_memory"].append(gpu_mb)

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        stats = {}

        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1],
                }

        return stats

    def print_report(self):
        """Print performance report."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("Performance Report")
        print("=" * 60)

        for metric, values in stats.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {values['mean']:.2f}")
            print(f"  Std:  {values['std']:.2f}")
            print(f"  Min:  {values['min']:.2f}")
            print(f"  Max:  {values['max']:.2f}")

        print("=" * 60)


def optimize_for_3080ti() -> BatchConfig:
    """Get optimized batch config for RTX 3080ti (12GB).

    Returns:
        Optimized batch configuration
    """
    return BatchConfig(
        batch_size=64,  # Good balance for 12GB
        max_batch_size=128,
        min_batch_size=16,
        dynamic_batching=True,
        prefetch_factor=2,
        num_workers=4,
        pin_memory=True,
    )


def optimize_for_4090() -> BatchConfig:
    """Get optimized batch config for RTX 4090 (24GB).

    Returns:
        Optimized batch configuration
    """
    return BatchConfig(
        batch_size=128,
        max_batch_size=256,
        min_batch_size=32,
        dynamic_batching=True,
        prefetch_factor=4,
        num_workers=8,
        pin_memory=True,
    )
