#!/usr/bin/env python
"""Benchmark Tool - Measure NexusMind performance.

Provides comprehensive benchmarking for:
- Indexing speed
- Search latency
- Memory usage
- Throughput at different scales
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus_mind.core.engine import NexusEngine


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    test_name: str
    duration_seconds: float
    items_processed: int
    throughput_per_second: float
    memory_peak_mb: float
    gpu_memory_peak_mb: float
    details: Dict[str, Any]


def benchmark_indexing(
    engine: NexusEngine,
    image_paths: List[Path],
    batch_sizes: List[int] = [16, 32, 64],
) -> List[BenchmarkResult]:
    """Benchmark indexing performance.
    
    Args:
        engine: NexusEngine instance
        image_paths: List of image paths
        batch_sizes: Batch sizes to test
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing indexing with batch_size={batch_size}...")
        
        import torch
        torch.cuda.empty_cache()
        
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        start = time.time()
        
        try:
            stats = engine.index_images(image_paths, batch_size=batch_size)
            
            duration = time.time() - start
            mem_after = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            result = BenchmarkResult(
                test_name=f"indexing_batch_{batch_size}",
                duration_seconds=duration,
                items_processed=stats["count"],
                throughput_per_second=stats["vectors_per_second"],
                memory_peak_mb=mem_after - mem_before,
                gpu_memory_peak_mb=mem_after,
                details={
                    "batch_size": batch_size,
                    "index_type": stats["index_type"],
                    "on_gpu": stats["on_gpu"],
                },
            )
            results.append(result)
            
            print(f"✅ Throughput: {result.throughput_per_second:.1f} images/s")
            print(f"   Duration: {duration:.1f}s")
            print(f"   GPU Memory: {mem_after:.0f}MB")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return results


def benchmark_search(
    engine: NexusEngine,
    queries: List[str],
    top_k_values: List[int] = [10, 50, 100],
) -> List[BenchmarkResult]:
    """Benchmark search performance.
    
    Args:
        engine: NexusEngine instance
        queries: List of test queries
        top_k_values: top_k values to test
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for top_k in top_k_values:
        print(f"\n🔍 Testing search with top_k={top_k}...")
        
        latencies = []
        
        # Warmup
        for _ in range(3):
            _ = engine.search(queries[0], top_k=top_k)
        
        # Benchmark
        for query in tqdm(queries, desc="Search"):
            start = time.time()
            _ = engine.search(query, top_k=top_k)
            latency = time.time() - start
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        result = BenchmarkResult(
            test_name=f"search_topk_{top_k}",
            duration_seconds=sum(latencies),
            items_processed=len(queries),
            throughput_per_second=len(queries) / sum(latencies),
            memory_peak_mb=0,
            gpu_memory_peak_mb=0,
            details={
                "top_k": top_k,
                "avg_latency_ms": avg_latency * 1000,
                "p95_latency_ms": p95_latency * 1000,
                "p99_latency_ms": p99_latency * 1000,
            },
        )
        results.append(result)
        
        print(f"✅ Avg Latency: {avg_latency*1000:.1f}ms")
        print(f"   P95 Latency: {p95_latency*1000:.1f}ms")
    
    return results


def benchmark_scalability(
    engine: NexusEngine,
    image_dir: Path,
    dataset_sizes: List[int] = [100, 500, 1000],
) -> List[BenchmarkResult]:
    """Benchmark scalability with different dataset sizes.
    
    Args:
        engine: NexusEngine instance
        image_dir: Directory with test images
        dataset_sizes: Dataset sizes to test
        
    Returns:
        List of benchmark results
    """
    results = []
    
    # Get all images
    all_images = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))
    
    for size in dataset_sizes:
        if size > len(all_images):
            print(f"⚠️  Skipping size {size} (only {len(all_images)} images available)")
            continue
        
        print(f"\n📈 Testing scalability with {size} images...")
        
        test_images = all_images[:size]
        
        # Benchmark indexing
        start = time.time()
        
        try:
            stats = engine.index_images(test_images)
            duration = time.time() - start
            
            # Benchmark search
            search_start = time.time()
            for _ in range(10):
                _ = engine.search("test query", top_k=10)
            search_duration = time.time() - search_start
            
            result = BenchmarkResult(
                test_name=f"scalability_{size}_images",
                duration_seconds=duration,
                items_processed=size,
                throughput_per_second=size / duration,
                memory_peak_mb=0,
                gpu_memory_peak_mb=0,
                details={
                    "dataset_size": size,
                    "index_time": duration,
                    "avg_search_time": search_duration / 10,
                    "index_type": stats["index_type"],
                },
            )
            results.append(result)
            
            print(f"✅ Index: {duration:.1f}s ({size/duration:.1f} img/s)")
            print(f"   Search: {search_duration/10*1000:.1f}ms avg")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return results


def run_full_benchmark(
    image_dir: Path,
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full benchmark suite.
    
    Args:
        image_dir: Directory with test images
        output_file: Optional output file for results
        
    Returns:
        Benchmark results dictionary
    """
    print("=" * 60)
    print("🚀 NexusMind Performance Benchmark")
    print("=" * 60)
    
    # Initialize engine
    engine = NexusEngine()
    
    # Collect all results
    all_results = []
    
    # Get test images
    test_images = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))
    test_images = test_images[:1000]  # Limit to 1000 for benchmark
    
    if len(test_images) < 10:
        print("❌ Need at least 10 test images")
        return {}
    
    print(f"\n📁 Found {len(test_images)} test images")
    
    # Run benchmarks
    print("\n" + "=" * 60)
    print("1️⃣  Indexing Benchmark")
    print("=" * 60)
    indexing_results = benchmark_indexing(engine, test_images[:100])
    all_results.extend(indexing_results)
    
    print("\n" + "=" * 60)
    print("2️⃣  Search Benchmark")
    print("=" * 60)
    test_queries = [
        "a red car",
        "sunset beach",
        "mountain landscape",
        "city skyline",
        "portrait photo",
    ]
    search_results = benchmark_search(engine, test_queries)
    all_results.extend(search_results)
    
    print("\n" + "=" * 60)
    print("3️⃣  Scalability Benchmark")
    print("=" * 60)
    scalability_results = benchmark_scalability(engine, image_dir)
    all_results.extend(scalability_results)
    
    # Compile final report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "results": [asdict(r) for r in all_results],
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Benchmark Summary")
    print("=" * 60)
    
    for result in all_results:
        print(f"\n{result.test_name}:")
        print(f"  Throughput: {result.throughput_per_second:.1f} items/s")
        print(f"  Duration: {result.duration_seconds:.2f}s")
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NexusMind Benchmark")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory with test images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="benchmark_results.json",
        help="Output file for results",
    )
    
    args = parser.parse_args()
    
    import torch
    
    run_full_benchmark(args.image_dir, args.output)


if __name__ == "__main__":
    main()