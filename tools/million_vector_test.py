#!/usr/bin/env python3
"""
Million Vector Stress Test for NexusMind

Tests FAISS with 1 million vectors on RTX 3080ti (12GB VRAM).
Uses IVFPQ for memory efficiency.

Usage:
    python tools/million_vector_test.py
    python tools/million_vector_test.py --scale 1000000 --index-type ivfpq
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus_mind.infrastructure.memory.manager import get_memory_manager
from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend


@dataclass
class MillionVectorResult:
    """Result of million vector test."""
    scale: int
    index_type: str
    dim: int
    
    # Build metrics
    build_time_sec: float = 0.0
    vectors_per_second: float = 0.0
    
    # Search metrics
    search_p50_ms: float = 0.0
    search_p95_ms: float = 0.0
    search_p99_ms: float = 0.0
    
    # Accuracy
    recall_at_10: float = 0.0
    nprobe_used: int = 0
    
    # Memory
    peak_memory_gb: float = 0.0
    final_memory_gb: float = 0.0
    index_size_mb: float = 0.0
    
    # Status
    success: bool = False
    error: str = ""


class ProgressReporter:
    """Report progress for long-running operations."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.start_time = time.time()
        self.last_report = 0
    
    def update(self, current: int, extra_info: str = ""):
        """Update progress."""
        pct = current / self.total * 100
        elapsed = time.time() - self.start_time
        
        # Report every 10% or 30 seconds
        if pct - self.last_report >= 10 or elapsed % 30 < 1:
            self.last_report = (pct // 10) * 10
            
            if current > 0:
                eta = elapsed / current * (self.total - current)
                eta_str = f"ETA: {eta/60:.1f}min"
            else:
                eta_str = "ETA: --"
            
            print(f"\r{self.desc}: {pct:.0f}% ({current:,}/{self.total:,}) [{elapsed:.0f}s] {eta_str} {extra_info}", 
                  end="", flush=True)
    
    def finish(self):
        """Finish progress reporting."""
        elapsed = time.time() - self.start_time
        print(f"\r{self.desc}: 100% ({self.total:,}/{self.total:,}) [Done in {elapsed:.1f}s]")


def generate_vectors_in_batches(
    n_vectors: int,
    dim: int,
    batch_size: int = 10000,
    distribution: str = "clustered",
    seed: int = 42,
) -> np.ndarray:
    """Generate large vector dataset in batches to avoid memory issues."""
    print(f"Generating {n_vectors:,} vectors ({dim}D) in batches of {batch_size:,}...")
    
    rng = np.random.RandomState(seed)
    all_vectors = []
    
    progress = ProgressReporter(n_vectors, "Generating")
    
    n_batches = (n_vectors + batch_size - 1) // batch_size
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_vectors)
        n = end_idx - start_idx
        
        if distribution == "normal":
            batch = rng.randn(n, dim).astype('float32')
        elif distribution == "uniform":
            batch = rng.uniform(-1, 1, (n, dim)).astype('float32')
        elif distribution == "clustered":
            # Simple clustering simulation
            batch = rng.randn(n, dim).astype('float32') * 0.5
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Normalize
        batch = batch / (np.linalg.norm(batch, axis=1, keepdims=True) + 1e-10)
        all_vectors.append(batch)
        
        progress.update(end_idx)
    
    progress.finish()
    return np.vstack(all_vectors)


def run_million_vector_test(
    n_vectors: int = 1_000_000,
    dim: int = 768,
    index_type: str = "ivfpq",
    use_gpu: bool = True,
    n_queries: int = 1000,
) -> MillionVectorResult:
    """Run million vector stress test.
    
    Args:
        n_vectors: Number of vectors (default 1M)
        dim: Vector dimension
        index_type: Index type (ivfpq recommended for 1M+)
        use_gpu: Use GPU if available
        n_queries: Number of queries for testing
        
    Returns:
        MillionVectorResult with all metrics
    """
    result = MillionVectorResult(
        scale=n_vectors,
        index_type=index_type,
        dim=dim,
    )
    
    memory_manager = get_memory_manager()
    
    try:
        print("\n" + "="*70)
        print(f"MILLION VECTOR STRESS TEST")
        print(f"Scale: {n_vectors:,} vectors | Dim: {dim} | Index: {index_type}")
        print("="*70)
        
        # Check initial memory
        initial_stats = memory_manager.get_stats()
        print(f"\nInitial GPU memory: {initial_stats.gpu_used/1e9:.2f}GB / {initial_stats.gpu_total/1e9:.2f}GB")
        
        # Phase 1: Generate data
        print("\n[Phase 1/4] Generating synthetic data...")
        start_time = time.time()
        vectors = generate_vectors_in_batches(
            n_vectors=n_vectors,
            dim=dim,
            batch_size=50000,
            distribution="clustered",
        )
        gen_time = time.time() - start_time
        print(f"Data generation complete in {gen_time:.1f}s")
        print(f"Data size: {vectors.nbytes / 1e9:.2f} GB")
        
        # Phase 2: Build index
        print("\n[Phase 2/4] Building index...")
        gc.collect()  # Clean up before building
        
        backend = FAISSBackend(dim=dim, index_type=index_type, use_gpu=use_gpu)
        
        build_start = time.time()
        backend.build(vectors)
        build_time = time.time() - build_start
        
        result.build_time_sec = build_time
        result.vectors_per_second = n_vectors / build_time
        
        # Memory after build
        post_build_stats = memory_manager.get_stats()
        result.peak_memory_gb = post_build_stats.gpu_used / 1e9
        
        device = "GPU" if (use_gpu and hasattr(backend.index, 'getDevice')) else "CPU"
        print(f"\n✅ Index built in {build_time:.1f}s ({result.vectors_per_second:.0f} vec/s) on {device}")
        print(f"Index now has {backend.ntotal:,} vectors")
        
        if hasattr(backend.index, 'nlist'):
            print(f"IVF nlist: {backend.index.nlist}")
        if hasattr(backend.index, 'pq'):
            print(f"PQ: enabled")
        
        # Estimate index size
        if hasattr(backend.index, 'ntotal'):
            # Rough estimate
            if index_type == "ivfpq":
                # ~32 bytes per vector + overhead
                result.index_size_mb = backend.ntotal * 40 / 1e6
            elif index_type == "ivf":
                # ~4 bytes per dim + overhead
                result.index_size_mb = backend.ntotal * dim * 4 / 1e6 * 1.2
            else:
                result.index_size_mb = backend.ntotal * dim * 4 / 1e6
            print(f"Estimated index size: {result.index_size_mb:.1f} MB")
        
        # Phase 3: Search performance test
        print("\n[Phase 3/4] Testing search performance...")
        
        # Generate queries from existing vectors
        np.random.seed(43)
        query_indices = np.random.choice(n_vectors, min(n_queries, n_vectors), replace=False)
        queries = vectors[query_indices]
        
        # Set appropriate nprobe for IVF
        if hasattr(backend.index, 'nlist'):
            nprobe = min(100, backend.index.nlist)  # Conservative for speed test
            backend.set_nprobe(nprobe)
            result.nprobe_used = nprobe
        
        # Warmup
        print("Warming up...")
        for i in range(min(100, n_queries)):
            backend.search(queries[i], top_k=10)
        
        # Timed searches
        print(f"Running {n_queries} searches...")
        latencies = []
        progress = ProgressReporter(n_queries, "Searching")
        
        for i in range(n_queries):
            start = time.perf_counter()
            backend.search(queries[i], top_k=10)
            latencies.append((time.perf_counter() - start) * 1000)
            
            if i % 100 == 0:
                progress.update(i, f"P50={np.percentile(latencies, 50):.2f}ms")
        
        progress.finish()
        
        latencies = np.array(latencies)
        result.search_p50_ms = float(np.percentile(latencies, 50))
        result.search_p95_ms = float(np.percentile(latencies, 95))
        result.search_p99_ms = float(np.percentile(latencies, 99))
        
        print(f"\nSearch latency:")
        print(f"  P50: {result.search_p50_ms:.2f}ms")
        print(f"  P95: {result.search_p95_ms:.2f}ms")
        print(f"  P99: {result.search_p99_ms:.2f}ms")
        print(f"  QPS: {1000/result.search_p50_ms:.0f} queries/second")
        
        # Phase 4: Accuracy test (on subset)
        print("\n[Phase 4/4] Testing accuracy...")
        print("Building exact index on subset for ground truth...")
        
        subset_size = min(50000, n_vectors)
        subset_indices = np.random.choice(n_vectors, subset_size, replace=False)
        subset = vectors[subset_size]  # Use first 50K for ground truth
        
        exact_backend = FAISSBackend(dim=dim, index_type="flat", use_gpu=False)
        exact_backend.build(vectors[:subset_size])
        
        # Test queries from subset
        test_queries = 100
        test_indices = np.random.choice(subset_size, test_queries, replace=False)
        test_queries_data = vectors[test_indices]
        
        recalls = []
        for query in test_queries_data:
            _, exact_ids = exact_backend.search(query, top_k=10)
            _, test_ids = backend.search(query, top_k=10)
            
            overlap = len(set(exact_ids[0]) & set(test_ids[0]))
            recalls.append(overlap / 10)
        
        result.recall_at_10 = float(np.mean(recalls))
        print(f"Recall@10: {result.recall_at_10:.3f} ({result.recall_at_10*100:.1f}%)")
        
        # Final memory
        final_stats = memory_manager.get_stats()
        result.final_memory_gb = final_stats.gpu_used / 1e9
        
        print(f"\nMemory usage:")
        print(f"  Peak: {result.peak_memory_gb:.2f} GB")
        print(f"  Final: {result.final_memory_gb:.2f} GB")
        
        result.success = True
        print("\n" + "="*70)
        print("✅ MILLION VECTOR TEST PASSED")
        print("="*70)
        
    except Exception as e:
        result.error = str(e)
        result.success = False
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def print_result_table(result: MillionVectorResult):
    """Print formatted result table."""
    print("\n" + "="*70)
    print("TEST RESULT SUMMARY")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Vectors: {result.scale:,}")
    print(f"  Dimension: {result.dim}")
    print(f"  Index type: {result.index_type}")
    
    print(f"\nBuild Performance:")
    print(f"  Time: {result.build_time_sec:.1f}s")
    print(f"  Throughput: {result.vectors_per_second:.0f} vec/s")
    
    print(f"\nSearch Performance:")
    print(f"  P50: {result.search_p50_ms:.2f}ms")
    print(f"  P95: {result.search_p95_ms:.2f}ms")
    print(f"  P99: {result.search_p99_ms:.2f}ms")
    print(f"  QPS: ~{1000/result.search_p50_ms:.0f}")
    
    print(f"\nQuality:")
    print(f"  Recall@10: {result.recall_at_10:.3f}")
    if result.nprobe_used:
        print(f"  nprobe: {result.nprobe_used}")
    
    print(f"\nMemory:")
    print(f"  Peak: {result.peak_memory_gb:.2f} GB")
    print(f"  Index size: {result.index_size_mb:.1f} MB")
    
    status = "✅ PASSED" if result.success else "❌ FAILED"
    print(f"\nStatus: {status}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Million Vector Stress Test")
    parser.add_argument("--scale", type=int, default=1_000_000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--index-type", default="ivfpq", help="Index type")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--queries", type=int, default=1000, help="Number of test queries")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("# NEXUSMIND MILLION VECTOR STRESS TEST")
    print("# This test may take several minutes...")
    print("#"*70)
    
    # Run test
    result = run_million_vector_test(
        n_vectors=args.scale,
        dim=args.dim,
        index_type=args.index_type,
        use_gpu=not args.cpu,
        n_queries=args.queries,
    )
    
    # Print summary
    print_result_table(result)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
