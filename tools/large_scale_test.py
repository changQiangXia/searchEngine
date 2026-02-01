#!/usr/bin/env python3
"""
Large Scale Dataset Testing Tool for NexusMind

Tests FAISS performance with datasets from 10K to 1M+ vectors.
Validates GPU/CPU switching, memory management, and search accuracy.

Usage:
    python tools/large_scale_test.py --scale 100000
    python tools/large_scale_test.py --scale 1000000 --index-type ivfpq
    python tools/large_scale_test.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus_mind.infrastructure.memory.manager import get_memory_manager
from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend


@dataclass
class TestResult:
    """Test result container."""
    scale: int
    index_type: str
    dim: int
    
    # Build metrics
    build_time_ms: float = 0.0
    build_memory_mb: float = 0.0
    
    # Search metrics
    search_time_ms: float = 0.0
    search_time_p50_ms: float = 0.0
    search_time_p95_ms: float = 0.0
    search_time_p99_ms: float = 0.0
    
    # Accuracy metrics
    recall_at_10: float = 0.0
    recall_nprobe: int = 0  # nprobe used for recall test
    accuracy: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    
    # Device info
    use_gpu: bool = False
    gpu_name: str = ""
    
    # Error info
    error: str = ""
    passed: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticDataGenerator:
    """Generate synthetic embedding data for testing."""
    
    def __init__(self, dim: int = 768, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(
        self,
        n_vectors: int,
        distribution: str = "normal",
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate random embeddings.
        
        Args:
            n_vectors: Number of vectors to generate
            distribution: "normal", "uniform", or "clustered"
            normalize: Whether to L2-normalize vectors
            
        Returns:
            Array of shape (n_vectors, dim)
        """
        if distribution == "normal":
            vectors = self.rng.randn(n_vectors, self.dim).astype('float32')
        elif distribution == "uniform":
            vectors = self.rng.uniform(-1, 1, (n_vectors, self.dim)).astype('float32')
        elif distribution == "clustered":
            # Create clustered data (more realistic)
            n_clusters = max(10, n_vectors // 1000)
            vectors = self._generate_clustered(n_vectors, n_clusters)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        if normalize:
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        
        return vectors
    
    def _generate_clustered(self, n_vectors: int, n_clusters: int) -> np.ndarray:
        """Generate clustered data."""
        vectors_per_cluster = n_vectors // n_clusters
        vectors = []
        
        for i in range(n_clusters):
            # Cluster center
            center = self.rng.randn(self.dim)
            center = center / np.linalg.norm(center)
            
            # Generate vectors around center
            n = vectors_per_cluster if i < n_clusters - 1 else n_vectors - len(vectors)
            cluster_vectors = self.rng.randn(n, self.dim) * 0.1 + center
            vectors.append(cluster_vectors)
        
        return np.vstack(vectors).astype('float32')
    
    def generate_queries(
        self,
        n_queries: int,
        source_vectors: np.ndarray | None = None,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """Generate query vectors.
        
        If source_vectors is provided, queries are derived from them
        (for accurate ground truth calculation).
        """
        if source_vectors is not None:
            # Sample from source and add noise
            indices = self.rng.choice(len(source_vectors), n_queries, replace=False)
            queries = source_vectors[indices].copy()
            queries += self.rng.randn(*queries.shape) * noise_level
        else:
            queries = self.rng.randn(n_queries, self.dim).astype('float32')
        
        # Normalize
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
        return queries


class LargeScaleTester:
    """Large scale dataset tester."""
    
    def __init__(self, dim: int = 768, verbose: bool = True):
        self.dim = dim
        self.verbose = verbose
        self.data_gen = SyntheticDataGenerator(dim=dim)
        self.memory_manager = get_memory_manager()
    
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
    
    def run_test(
        self,
        n_vectors: int,
        index_type: str = "auto",
        use_gpu: bool = True,
        n_queries: int = 100,
        n_ground_truth: int = 1000,
    ) -> TestResult:
        """Run a single scale test.
        
        Args:
            n_vectors: Number of vectors in dataset
            index_type: Index type (auto, flat, ivf, ivfpq)
            use_gpu: Whether to use GPU
            n_queries: Number of queries for timing
            n_ground_truth: Number of queries for accuracy measurement
            
        Returns:
            TestResult with all metrics
        """
        result = TestResult(
            scale=n_vectors,
            index_type=index_type,
            dim=self.dim,
            use_gpu=use_gpu,
        )
        
        try:
            # Get GPU info
            try:
                import torch
                if torch.cuda.is_available():
                    result.gpu_name = torch.cuda.get_device_name(0)
            except:
                pass
            
            self._log(f"\n{'='*60}")
            self._log(f"Testing scale: {n_vectors:,} vectors ({self.dim}D)")
            self._log(f"Index type: {index_type}, GPU: {use_gpu}")
            self._log(f"{'='*60}")
            
            # Generate data
            self._log("\n1. Generating synthetic data...")
            start_time = time.perf_counter()
            vectors = self.data_gen.generate(n_vectors, distribution="clustered")
            gen_time = (time.perf_counter() - start_time) * 1000
            self._log(f"   Generated in {gen_time:.0f}ms")
            
            # Record initial memory
            initial_mem = self.memory_manager.get_stats().gpu_used / (1024**2)
            
            # Build index
            self._log("\n2. Building index...")
            backend = FAISSBackend(dim=self.dim, index_type=index_type, use_gpu=use_gpu)
            
            build_start = time.perf_counter()
            backend.build(vectors)
            build_time = (time.perf_counter() - build_start) * 1000
            result.build_time_ms = build_time
            
            # Record build memory
            build_mem = self.memory_manager.get_stats().gpu_used / (1024**2)
            result.build_memory_mb = build_mem - initial_mem
            
            device = "GPU" if (use_gpu and hasattr(backend.index, 'getDevice')) else "CPU"
            self._log(f"   Built in {build_time:.0f}ms on {device}")
            self._log(f"   Memory delta: {result.build_memory_mb:.1f}MB")
            
            # Generate queries for timing
            self._log("\n3. Testing search performance...")
            queries = self.data_gen.generate_queries(n_queries, vectors, noise_level=0.1)
            
            # Warmup
            for i in range(min(10, n_queries)):
                backend.search(queries[i], top_k=10)
            
            # Timed searches
            search_times = []
            for i in range(n_queries):
                start = time.perf_counter()
                backend.search(queries[i], top_k=10)
                elapsed = (time.perf_counter() - start) * 1000
                search_times.append(elapsed)
            
            search_times = np.array(search_times)
            result.search_time_p50_ms = float(np.percentile(search_times, 50))
            result.search_time_p95_ms = float(np.percentile(search_times, 95))
            result.search_time_p99_ms = float(np.percentile(search_times, 99))
            result.search_time_ms = float(np.mean(search_times))
            
            self._log(f"   P50 latency: {result.search_time_p50_ms:.2f}ms")
            self._log(f"   P95 latency: {result.search_time_p95_ms:.2f}ms")
            self._log(f"   P99 latency: {result.search_time_p99_ms:.2f}ms")
            
            # Test accuracy
            self._log("\n4. Testing accuracy...")
            
            # For IVF index, test with recommended nprobe
            if index_type in ["ivf", "ivfpq"] and hasattr(backend.index, 'nlist'):
                recommended_nprobe = backend.recommend_nprobe(0.95)
                result.recall_nprobe = recommended_nprobe
                self._log(f"   Testing with recommended nprobe={recommended_nprobe} (target 95% recall)")
                result.recall_at_10 = self._test_accuracy(backend, vectors, n_ground_truth, nprobe=recommended_nprobe)
                self._log(f"   Recall@10: {result.recall_at_10:.3f} (nprobe={recommended_nprobe})")
                
                # Also test with nprobe=1 for comparison
                recall_nprobe1 = self._test_accuracy(backend, vectors, min(100, n_ground_truth), nprobe=1)
                self._log(f"   Recall@10 (nprobe=1): {recall_nprobe1:.3f} (baseline)")
            else:
                result.recall_at_10 = self._test_accuracy(backend, vectors, n_ground_truth)
                self._log(f"   Recall@10: {result.recall_at_10:.3f}")
            
            # Final memory
            result.final_memory_mb = self.memory_manager.get_stats().gpu_used / (1024**2)
            result.peak_memory_mb = result.build_memory_mb
            
            result.passed = True
            self._log(f"\n✅ Test passed!")
            
        except Exception as e:
            result.error = str(e)
            result.passed = False
            self._log(f"\n❌ Test failed: {e}")
            import traceback
            self._log(traceback.format_exc())
        
        return result
    
    def _test_accuracy(
        self,
        backend: FAISSBackend,
        vectors: np.ndarray,
        n_queries: int,
        nprobe: int | None = None,
    ) -> float:
        """Test search accuracy using ground truth from exact search.
        
        Args:
            backend: Backend to test
            vectors: Vector dataset
            n_queries: Number of queries for testing
            nprobe: nprobe value for IVF index (if applicable)
        """
        # For large datasets, use exact search on a subset
        if len(vectors) > 50000:
            # Sample subset for ground truth
            indices = np.random.choice(len(vectors), 50000, replace=False)
            subset = vectors[indices]
        else:
            subset = vectors
            indices = np.arange(len(vectors))
        
        # Build exact index on subset
        exact_index = FAISSBackend(dim=self.dim, index_type="flat", use_gpu=False)
        exact_index.build(subset)
        
        # Generate queries from subset
        query_indices = np.random.choice(len(subset), min(n_queries, len(subset)), replace=False)
        queries = subset[query_indices]
        
        # Set nprobe if provided
        if nprobe is not None and hasattr(backend.index, 'nprobe'):
            backend.index.nprobe = nprobe
        
        # Search with both indexes
        recalls = []
        for query, true_idx in zip(queries, query_indices):
            # Ground truth (exact search)
            _, exact_ids = exact_index.search(query, top_k=10)
            exact_set = set(exact_ids[0])
            
            # Test search
            _, test_ids = backend.search(query, top_k=10)
            test_set = set(test_ids[0])
            
            # Calculate recall
            overlap = len(exact_set & test_set)
            recalls.append(overlap / 10)
        
        return float(np.mean(recalls))
    
    def run_scale_tests(
        self,
        scales: list[int],
        index_types: list[str] | None = None,
    ) -> list[TestResult]:
        """Run tests at multiple scales.
        
        Args:
            scales: List of vector counts to test
            index_types: List of index types to test (default: auto)
            
        Returns:
            List of TestResult
        """
        if index_types is None:
            index_types = ["auto"]
        
        results = []
        for scale in scales:
            for idx_type in index_types:
                # Auto-select GPU/CPU based on scale
                use_gpu = scale <= 500000  # Use CPU for very large scales
                
                result = self.run_test(
                    n_vectors=scale,
                    index_type=idx_type,
                    use_gpu=use_gpu,
                )
                results.append(result)
        
        return results


def print_summary(results: list[TestResult]) -> None:
    """Print test summary table."""
    print("\n" + "="*80)
    print("LARGE SCALE TEST SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Scale':>12} {'Index':>8} {'Device':>6} {'Build(s)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Recall':>8} {'Status':>8}")
    print("-"*80)
    
    # Rows
    for r in results:
        device = "GPU" if r.use_gpu else "CPU"
        build_sec = r.build_time_ms / 1000
        status = "✅ PASS" if r.passed else "❌ FAIL"
        recall = f"{r.recall_at_10:.3f}" if r.recall_at_10 > 0 else "N/A"
        
        print(f"{r.scale:>12,} {r.index_type:>8} {device:>6} {build_sec:>10.2f} {r.search_time_p50_ms:>10.2f} {r.search_time_p95_ms:>10.2f} {recall:>8} {status:>8}")
    
    print("="*80)
    
    # Count passed
    passed = sum(1 for r in results if r.passed)
    print(f"\nTotal: {passed}/{len(results)} tests passed")


def save_results(results: list[TestResult], output_path: str) -> None:
    """Save results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Large Scale FAISS Testing")
    parser.add_argument("--scale", type=int, help="Number of vectors to test")
    parser.add_argument("--index-type", default="auto", help="Index type (auto/flat/ivf/ivfpq)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--all", action="store_true", help="Run all standard tests")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    
    args = parser.parse_args()
    
    # Determine scales to test
    if args.all:
        scales = [10000, 50000, 100000, 500000, 1000000]
        index_types = ["flat", "ivf", "ivfpq"]
    elif args.scale:
        scales = [args.scale]
        index_types = [args.index_type]
    else:
        # Default test
        scales = [10000, 100000]
        index_types = ["auto"]
    
    use_gpu = not args.cpu and args.gpu
    
    # Run tests
    tester = LargeScaleTester(dim=args.dim, verbose=not args.quiet)
    results = tester.run_scale_tests(scales, index_types)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        save_results(results, args.output)
    
    # Exit code
    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
