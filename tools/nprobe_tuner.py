#!/usr/bin/env python3
"""
NProbe Tuner for IVF Index

Automatically finds optimal nprobe value balancing recall and speed.

Usage:
    python tools/nprobe_tuner.py --index-path ./my_index
    python tools/nprobe_tuner.py --scale 50000 --target-recall 0.95
    python tools/nprobe_tuner.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend


@dataclass
class NProbeResult:
    """Result of nprobe tuning."""
    nprobe: int
    recall: float
    p50_latency_ms: float
    p95_latency_ms: float
    speed_vs_baseline: float  # Compared to nprobe=1


class NProbeTuner:
    """Tuner for finding optimal nprobe value."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
    
    def evaluate_nprobe(
        self,
        backend: FAISSBackend,
        queries: np.ndarray,
        ground_truth: list[set[int]],
        nprobe: int,
        top_k: int = 10,
    ) -> NProbeResult:
        """Evaluate a specific nprobe value.
        
        Args:
            backend: FAISS backend with IVF index
            queries: Query vectors (N, D)
            ground_truth: List of sets containing true nearest neighbor indices
            nprobe: nprobe value to test
            top_k: Number of results to retrieve
            
        Returns:
            NProbeResult with metrics
        """
        # Set nprobe
        if hasattr(backend.index, 'nprobe'):
            backend.index.nprobe = nprobe
        
        # Measure latency
        latencies = []
        for query in queries:
            start = time.perf_counter()
            backend.search(query, top_k=top_k, nprobe=nprobe)
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        
        # Measure recall
        recalls = []
        for i, query in enumerate(queries):
            scores, indices = backend.search(query, top_k=top_k, nprobe=nprobe)
            retrieved = set(indices[0])
            true_set = ground_truth[i]
            
            # Calculate recall@k
            overlap = len(true_set & retrieved)
            recalls.append(overlap / len(true_set))
        
        avg_recall = float(np.mean(recalls))
        
        return NProbeResult(
            nprobe=nprobe,
            recall=avg_recall,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            speed_vs_baseline=1.0,  # Will be updated later
        )
    
    def tune(
        self,
        backend: FAISSBackend,
        test_vectors: np.ndarray,
        n_queries: int = 100,
        top_k: int = 10,
        target_recall: float = 0.95,
        nprobe_candidates: list[int] | None = None,
    ) -> dict[str, Any]:
        """Find optimal nprobe for target recall.
        
        Args:
            backend: FAISS backend with built index
            test_vectors: Vectors to use for testing
            n_queries: Number of queries for evaluation
            top_k: Number of results to retrieve
            target_recall: Target recall value (0-1)
            nprobe_candidates: List of nprobe values to try
            
        Returns:
            Dictionary with optimal nprobe and evaluation results
        """
        if nprobe_candidates is None:
            # Default candidates: geometric progression
            nprobe_candidates = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Filter candidates based on index nlist
        if hasattr(backend.index, 'nlist'):
            nlist = backend.index.nlist
            nprobe_candidates = [n for n in nprobe_candidates if n <= nlist]
            self._log(f"Index has {nlist} clusters, testing nprobe: {nprobe_candidates}")
        
        # Generate queries from test vectors
        np.random.seed(42)
        query_indices = np.random.choice(
            len(test_vectors), 
            min(n_queries, len(test_vectors)), 
            replace=False
        )
        queries = test_vectors[query_indices]
        
        # Build ground truth with exact search (flat index)
        self._log("\nBuilding ground truth with exact search...")
        exact_backend = FAISSBackend(dim=backend.dim, index_type="flat", use_gpu=False)
        exact_backend.build(test_vectors)
        
        ground_truth = []
        for query in queries:
            _, indices = exact_backend.search(query, top_k=top_k)
            ground_truth.append(set(indices[0]))
        
        self._log(f"Ground truth built for {len(queries)} queries")
        
        # Test each nprobe value
        self._log("\nTesting nprobe values...")
        results = []
        
        for nprobe in nprobe_candidates:
            result = self.evaluate_nprobe(backend, queries, ground_truth, nprobe, top_k)
            results.append(result)
            
            status = "✅" if result.recall >= target_recall else "❌"
            self._log(
                f"  nprobe={nprobe:3d}: "
                f"recall={result.recall:.3f}, "
                f"P50={result.p50_latency_ms:.2f}ms "
                f"{status}"
            )
        
        # Calculate speed relative to nprobe=1
        baseline_p50 = next(r.p50_latency_ms for r in results if r.nprobe == 1)
        for r in results:
            r.speed_vs_baseline = r.p50_latency_ms / baseline_p50
        
        # Find optimal nprobe
        # Priority: meet target recall with minimum nprobe
        valid_results = [r for r in results if r.recall >= target_recall]
        
        if valid_results:
            optimal = min(valid_results, key=lambda r: r.nprobe)
            recommendation = (
                f"Optimal nprobe={optimal.nprobe} achieves "
                f"{optimal.recall:.3f} recall at {optimal.p50_latency_ms:.2f}ms "
                f"({optimal.speed_vs_baseline:.1f}x slower than nprobe=1)"
            )
        else:
            # If no nprobe meets target, choose the one with highest recall
            optimal = max(results, key=lambda r: r.recall)
            recommendation = (
                f"⚠️  No nprobe meets target recall {target_recall}. "
                f"Best: nprobe={optimal.nprobe} with {optimal.recall:.3f} recall"
            )
        
        return {
            "optimal_nprobe": optimal.nprobe,
            "optimal_recall": optimal.recall,
            "optimal_latency_ms": optimal.p50_latency_ms,
            "target_recall": target_recall,
            "recommendation": recommendation,
            "all_results": [asdict(r) for r in results],
        }
    
    def recommend_nprobe(
        self,
        nlist: int,
        target_recall: float = 0.95,
        index_size: int | None = None,
    ) -> int:
        """Get recommended nprobe based on heuristics.
        
        This is a quick estimate without actual testing.
        
        Args:
            nlist: Number of clusters in IVF index
            target_recall: Desired recall
            index_size: Number of vectors in index
            
        Returns:
            Recommended nprobe value
        """
        # Base recommendation: search ~5-10% of clusters
        base = max(1, int(nlist * 0.05))
        
        # Adjust for target recall
        if target_recall >= 0.99:
            multiplier = 4
        elif target_recall >= 0.95:
            multiplier = 2
        elif target_recall >= 0.90:
            multiplier = 1
        else:
            multiplier = 0.5
        
        recommended = int(base * multiplier)
        
        # Clamp to reasonable range
        recommended = max(1, min(recommended, nlist, 128))
        
        return recommended


def print_tuning_report(result: dict[str, Any]) -> None:
    """Print formatted tuning report."""
    print("\n" + "="*70)
    print("NPROBE TUNING REPORT")
    print("="*70)
    print(f"\nTarget Recall: {result['target_recall']:.0%}")
    print(f"Optimal nprobe: {result['optimal_nprobe']}")
    print(f"Achieved Recall: {result['optimal_recall']:.3f}")
    print(f"Latency (P50): {result['optimal_latency_ms']:.2f}ms")
    print(f"\n📌 Recommendation:")
    print(f"   {result['recommendation']}")
    
    print("\n" + "-"*70)
    print(f"{'nprobe':>8} {'Recall':>10} {'P50(ms)':>12} {'Speed':>10}")
    print("-"*70)
    
    for r in result['all_results']:
        marker = " <--" if r['nprobe'] == result['optimal_nprobe'] else ""
        print(
            f"{r['nprobe']:>8} {r['recall']:>10.3f} "
            f"{r['p50_latency_ms']:>12.2f} {r['speed_vs_baseline']:>9.1f}x{marker}"
        )
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="NProbe Tuner for IVF Index")
    parser.add_argument("--index-path", help="Path to saved FAISS index")
    parser.add_argument("--scale", type=int, default=50000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--target-recall", type=float, default=0.95, help="Target recall")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    
    args = parser.parse_args()
    
    tuner = NProbeTuner(verbose=True)
    
    if args.benchmark:
        # Run comprehensive benchmark
        print("Running nprobe benchmark...")
        
        # Generate test data
        np.random.seed(42)
        vectors = np.random.randn(args.scale, args.dim).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Build IVF index
        print(f"\nBuilding IVF index for {args.scale} vectors...")
        backend = FAISSBackend(dim=args.dim, index_type="ivf", use_gpu=True)
        backend.build(vectors)
        
        # Tune
        result = tuner.tune(
            backend,
            vectors,
            target_recall=args.target_recall,
        )
        
        print_tuning_report(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    elif args.index_path:
        # Load existing index
        print(f"Loading index from {args.index_path}...")
        backend = FAISSBackend(dim=args.dim)
        backend.load(args.index_path)
        
        # Need test vectors - generate some
        print("Generating test vectors...")
        np.random.seed(42)
        vectors = np.random.randn(1000, args.dim).astype('float32')
        
        result = tuner.tune(backend, vectors, target_recall=args.target_recall)
        print_tuning_report(result)
    
    else:
        # Quick recommendation
        print("Quick nprobe recommendation:")
        print(f"  For nlist=100, target_recall=0.95: nprobe={tuner.recommend_nprobe(100, 0.95)}")
        print(f"  For nlist=1000, target_recall=0.95: nprobe={tuner.recommend_nprobe(1000, 0.95)}")
        print(f"  For nlist=4096, target_recall=0.95: nprobe={tuner.recommend_nprobe(4096, 0.95)}")
        print("\nRun with --benchmark for detailed tuning.")


if __name__ == "__main__":
    main()
