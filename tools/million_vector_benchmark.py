#!/usr/bin/env python3
"""
Streamlined Million Vector Benchmark

Optimized for RTX 3080ti with progress reporting.
Expected time: 15-20 minutes for 1M vectors.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import faiss
from nexus_mind.infrastructure.memory.manager import get_memory_manager


def print_progress(current, total, start_time, prefix=""):
    """Print progress bar."""
    pct = current / total * 100
    elapsed = time.time() - start_time
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = f"ETA {eta/60:.1f}min"
    else:
        eta_str = "ETA --"
    bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
    print(f"\r{prefix} [{bar}] {pct:.1f}% {eta_str}", end="", flush=True)


def run_benchmark():
    """Run million vector benchmark."""
    N = 1_000_000
    D = 768
    
    print("="*70)
    print("MILLION VECTOR BENCHMARK (1M vectors, 768D)")
    print("="*70)
    print(f"\nTarget: {N:,} vectors")
    print(f"Estimated time: 15-20 minutes")
    print(f"Press Ctrl+C to cancel\n")
    
    memory_manager = get_memory_manager()
    initial_mem = memory_manager.get_stats().gpu_used / 1e9
    print(f"Initial GPU memory: {initial_mem:.2f} GB")
    
    # Step 1: Generate data in batches
    print("\n[Step 1/3] Generating 1M vectors...")
    start_total = time.time()
    start = time.time()
    
    rng = np.random.RandomState(42)
    all_vectors = []
    batch_size = 50_000
    n_batches = N // batch_size
    
    for i in range(n_batches):
        batch = rng.randn(batch_size, D).astype('float32')
        batch = batch / (np.linalg.norm(batch, axis=1, keepdims=True) + 1e-10)
        all_vectors.append(batch)
        print_progress(i + 1, n_batches, start, "Generating")
    
    print()  # New line after progress
    vectors = np.vstack(all_vectors)
    gen_time = time.time() - start
    print(f"✓ Generated in {gen_time:.1f}s, size: {vectors.nbytes/1e9:.2f}GB")
    
    # Step 2: Build IVFPQ index
    print("\n[Step 2/3] Building IVFPQ index...")
    print("  - Training quantizer (may take 5-10 minutes)...")
    start = time.time()
    
    nlist = min(4096, max(100, int(4 * np.sqrt(N))))
    print(f"  - Using nlist={nlist}, m=32, nbits=8")
    
    quantizer = faiss.IndexFlatIP(D)
    cpu_index = faiss.IndexIVFPQ(quantizer, D, nlist, 32, 8)
    
    # Train
    train_start = time.time()
    cpu_index.train(vectors)
    train_time = time.time() - train_start
    print(f"  ✓ Trained in {train_time:.1f}s")
    
    # Add vectors in chunks
    print("  - Adding vectors...")
    add_start = time.time()
    chunk_size = 100_000
    for i in range(0, N, chunk_size):
        chunk = vectors[i:i+chunk_size]
        cpu_index.add(chunk)
        print_progress(min(i + chunk_size, N), N, add_start, "  Adding")
    print()
    add_time = time.time() - add_start
    
    build_time = time.time() - start
    print(f"✓ Index built in {build_time:.1f}s total")
    print(f"  Total throughput: {N/build_time:.0f} vec/s")
    
    # Move to GPU
    print("\n[Step 3/3] Moving to GPU...")
    start = time.time()
    res = faiss.StandardGpuResources()
    res.setTempMemory(1 * 1024 * 1024 * 1024)  # 1GB temp
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_time = time.time() - start
    print(f"✓ GPU transfer in {gpu_time:.1f}s")
    
    post_mem = memory_manager.get_stats().gpu_used / 1e9
    print(f"\nGPU memory: {post_mem:.2f} GB (delta: {post_mem-initial_mem:.2f} GB)")
    
    # Test search
    print("\n[Bonus] Testing search performance...")
    gpu_index.nprobe = 100
    
    # Generate test queries
    test_queries = 1000
    query_indices = np.random.choice(N, test_queries, replace=False)
    queries = vectors[query_indices]
    
    # Warmup
    print("  Warming up...")
    for i in range(50):
        gpu_index.search(queries[i:i+1], k=10)
    
    # Test
    print(f"  Testing {test_queries} searches...")
    latencies = []
    for i in range(test_queries):
        t0 = time.perf_counter()
        gpu_index.search(queries[i:i+1], k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
    
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"\n  Search Performance:")
    print(f"    P50: {p50:.2f}ms")
    print(f"    P95: {p95:.2f}ms")
    print(f"    P99: {p99:.2f}ms")
    print(f"    QPS: ~{1000/p50:.0f}")
    
    # Accuracy check (on subset)
    print("\n[Bonus] Checking accuracy...")
    print("  Building exact index on 10K subset...")
    subset = vectors[:10000]
    exact = faiss.IndexFlatIP(D)
    exact.add(subset)
    
    test_acc = 100
    recalls = []
    for i in range(test_acc):
        q = subset[i:i+1]
        _, exact_ids = exact.search(q, k=10)
        _, ivf_ids = gpu_index.search(q, k=10)
        overlap = len(set(exact_ids[0]) & set(ivf_ids[0]))
        recalls.append(overlap / 10)
    
    avg_recall = np.mean(recalls)
    print(f"  Recall@10: {avg_recall:.3f} ({avg_recall*100:.1f}%)")
    
    # Summary
    total_time = time.time() - start_total
    print("\n" + "="*70)
    print("MILLION VECTOR BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  Scale: {N:,} vectors x {D}D")
    print(f"  Index: IVFPQ (nlist={nlist})")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"\nBuild Performance:")
    print(f"  Data generation: {gen_time:.1f}s")
    print(f"  Index training: {train_time:.1f}s")
    print(f"  Vector adding: {add_time:.1f}s")
    print(f"  GPU transfer: {gpu_time:.1f}s")
    print(f"  Throughput: {N/build_time:.0f} vec/s")
    print(f"\nSearch Performance:")
    print(f"  P50 latency: {p50:.2f}ms")
    print(f"  QPS: {1000/p50:.0f}")
    print(f"\nQuality:")
    print(f"  Recall@10: {avg_recall:.1%} (nprobe=100)")
    print(f"\nMemory:")
    print(f"  GPU memory used: {post_mem-initial_mem:.2f} GB")
    print("="*70)
    
    # Save results
    results = {
        "scale": N,
        "dim": D,
        "index_type": "IVFPQ",
        "nlist": nlist,
        "total_time_sec": total_time,
        "build_time_sec": build_time,
        "throughput": N / build_time,
        "search_p50_ms": float(p50),
        "search_p95_ms": float(p95),
        "qps": 1000 / p50,
        "recall_at_10": float(avg_recall),
        "gpu_memory_gb": post_mem - initial_mem,
    }
    
    import json
    with open("million_vector_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: million_vector_results.json")


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
