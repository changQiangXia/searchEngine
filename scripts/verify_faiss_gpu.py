#!/usr/bin/env python3
"""
FAISS GPU Verification Script for NexusMind

This script verifies that FAISS GPU is properly installed and working.
Run this after installing faiss-gpu to confirm GPU acceleration is available.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def check_faiss_installation() -> dict:
    """Check FAISS installation details."""
    results = {
        "installed": False,
        "version": None,
        "gpu_available": False,
        "num_gpus": 0,
        "errors": [],
    }
    
    try:
        import faiss
        results["installed"] = True
        results["version"] = faiss.__version__
        
        # Check GPU support
        if hasattr(faiss, 'StandardGpuResources'):
            results["gpu_available"] = True
            results["num_gpus"] = faiss.get_num_gpus()
        
    except ImportError as e:
        results["errors"].append(f"FAISS not installed: {e}")
    except Exception as e:
        results["errors"].append(f"Unexpected error: {e}")
    
    return results


def test_gpu_functionality(dim: int = 768, n_vectors: int = 10000) -> dict:
    """Test actual GPU functionality."""
    results = {
        "cpu_test": {"passed": False, "time_ms": 0},
        "gpu_test": {"passed": False, "time_ms": 0},
        "speedup": 0.0,
    }
    
    try:
        import faiss
        
        # Generate test data
        print(f"Generating {n_vectors} random vectors ({dim}D)...")
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dim).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        query = np.random.randn(dim).astype('float32')
        query = query / np.linalg.norm(query)
        
        # CPU test
        print("Testing CPU index...")
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(vectors)
        
        start = time.perf_counter()
        for _ in range(10):
            cpu_index.search(query.reshape(1, -1), k=10)
        cpu_time = (time.perf_counter() - start) * 100
        results["cpu_test"] = {"passed": True, "time_ms": cpu_time}
        print(f"  CPU: {cpu_time:.2f}ms (10 searches)")
        
        # GPU test
        if faiss.get_num_gpus() > 0:
            print("Testing GPU index...")
            res = faiss.StandardGpuResources()
            res.setTempMemory(512 * 1024 * 1024)  # 512MB temp memory
            
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            
            # Warmup
            gpu_index.search(query.reshape(1, -1), k=10)
            
            start = time.perf_counter()
            for _ in range(10):
                gpu_index.search(query.reshape(1, -1), k=10)
            gpu_time = (time.perf_counter() - start) * 100
            results["gpu_test"] = {"passed": True, "time_ms": gpu_time}
            results["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  GPU: {gpu_time:.2f}ms (10 searches)")
            print(f"  Speedup: {results['speedup']:.1f}x")
        
    except Exception as e:
        results["errors"] = [str(e)]
        print(f"❌ Test failed: {e}")
    
    return results


def test_nexusmind_integration() -> dict:
    """Test NexusMind FAISSBackend with GPU."""
    results = {"passed": False, "details": ""}
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend
        
        print("\nTesting NexusMind FAISSBackend...")
        
        # Create backend with GPU
        backend = FAISSBackend(dim=768, use_gpu=True)
        
        # Build index
        np.random.seed(42)
        vectors = np.random.randn(1000, 768).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        backend.build(vectors)
        
        # Check if on GPU
        if hasattr(backend.index, 'getDevice'):
            results["passed"] = True
            results["details"] = f"Index on GPU device {backend.index.getDevice()}"
            print(f"✅ FAISSBackend using GPU: {results['details']}")
        else:
            results["details"] = "Index on CPU (GPU not used)"
            print(f"⚠️  FAISSBackend using CPU")
            
    except Exception as e:
        results["details"] = str(e)
        print(f"❌ NexusMind integration test failed: {e}")
    
    return results


def print_summary(results: dict) -> int:
    """Print summary and return exit code."""
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    install = results["installation"]
    print(f"FAISS Version: {install.get('version', 'N/A')}")
    print(f"GPU Available: {'✅ Yes' if install.get('gpu_available') else '❌ No'}")
    print(f"Number of GPUs: {install.get('num_gpus', 0)}")
    
    if "functionality" in results:
        func = results["functionality"]
        if func["cpu_test"]["passed"]:
            print(f"CPU Test: ✅ {func['cpu_test']['time_ms']:.2f}ms")
        if func["gpu_test"]["passed"]:
            print(f"GPU Test: ✅ {func['gpu_test']['time_ms']:.2f}ms")
            print(f"Speedup: {func.get('speedup', 0):.1f}x")
    
    if "nexusmind" in results:
        nm = results["nexusmind"]
        status = "✅" if nm["passed"] else "❌"
        print(f"NexusMind Integration: {status} {nm.get('details', '')}")
    
    # Return appropriate exit code
    if install.get("gpu_available") and install.get("num_gpus", 0) > 0:
        print("\n🎉 FAISS GPU is ready for NexusMind!")
        return 0
    else:
        print("\n⚠️  FAISS GPU not available. Using CPU mode.")
        return 1


def main():
    """Main entry point."""
    print("=" * 50)
    print("NexusMind FAISS GPU Verification")
    print("=" * 50)
    
    results = {}
    
    # Check installation
    print("\n1. Checking FAISS installation...")
    results["installation"] = check_faiss_installation()
    
    install = results["installation"]
    if not install["installed"]:
        print("❌ FAISS not installed!")
        return 1
    
    print(f"✅ FAISS {install['version']} installed")
    if install["gpu_available"]:
        print(f"✅ GPU support available ({install['num_gpus']} GPU(s))")
    else:
        print("⚠️  GPU support NOT available")
        return print_summary(results)
    
    # Test functionality
    print("\n2. Testing GPU functionality...")
    results["functionality"] = test_gpu_functionality()
    
    # Test NexusMind integration
    print("\n3. Testing NexusMind integration...")
    results["nexusmind"] = test_nexusmind_integration()
    
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
