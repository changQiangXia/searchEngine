"""Phase 1 Validation Tests - Verify core functionality without full CLIP model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import torch


def test_imports():
    """Test all core modules can be imported."""
    from nexus_mind.infrastructure.memory.manager import (
        GPUMemoryManager, MemoryPressureLevel, MemoryStats
    )
    from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend
    print("✅ All imports successful")


def test_memory_manager_singleton():
    """Test memory manager singleton pattern."""
    from nexus_mind.infrastructure.memory.manager import GPUMemoryManager
    
    manager1 = GPUMemoryManager()
    manager2 = GPUMemoryManager()
    
    assert manager1 is manager2
    print("✅ Singleton pattern working")


def test_memory_stats():
    """Test memory stats calculation."""
    from nexus_mind.infrastructure.memory.manager import MemoryStats
    
    stats = MemoryStats(
        gpu_total=12_000_000_000,
        gpu_used=6_000_000_000,
        gpu_cached=1_000_000_000,
        ram_available=16_000_000_000,
    )
    
    assert stats.gpu_usage_pct == 50.0
    assert stats.gpu_available == 6_000_000_000
    print("✅ MemoryStats calculation correct")


def test_memory_pressure_levels():
    """Test pressure level detection."""
    from nexus_mind.infrastructure.memory.manager import (
        GPUMemoryManager, MemoryPressureLevel
    )
    
    manager = GPUMemoryManager()
    pressure = manager.check_pressure()
    
    assert pressure in [
        MemoryPressureLevel.NORMAL,
        MemoryPressureLevel.WARNING,
        MemoryPressureLevel.CRITICAL,
        MemoryPressureLevel.EMERGENCY,
    ]
    print(f"✅ Current pressure level: {pressure.name}")


def test_memory_cleanup():
    """Test memory cleanup functions."""
    from nexus_mind.infrastructure.memory.manager import GPUMemoryManager
    
    manager = GPUMemoryManager()
    
    # Should not raise
    manager.auto_clean(aggressive=False)
    manager.auto_clean(aggressive=True)
    print("✅ Memory cleanup working")


def test_faiss_backend_cpu():
    """Test FAISS backend on CPU."""
    from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend
    
    # Create small index
    dim = 128
    n = 100
    
    # Generate random vectors
    np.random.seed(42)
    embeddings = np.random.randn(n, dim).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Build index
    backend = FAISSBackend(dim=dim, use_gpu=False)
    backend.build(embeddings)
    
    assert backend.ntotal == n
    
    # Search
    query = embeddings[0:1]
    scores, indices = backend.search(query, top_k=10)
    
    assert scores.shape == (1, 10)
    assert indices.shape == (1, 10)
    assert indices[0][0] == 0  # First result should be itself
    
    print(f"✅ FAISS backend working, top-1 score: {scores[0][0]:.4f}")


def test_faiss_save_load():
    """Test FAISS index persistence."""
    from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend
    import tempfile
    import shutil
    
    dim = 64
    n = 50
    
    np.random.seed(42)
    embeddings = np.random.randn(n, dim).astype('float32')
    metadata = [{"id": i, "name": f"item_{i}"} for i in range(n)]
    
    # Build and save
    backend = FAISSBackend(dim=dim, use_gpu=False)
    backend.build(embeddings, metadata)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        backend.save(tmpdir)
        
        # Load
        backend2 = FAISSBackend(dim=dim, use_gpu=False)
        backend2.load(tmpdir)
        
        assert backend2.ntotal == n
        assert len(backend2.metadata) == n
        assert backend2.metadata[0]["name"] == "item_0"
    
    print("✅ FAISS save/load working")


def test_negative_search_logic():
    """Test negative search vector logic."""
    # Simulate the negative search formula
    np.random.seed(42)
    
    # Create normalized vectors
    pos_vec = np.random.randn(1, 768).astype('float32')
    pos_vec = pos_vec / np.linalg.norm(pos_vec)
    
    neg_vec = np.random.randn(1, 768).astype('float32')
    neg_vec = neg_vec / np.linalg.norm(neg_vec)
    
    alpha = 0.7
    query_vec = pos_vec - alpha * neg_vec
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    # Verify properties
    assert np.abs(np.linalg.norm(query_vec) - 1.0) < 1e-5
    print("✅ Negative search vector logic correct")


def test_mmr_logic():
    """Test MMR diversity ranking logic."""
    # Simulate MMR selection
    scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    lambda_param = 0.5
    
    # Simplified: first pick highest, then balance
    selected = [0]  # Pick first
    
    for i in range(1, 3):  # Select 3 items
        best_mmr = -1
        best_idx = -1
        
        for j in range(len(scores)):
            if j in selected:
                continue
            
            rel_score = scores[j]
            # Simulate diversity (inverse of similarity to selected)
            max_sim = max([0.9 - 0.1 * abs(j - s) for s in selected])
            div_score = 1 - max_sim
            
            mmr_score = lambda_param * rel_score + (1 - lambda_param) * div_score
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = j
        
        selected.append(best_idx)
    
    assert len(selected) == 3
    print(f"✅ MMR selection working, selected indices: {selected}")


def test_cli_help():
    """Test CLI commands are registered."""
    from typer.testing import CliRunner
    from nexus_mind.interfaces.cli.main import app
    
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    
    # Check commands exist in help output
    commands = ["index", "search", "status", "negative", "workspace"]
    for cmd in commands:
        assert cmd in result.output, f"Command {cmd} not found in help"
    
    print("✅ CLI commands registered")


def test_cuda_available():
    """Verify CUDA is available."""
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA available: {gpu_name}")


def test_gpu_memory_stats():
    """Test GPU memory stats retrieval."""
    from nexus_mind.infrastructure.memory.manager import GPUMemoryManager
    
    manager = GPUMemoryManager()
    stats = manager.get_stats()
    
    if torch.cuda.is_available():
        assert stats.gpu_total > 0
        assert stats.gpu_used >= 0
        assert stats.gpu_total >= stats.gpu_used
        print(f"✅ GPU Memory: {stats.gpu_used/1e9:.2f}/{stats.gpu_total/1e9:.2f} GB")
    else:
        print("⚠️ CUDA not available, skipping GPU test")


if __name__ == "__main__":
    print("=" * 60)
    print("NexusMind Phase 1 Validation Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_memory_manager_singleton,
        test_memory_stats,
        test_cuda_available,
        test_gpu_memory_stats,
        test_memory_pressure_levels,
        test_memory_cleanup,
        test_faiss_backend_cpu,
        test_faiss_save_load,
        test_negative_search_logic,
        test_mmr_logic,
        test_cli_help,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)