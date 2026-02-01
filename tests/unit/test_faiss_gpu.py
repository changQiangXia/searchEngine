"""Unit tests for FAISS GPU functionality.

These tests validate GPU-accelerated FAISS operations.
Skip if GPU is not available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend

# Skip all tests if GPU not available
try:
    import faiss
    GPU_AVAILABLE = faiss.get_num_gpus() > 0
except:
    GPU_AVAILABLE = False


pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")


class TestFAISSGPU:
    """Test FAISS GPU functionality."""
    
    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors for testing."""
        np.random.seed(42)
        vectors = np.random.randn(1000, 768).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors
    
    def test_gpu_index_build(self, sample_vectors):
        """Test building index on GPU."""
        backend = FAISSBackend(dim=768, use_gpu=True)
        backend.build(sample_vectors)
        
        # Verify index is on GPU
        assert hasattr(backend.index, 'getDevice'), "Index should be on GPU"
        assert backend.index.getDevice() == 0, "Index should be on GPU device 0"
        assert backend.ntotal == len(sample_vectors)
    
    def test_gpu_search(self, sample_vectors):
        """Test searching on GPU index."""
        backend = FAISSBackend(dim=768, use_gpu=True)
        backend.build(sample_vectors)
        
        # Search
        query = sample_vectors[0]
        scores, indices = backend.search(query, top_k=10)
        
        # Verify results
        assert scores.shape == (1, 10)
        assert indices.shape == (1, 10)
        # First result should be the query itself (or very close)
        assert indices[0][0] == 0 or scores[0][0] > 0.99
    
    def test_gpu_flat_index(self, sample_vectors):
        """Test GPU Flat index."""
        backend = FAISSBackend(dim=768, index_type="flat", use_gpu=True)
        backend.build(sample_vectors)
        
        assert backend.index_type == "flat"
        assert hasattr(backend.index, 'getDevice')
    
    def test_gpu_ivf_index(self, sample_vectors):
        """Test GPU IVF index."""
        backend = FAISSBackend(dim=768, index_type="ivf", use_gpu=True)
        backend.build(sample_vectors)
        
        assert backend.index_type == "ivf"
        # IVF index should be trained
        assert backend._is_trained
    
    def test_gpu_fallback_to_cpu(self):
        """Test automatic fallback to CPU when GPU unavailable."""
        # Create backend with use_gpu=True but force CPU
        backend = FAISSBackend(dim=768, use_gpu=True)
        
        vectors = np.random.randn(100, 768).astype('float32')
        
        # Build with force_cpu=True
        backend.build(vectors, force_cpu=True)
        
        # Should be on CPU
        assert not hasattr(backend.index, 'getDevice'), "Index should be on CPU"
    
    def test_save_load_gpu_index(self, sample_vectors, tmp_path):
        """Test saving and loading GPU index."""
        # Build GPU index
        backend = FAISSBackend(dim=768, use_gpu=True)
        backend.build(sample_vectors)
        
        # Save
        save_path = tmp_path / "test_index"
        backend.save(save_path)
        
        # Load
        backend2 = FAISSBackend(dim=768, use_gpu=True)
        backend2.load(save_path)
        
        # Verify
        assert backend2.ntotal == len(sample_vectors)
        
        # Test search still works
        query = sample_vectors[0]
        scores, indices = backend2.search(query, top_k=5)
        assert scores.shape == (1, 5)
    
    def test_batch_search_gpu(self, sample_vectors):
        """Test batch search on GPU."""
        backend = FAISSBackend(dim=768, use_gpu=True)
        backend.build(sample_vectors)
        
        # Batch search
        queries = sample_vectors[:10]
        results = backend.batch_search(queries, top_k=5, batch_size=3)
        
        # Should have 4 batches (3+3+3+1)
        assert len(results) == 4
        
        # Each result should have scores and indices
        for scores, indices in results:
            assert scores.shape[1] == 5
            assert indices.shape[1] == 5
    
    def test_gpu_search_performance(self, sample_vectors):
        """Test that GPU search is reasonably fast."""
        import time
        
        backend = FAISSBackend(dim=768, use_gpu=True)
        backend.build(sample_vectors)
        
        # Warmup
        query = sample_vectors[0]
        backend.search(query, top_k=10)
        
        # Time search
        start = time.perf_counter()
        for _ in range(100):
            backend.search(query, top_k=10)
        elapsed = (time.perf_counter() - start) * 1000
        
        avg_time = elapsed / 100
        # GPU search should be fast (< 5ms per query for 1000 vectors)
        assert avg_time < 5.0, f"GPU search too slow: {avg_time:.2f}ms"


class TestFAISSGPUAvailability:
    """Test GPU availability detection."""
    
    def test_faiss_gpu_available(self):
        """Test that FAISS GPU is available."""
        import faiss
        
        assert hasattr(faiss, 'StandardGpuResources'), "StandardGpuResources not available"
        assert faiss.get_num_gpus() > 0, "No GPUs detected"
    
    def test_gpu_resources_creation(self):
        """Test creating GPU resources."""
        import faiss
        
        res = faiss.StandardGpuResources()
        assert res is not None
        
        # Set temp memory
        res.setTempMemory(512 * 1024 * 1024)
    
    def test_cpu_to_gpu_transfer(self):
        """Test transferring index from CPU to GPU."""
        import faiss
        
        # Create CPU index
        dim = 128
        n = 1000
        index = faiss.IndexFlatIP(dim)
        
        vectors = np.random.randn(n, dim).astype('float32')
        index.add(vectors)
        
        # Transfer to GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        
        assert hasattr(gpu_index, 'getDevice')
        assert gpu_index.ntotal == n
