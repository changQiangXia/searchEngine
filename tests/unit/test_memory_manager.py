"""Tests for GPU memory manager."""

import pytest
import torch

from nexus_mind.infrastructure.memory.manager import (
    GPUMemoryManager,
    MemoryPressureLevel,
    MemoryStats,
)


class TestMemoryStats:
    """Test MemoryStats dataclass."""
    
    def test_basic_properties(self):
        stats = MemoryStats(
            gpu_total=12_000_000_000,
            gpu_used=6_000_000_000,
            gpu_cached=1_000_000_000,
            ram_available=16_000_000_000,
        )
        
        assert stats.gpu_usage_pct == 50.0
        assert stats.gpu_available == 6_000_000_000
    
    def test_zero_gpu(self):
        stats = MemoryStats(
            gpu_total=0,
            gpu_used=0,
            gpu_cached=0,
            ram_available=16_000_000_000,
        )
        
        assert stats.gpu_usage_pct == 0.0


class TestGPUMemoryManager:
    """Test GPUMemoryManager."""
    
    def test_singleton(self):
        """Test singleton pattern."""
        manager1 = GPUMemoryManager()
        manager2 = GPUMemoryManager()
        assert manager1 is manager2
    
    def test_get_stats(self):
        """Test memory stats retrieval."""
        manager = GPUMemoryManager()
        stats = manager.get_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.gpu_total >= 0
        assert stats.gpu_used >= 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_check_pressure(self):
        """Test pressure level detection."""
        manager = GPUMemoryManager()
        pressure = manager.check_pressure()
        
        assert isinstance(pressure, MemoryPressureLevel)
        assert pressure in [
            MemoryPressureLevel.NORMAL,
            MemoryPressureLevel.WARNING,
            MemoryPressureLevel.CRITICAL,
            MemoryPressureLevel.EMERGENCY,
        ]
    
    def test_auto_clean(self):
        """Test cleanup functionality."""
        manager = GPUMemoryManager()
        
        # Should not raise
        manager.auto_clean(aggressive=False)
        manager.auto_clean(aggressive=True)
    
    def test_ensure_available(self):
        """Test memory availability check."""
        manager = GPUMemoryManager()
        
        # Should always return True for small requests
        assert manager.ensure_available(100) is True


@pytest.mark.gpu
class TestGPUMemoryManagerCUDA:
    """CUDA-specific tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_initialized(self):
        manager = GPUMemoryManager()
        assert manager.total_memory > 0
        assert manager.safe_limit > 0
        assert manager.safe_limit < manager.total_memory