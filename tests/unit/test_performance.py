"""Performance Optimization Tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import numpy as np


def test_quantization_config():
    """Test quantization configuration."""
    from nexus_mind.infrastructure.models.quantization import (
        QuantizationConfig, QuantizationType, estimate_quantized_memory
    )
    
    # Test config creation
    config = QuantizationConfig(
        quant_type=QuantizationType.FP16,
        device="cuda",
    )
    
    assert config.quant_type == QuantizationType.FP16
    assert config.memory_reduction == 2.0
    
    # Test memory estimation
    mem_info = estimate_quantized_memory(
        model_name="openai/clip-vit-large-patch14",
        quant_type=QuantizationType.INT8,
    )
    
    assert mem_info["reduction_factor"] == 4.0
    assert mem_info["savings_gb"] > 0
    
    print("✅ Quantization config working")


def test_batch_config():
    """Test batch configuration."""
    from nexus_mind.infrastructure.compute.optimizer import (
        BatchConfig, optimize_for_3080ti, optimize_for_4090
    )
    
    # Test default config
    config = BatchConfig()
    assert config.batch_size == 32
    
    # Test 3080ti optimization
    config_3080 = optimize_for_3080ti()
    assert config_3080.batch_size == 64
    assert config_3080.dynamic_batching is True
    
    # Test 4090 optimization
    config_4090 = optimize_for_4090()
    assert config_4090.batch_size == 128
    
    print("✅ Batch config optimization working")


def test_dynamic_batcher():
    """Test dynamic batcher."""
    from nexus_mind.infrastructure.compute.optimizer import (
        BatchConfig, DynamicBatcher
    )
    
    config = BatchConfig(batch_size=32, min_batch_size=8, max_batch_size=64)
    batcher = DynamicBatcher(config)
    
    # Test initial batch size (may differ based on CUDA availability)
    bs = batcher.get_batch_size()
    assert config.min_batch_size <= bs <= config.max_batch_size
    
    # Test recording
    batcher.record_batch(bs, 1.0)
    batcher.record_batch(bs, 0.9)
    
    throughput = batcher.get_throughput()
    assert throughput > 0
    
    print("✅ Dynamic batcher working")


def test_performance_monitor():
    """Test performance monitor."""
    from nexus_mind.infrastructure.compute.optimizer import PerformanceMonitor
    import time
    
    monitor = PerformanceMonitor()
    
    # Test timing
    monitor.start_timer("test_op")
    time.sleep(0.01)
    duration = monitor.end_timer("test_op")
    
    assert duration > 0
    
    # Test recording
    monitor.record_throughput(100, 1.0)
    monitor.record_memory()
    
    stats = monitor.get_stats()
    assert "throughput" in stats
    assert "latency" in stats
    
    print("✅ Performance monitor working")


def test_streaming_batcher():
    """Test streaming batcher."""
    from nexus_mind.infrastructure.compute.optimizer import StreamingBatcher
    
    batcher = StreamingBatcher(batch_size=10)
    
    # Create iterator
    items = iter(range(25))
    
    batches = list(batcher.stream_batches(items))
    
    assert len(batches) == 3  # 10 + 10 + 5
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5
    
    print("✅ Streaming batcher working")


def test_tiered_cache_integration():
    """Test tiered cache with optimization."""
    import tempfile
    from nexus_mind.infrastructure.storage.cache.tiered_cache import TieredCache
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TieredCache(
            l1_size=int(1e6),
            l2_path=os.path.join(tmpdir, "l2"),
            l3_path=os.path.join(tmpdir, "l3"),
        )
        
        # Test cache with different levels
        data = np.random.randn(100, 100)
        
        # Put in L1
        cache.put("l1_test", data, level=1)
        retrieved = cache.get("l1_test")
        assert retrieved is not None
        assert np.allclose(retrieved, data)
        
        # Put in L2
        cache.put("l2_test", data, level=2)
        retrieved = cache.get("l2_test")
        assert retrieved is not None
        
        # Check stats
        stats = cache.get_stats()
        assert stats["l1"]["items"] >= 1
        
        print("✅ Tiered cache integration working")


if __name__ == "__main__":
    print("=" * 60)
    print("Performance Optimization Tests")
    print("=" * 60)
    
    tests = [
        test_quantization_config,
        test_batch_config,
        test_dynamic_batcher,
        test_performance_monitor,
        test_streaming_batcher,
        test_tiered_cache_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)