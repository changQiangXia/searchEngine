"""Phase 4 Feature Tests - Advanced features and optimization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import numpy as np


def test_cross_modal_chain():
    """Test CrossModalChain."""
    from nexus_mind.application.use_cases.discovery.cross_modal_chain import (
        CrossModalChain, ChainNode, ModalityType
    )
    
    class MockEngine:
        def __init__(self):
            self.index = None
            self.clip = MockCLIP()
    
    class MockCLIP:
        def encode_text(self, texts):
            return np.random.randn(len(texts), 768)
        
        def encode_images(self, images):
            return np.random.randn(len(images), 768)
    
    engine = MockEngine()
    chain = CrossModalChain(engine)
    
    # Test node creation
    node = ChainNode(
        step=0,
        modality=ModalityType.TEXT,
        content="test",
        embedding=np.random.randn(768),
        description="Test node",
    )
    
    assert node.step == 0
    assert node.modality == ModalityType.TEXT
    print("✅ CrossModalChain structure working")


def test_plugin_base():
    """Test plugin base classes."""
    from nexus_mind.plugins.base import PluginInfo, PluginRegistry
    
    # Test PluginInfo
    info = PluginInfo(
        name="test_plugin",
        version="1.0.0",
        description="Test plugin",
    )
    
    assert info.name == "test_plugin"
    assert info.version == "1.0.0"
    
    # Test singleton registry
    reg1 = PluginRegistry()
    reg2 = PluginRegistry()
    assert reg1 is reg2
    
    print("✅ Plugin base classes working")


def test_builtin_plugins():
    """Test built-in plugins."""
    from nexus_mind.plugins.builtin import CSVExporter, JSONExporter
    
    # Test CSV exporter
    csv_exp = CSVExporter()
    assert csv_exp.initialize()
    assert ".csv" in csv_exp.supported_formats()
    
    # Test export
    results = [
        {"rank": 1, "score": 0.9, "metadata": {"name": "test.jpg"}},
    ]
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    
    success = csv_exp.export(results, path)
    assert success
    
    # Cleanup
    os.unlink(path)
    
    print("✅ Built-in plugins working")


def test_tiered_cache():
    """Test tiered cache system."""
    from nexus_mind.infrastructure.storage.cache.tiered_cache import TieredCache
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TieredCache(
            l1_size=int(1e6),  # 1MB for testing
            l2_path=os.path.join(tmpdir, "l2"),
            l3_path=os.path.join(tmpdir, "l3"),
        )
        
        # Test put and get
        key = "test_key"
        data = np.random.randn(100, 100)
        
        # Put in L2
        success = cache.put(key, data, level=2)
        assert success
        
        # Get back
        retrieved = cache.get(key)
        assert retrieved is not None
        assert np.allclose(retrieved, data)
        
        # Check stats
        stats = cache.get_stats()
        assert stats["l2"]["items"] == 1
        
        print("✅ Tiered cache working")


def test_semantic_path_finder():
    """Test SemanticPathFinder."""
    from nexus_mind.application.use_cases.discovery.cross_modal_chain import (
        SemanticPathFinder
    )
    
    class MockEngine:
        def __init__(self):
            self.index = MockIndex()
            self.clip = MockCLIP()
    
    class MockIndex:
        def search_with_metadata(self, embedding, top_k=1):
            return [
                {"score": 0.8, "metadata": {"name": "intermediate.jpg"}},
            ]
    
    class MockCLIP:
        def encode_text(self, texts):
            return np.random.randn(len(texts), 768)
    
    engine = MockEngine()
    finder = SemanticPathFinder(engine)
    
    path = finder.find_path("start", "end", max_hops=2)
    
    assert len(path) > 0
    assert path[0]["type"] == "start"
    assert path[-1]["type"] == "end"
    
    print("✅ SemanticPathFinder working")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Feature Tests")
    print("=" * 60)
    
    tests = [
        test_cross_modal_chain,
        test_plugin_base,
        test_builtin_plugins,
        test_tiered_cache,
        test_semantic_path_finder,
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