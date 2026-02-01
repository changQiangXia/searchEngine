"""Phase 2 Feature Tests - Concept interpolation, clustering, etc."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import pytest


def test_slerp_basic():
    """Test SLERP interpolation."""
    from nexus_mind.application.use_cases.discovery.concept_interpolation import slerp
    
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.0, 1.0, 0.0])
    
    # Test endpoints
    result_0 = slerp(v0, v1, 0.0)
    result_1 = slerp(v0, v1, 1.0)
    
    assert np.allclose(result_0, v0 / np.linalg.norm(v0))
    assert np.allclose(result_1, v1 / np.linalg.norm(v1))
    
    # Test midpoint is normalized
    result_mid = slerp(v0, v1, 0.5)
    assert np.abs(np.linalg.norm(result_mid) - 1.0) < 1e-5
    
    print("✅ SLERP basic test passed")


def test_lerp_basic():
    """Test linear interpolation."""
    from nexus_mind.application.use_cases.discovery.concept_interpolation import lerp
    
    v0 = np.array([1.0, 0.0])
    v1 = np.array([0.0, 1.0])
    
    result_mid = lerp(v0, v1, 0.5)
    
    # Midpoint should be close to [0.707, 0.707] (normalized)
    expected = np.array([0.70710678, 0.70710678])
    assert np.allclose(result_mid, expected, atol=1e-5)
    
    print("✅ LERP basic test passed")


def test_concept_interpolator():
    """Test ConceptInterpolator with mock functions."""
    from nexus_mind.application.use_cases.discovery.concept_interpolation import (
        ConceptInterpolator
    )
    
    # Mock encoder
    def mock_encoder(texts):
        # Simple mock: encode text to vector based on hash
        results = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            vec = np.random.randn(768)
            vec = vec / np.linalg.norm(vec)
            results.append(vec)
        return np.array(results)
    
    # Mock searcher
    def mock_searcher(embedding, top_k=3):
        return [
            {"score": 0.9 - i*0.1, "metadata": {"name": f"result_{i}"}}
            for i in range(top_k)
        ]
    
    interpolator = ConceptInterpolator(
        encoder=mock_encoder,
        searcher=mock_searcher,
        method="slerp",
    )
    
    path = interpolator.interpolate("cat", "tiger", steps=3, top_k=2)
    
    assert len(path) == 3
    assert path[0].step == 0.0
    assert path[1].step == 0.5
    assert path[2].step == 1.0
    
    # Check descriptions
    assert "100% cat" in path[0].description
    assert "100% tiger" in path[2].description
    
    print("✅ ConceptInterpolator test passed")


def test_multi_concept_blend():
    """Test MultiConceptBlend."""
    from nexus_mind.application.use_cases.discovery.concept_interpolation import (
        MultiConceptBlend
    )
    
    def mock_encoder(texts):
        results = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            vec = np.random.randn(768)
            vec = vec / np.linalg.norm(vec)
            results.append(vec)
        return np.array(results)
    
    def mock_searcher(embedding, top_k=3):
        return [{"score": 0.9, "metadata": {"name": "blend_result"}}]
    
    blender = MultiConceptBlend(
        encoder=mock_encoder,
        searcher=mock_searcher,
    )
    
    concepts = [("vintage", 0.8), ("neon", 0.2)]
    results = blender.blend(concepts, top_k=1)
    
    assert len(results) == 1
    
    print("✅ MultiConceptBlend test passed")


def test_semantic_clustering_kmeans():
    """Test SemanticClustering with KMeans."""
    from nexus_mind.application.use_cases.discovery.semantic_clustering import (
        SemanticClustering
    )
    
    # Create synthetic clustered data
    np.random.seed(42)
    
    # Cluster 1: around [1, 0, 0]
    cluster1 = np.random.randn(10, 3) * 0.1 + np.array([1, 0, 0])
    # Cluster 2: around [0, 1, 0]
    cluster2 = np.random.randn(10, 3) * 0.1 + np.array([0, 1, 0])
    
    embeddings = np.vstack([cluster1, cluster2])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    metadata = [{"name": f"item_{i}"} for i in range(20)]
    
    clusterer = SemanticClustering(method="kmeans", n_clusters=2)
    clusters = clusterer.cluster(embeddings, metadata)
    
    assert len(clusters) == 2
    assert all(c.size > 0 for c in clusters)
    assert all(0 <= c.coherence <= 1 for c in clusters)
    
    print(f"✅ KMeans clustering test passed: {len(clusters)} clusters found")


def test_time_range():
    """Test TimeRange functionality."""
    from datetime import datetime
    from nexus_mind.application.use_cases.search.temporal_search import TimeRange
    
    # Test basic range
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    tr = TimeRange(start=start, end=end)
    
    assert tr.contains(datetime(2024, 6, 15))
    assert not tr.contains(datetime(2023, 12, 31))
    assert not tr.contains(datetime(2025, 1, 1))
    
    # Test from_string
    tr2 = TimeRange.from_string("2024-01-01", "2024-12-31")
    assert tr2.start == start
    assert tr2.end == end
    
    # Test last_n_days
    tr3 = TimeRange.last_n_days(7)
    assert tr3.start < tr3.end
    
    print("✅ TimeRange test passed")


def test_temporal_bucket():
    """Test TemporalBucket dataclass."""
    from datetime import datetime
    from nexus_mind.application.use_cases.search.temporal_search import TemporalBucket
    
    bucket = TemporalBucket(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
        count=10,
        items=[],
        trend="up",
    )
    
    assert bucket.count == 10
    assert bucket.trend == "up"
    
    print("✅ TemporalBucket test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Feature Tests")
    print("=" * 60)
    
    tests = [
        test_slerp_basic,
        test_lerp_basic,
        test_concept_interpolator,
        test_multi_concept_blend,
        test_semantic_clustering_kmeans,
        test_time_range,
        test_temporal_bucket,
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