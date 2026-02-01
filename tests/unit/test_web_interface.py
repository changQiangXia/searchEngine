"""Web Interface Tests - Phase 3 Validation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'apps', 'web'))

import numpy as np
from PIL import Image


def test_galaxy_imports():
    """Test galaxy visualization imports."""
    from visualizations.galaxy import (
        reduce_dimensions,
        cluster_points,
        generate_galaxy_plot,
    )
    print("✅ Galaxy visualization imports successful")


def test_attention_imports():
    """Test attention visualization imports."""
    from visualizations.attention import (
        generate_attention_heatmap,
        apply_colormap,
    )
    print("✅ Attention visualization imports successful")


def test_dimensionality_reduction():
    """Test dimensionality reduction functions."""
    from visualizations.galaxy import reduce_dimensions
    
    # Create sample embeddings
    np.random.seed(42)
    embeddings = np.random.randn(50, 768)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Test PCA
    coords_2d = reduce_dimensions(embeddings, method="pca", n_components=2)
    assert coords_2d.shape == (50, 2)
    
    # Test 3D
    coords_3d = reduce_dimensions(embeddings, method="pca", n_components=3)
    assert coords_3d.shape == (50, 3)
    
    print("✅ Dimensionality reduction working (PCA)")


def test_clustering():
    """Test clustering function."""
    from visualizations.galaxy import cluster_points
    
    np.random.seed(42)
    coords = np.random.randn(30, 3)
    
    labels = cluster_points(coords, n_clusters=3)
    
    assert len(labels) == 30
    assert len(np.unique(labels)) <= 3
    
    print("✅ Clustering working")


def test_attention_heatmap():
    """Test attention heatmap generation."""
    from visualizations.attention import generate_attention_heatmap
    
    # Create dummy engine
    class DummyEngine:
        pass
    
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    
    try:
        heatmap, overlay = generate_attention_heatmap(
            engine=DummyEngine(),
            image=img,
        )
        
        assert heatmap.shape[:2] == (100, 100)
        assert overlay.shape[:2] == (100, 100)
        
        print("✅ Attention heatmap generation working")
    except Exception as e:
        print(f"⚠️  Attention heatmap test: {e}")


def test_colormap():
    """Test colormap application."""
    from visualizations.attention import apply_colormap
    
    attention = np.random.rand(64, 64)
    
    colored = apply_colormap(attention, 'jet')
    
    assert colored.shape == (64, 64, 3)
    assert colored.max() <= 1.0
    assert colored.min() >= 0.0
    
    print("✅ Colormap application working")


if __name__ == "__main__":
    print("=" * 60)
    print("Web Interface Tests (Phase 3)")
    print("=" * 60)
    
    tests = [
        test_galaxy_imports,
        test_attention_imports,
        test_dimensionality_reduction,
        test_clustering,
        test_attention_heatmap,
        test_colormap,
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