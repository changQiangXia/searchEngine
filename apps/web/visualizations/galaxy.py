"""3D Semantic Galaxy Visualization.

Creates interactive 3D scatter plots of the semantic space using
dimensionality reduction techniques (PCA, t-SNE, UMAP).
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 3,
) -> np.ndarray:
    """Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: Array of shape (N, D)
        method: "pca", "tsne", or "umap"
        n_components: Number of output dimensions (2 or 3)
        
    Returns:
        Array of shape (N, n_components)
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(embeddings)
    
    elif method == "tsne":
        from sklearn.manifold import TSNE
        # Adjust perplexity based on sample size
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not installed, falling back to PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            return reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def cluster_points(coords: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Cluster points for coloring.
    
    Args:
        coords: Coordinates array (N, 2) or (N, 3)
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels array (N,)
    """
    from sklearn.cluster import KMeans
    
    n_clusters = min(n_clusters, len(coords))
    if n_clusters < 2:
        return np.zeros(len(coords), dtype=int)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(coords)


def compute_similarity_to_query(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> np.ndarray:
    """Compute similarity between embeddings and query.
    
    Args:
        embeddings: Array of shape (N, D)
        query_embedding: Query vector (D,)
        
    Returns:
        Similarity scores (N,)
    """
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query = query_embedding / np.linalg.norm(query_embedding)
    
    # Cosine similarity
    similarities = np.dot(embeddings, query)
    return similarities


def generate_galaxy_plot(
    engine: Any,
    n_samples: int = 100,
    method: str = "pca",
    color_by: str = "Cluster",
    query: Optional[str] = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Generate 3D semantic galaxy plot.
    
    Args:
        engine: NexusEngine instance
        n_samples: Number of points to sample
        method: Dimensionality reduction method
        color_by: "Similarity to Query", "Cluster", or "Random"
        query: Query for similarity coloring
        width: Plot width
        height: Plot height
        
    Returns:
        Plotly Figure object
    """
    # Check if we can get embeddings
    # Note: This is a limitation - we need to store embeddings separately
    # For now, generate random data as placeholder
    
    st = None
    try:
        import streamlit as st
    except ImportError:
        pass
    
    if st:
        st.warning("⚠️ Galaxy visualization requires embedding cache. Using sample data for demo.")
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = min(n_samples, len(engine.index) if engine.index else 100)
    
    # Create synthetic embeddings that form clusters
    n_clusters = 5
    points_per_cluster = n_samples // n_clusters
    
    embeddings = []
    cluster_labels = []
    
    for i in range(n_clusters):
        # Cluster center
        center = np.random.randn(768)
        center = center / np.linalg.norm(center)
        
        # Generate points around center
        for _ in range(points_per_cluster):
            noise = np.random.randn(768) * 0.1
            point = center + noise
            point = point / np.linalg.norm(point)
            embeddings.append(point)
            cluster_labels.append(i)
    
    embeddings = np.array(embeddings)
    cluster_labels = np.array(cluster_labels)
    
    # Get metadata
    metadata = []
    if engine.index and engine.index.metadata:
        for i in range(min(n_samples, len(engine.index.metadata))):
            metadata.append(engine.index.metadata[i])
    else:
        metadata = [{"name": f"item_{i}"} for i in range(n_samples)]
    
    # Reduce dimensions
    coords_3d = reduce_dimensions(embeddings, method=method, n_components=3)
    
    # Determine colors
    if color_by == "Similarity to Query" and query:
        query_emb = engine.clip.encode_text([query])
        colors = compute_similarity_to_query(embeddings, query_emb[0])
        color_scale = "Viridis"
        color_title = "Similarity"
    elif color_by == "Cluster":
        colors = cluster_labels
        color_scale = "Set1"
        color_title = "Cluster"
    else:
        colors = np.random.rand(n_samples)
        color_scale = "Rainbow"
        color_title = "Random"
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale=color_scale,
            opacity=0.8,
            showscale=True,
            colorbar=dict(title=color_title),
        ),
        text=[m.get("name", f"item_{i}") for i, m in enumerate(metadata)],
        hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
    )])
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"🌌 Semantic Galaxy ({method.upper()})",
            x=0.5,
            font=dict(size=20),
        ),
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
            aspectmode='cube',
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    # Add query point if applicable
    if color_by == "Similarity to Query" and query:
        # Place query at center or compute its position
        query_center = np.mean(coords_3d, axis=0)
        
        fig.add_trace(go.Scatter3d(
            x=[query_center[0]],
            y=[query_center[1]],
            z=[query_center[2]],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='white', width=2),
            ),
            name='Query',
            text=[query],
            hovertemplate='<b>Query: %{text}</b><extra></extra>',
        ))
    
    return fig


def generate_galaxy_2d(
    engine: Any,
    n_samples: int = 100,
    method: str = "pca",
) -> go.Figure:
    """Generate 2D semantic galaxy plot.
    
    Args:
        engine: NexusEngine instance
        n_samples: Number of points to sample
        method: Dimensionality reduction method
        
    Returns:
        Plotly Figure object
    """
    # Similar to 3D version but with 2D projection
    np.random.seed(42)
    n_samples = min(n_samples, len(engine.index) if engine.index else 100)
    
    # Generate sample data
    n_clusters = 5
    points_per_cluster = n_samples // n_clusters
    
    embeddings = []
    cluster_labels = []
    
    for i in range(n_clusters):
        center = np.random.randn(768)
        center = center / np.linalg.norm(center)
        
        for _ in range(points_per_cluster):
            noise = np.random.randn(768) * 0.1
            point = center + noise
            point = point / np.linalg.norm(point)
            embeddings.append(point)
            cluster_labels.append(i)
    
    embeddings = np.array(embeddings)
    
    # Reduce to 2D
    coords_2d = reduce_dimensions(embeddings, method=method, n_components=2)
    
    # Create 2D scatter
    fig = go.Figure(data=[go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=cluster_labels,
            colorscale='Set1',
            opacity=0.7,
            line=dict(width=1, color='white'),
        ),
        text=[f"item_{i}" for i in range(n_samples)],
        hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
    )])
    
    fig.update_layout(
        title=dict(
            text=f"🌌 Semantic Map ({method.upper()})",
            x=0.5,
        ),
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=700,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig