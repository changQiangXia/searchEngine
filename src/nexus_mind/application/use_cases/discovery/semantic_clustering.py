"""Semantic Clustering - Automatically discover concept groups in the index.

Uses clustering algorithms (HDBSCAN, KMeans) to group similar images
and automatically label them based on representative samples.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Cluster:
    """A semantic cluster of images.

    Attributes:
        id: Unique cluster identifier
        size: Number of items in cluster
        center: Centroid embedding of cluster
        members: List of member indices and metadata
        label: Auto-generated label for the cluster
        coherence: Cluster coherence score (0-1)
    """

    id: int
    size: int
    center: np.ndarray
    members: list[dict[str, Any]]
    label: str
    coherence: float


class SemanticClustering:
    """Cluster images based on semantic similarity.

    This class provides multiple clustering strategies:
    - HDBSCAN: Density-based, finds irregular-shaped clusters
    - KMeans: Partition-based, good for spherical clusters
    - Agglomerative: Hierarchical clustering for concept trees

    Example:
        >>> clusterer = SemanticClustering(method="hdbscan")
        >>> clusters = clusterer.cluster(embeddings, metadata)
        >>> for cluster in clusters:
        ...     print(f"Cluster {cluster.label}: {cluster.size} items")

    Attributes:
        method: Clustering algorithm ("hdbscan", "kmeans", "agglomerative")
        min_cluster_size: Minimum items per cluster
        min_samples: HDBSCAN min_samples parameter
    """

    def __init__(
        self,
        method: str = "hdbscan",
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        n_clusters: int | None = None,
    ):
        """Initialize clustering.

        Args:
            method: Clustering algorithm
            min_cluster_size: Minimum cluster size (HDBSCAN)
            min_samples: HDBSCAN min_samples
            n_clusters: Number of clusters (KMeans)
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.n_clusters = n_clusters
        self._clusterer = None

    def cluster(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        auto_label: bool = True,
    ) -> list[Cluster]:
        """Cluster embeddings and return cluster information.

        Args:
            embeddings: Array of shape (N, D)
            metadata: Metadata for each embedding
            auto_label: Whether to auto-generate labels

        Returns:
            List of Cluster objects
        """
        if self.method == "hdbscan":
            labels = self._cluster_hdbscan(embeddings)
        elif self.method == "kmeans":
            labels = self._cluster_kmeans(embeddings)
        elif self.method == "agglomerative":
            labels = self._cluster_agglomerative(embeddings)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Build clusters from labels
        clusters = self._build_clusters(embeddings, metadata, labels)

        if auto_label:
            clusters = self._label_clusters(clusters)

        return clusters

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using HDBSCAN."""
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan required. Install: pip install hdbscan")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        return labels

    def _cluster_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using KMeans."""
        from sklearn.cluster import KMeans

        n_clusters = self.n_clusters or max(2, len(embeddings) // 20)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)
        return labels

    def _cluster_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using Agglomerative Clustering."""
        from sklearn.cluster import AgglomerativeClustering

        n_clusters = self.n_clusters or max(2, len(embeddings) // 20)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="ward",
        )
        labels = clustering.fit_predict(embeddings)
        return labels

    def _build_clusters(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        labels: np.ndarray,
    ) -> list[Cluster]:
        """Build Cluster objects from labels."""
        # Group by cluster label
        clusters_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters_dict[label].append((idx, embeddings[idx], metadata[idx]))

        clusters = []
        for cluster_id, members in clusters_dict.items():
            if cluster_id == -1:  # HDBSCAN noise points
                continue

            # Calculate centroid
            member_embeddings = np.array([m[1] for m in members])
            centroid = np.mean(member_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # Calculate coherence (average similarity to centroid)
            similarities = np.dot(member_embeddings, centroid)
            coherence = float(np.mean(similarities))

            # Build member list
            member_list = [
                {
                    "index": m[0],
                    "metadata": m[2],
                    "similarity_to_center": float(similarities[i]),
                }
                for i, m in enumerate(members)
            ]

            # Sort by similarity to center
            member_list.sort(key=lambda x: x["similarity_to_center"], reverse=True)

            cluster = Cluster(
                id=cluster_id,
                size=len(members),
                center=centroid,
                members=member_list,
                label=f"Cluster_{cluster_id}",
                coherence=coherence,
            )
            clusters.append(cluster)

        # Sort by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)

        return clusters

    def _label_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        """Auto-generate labels for clusters.

        In a real implementation, this would use:
        1. LLM to describe representative images
        2. Common visual concepts from metadata
        3. Tag frequency analysis

        For now, use simple heuristics.
        """
        for cluster in clusters:
            # Use most common words from member names
            names = [m["metadata"].get("name", "") for m in cluster.members[:5]]

            # Extract common patterns
            common_terms = self._extract_common_terms(names)

            if common_terms:
                cluster.label = common_terms[0].replace("_", " ").title()
            else:
                cluster.label = f"Group {cluster.id}"

        return clusters

    def _extract_common_terms(self, names: list[str]) -> list[str]:
        """Extract common terms from filenames."""
        # Simple extraction - split by common delimiters
        terms = []
        for name in names:
            # Remove extension and split
            base = name.split(".")[0]
            parts = base.replace("-", "_").split("_")
            terms.extend(parts)

        # Count frequency
        from collections import Counter

        counter = Counter(terms)

        # Return most common (excluding very short terms)
        return [term for term, count in counter.most_common(3) if len(term) > 2]

    def get_cluster_hierarchy(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        n_levels: int = 3,
    ) -> dict[int, list[Cluster]]:
        """Get hierarchical clustering (concept tree).

        Args:
            embeddings: Embeddings to cluster
            metadata: Metadata for embeddings
            n_levels: Number of hierarchy levels

        Returns:
            Dictionary mapping level to list of clusters
        """
        hierarchy = {}

        for level in range(n_levels):
            # Increase granularity at each level
            n_clusters = 2 ** (level + 1)

            temp_clusterer = SemanticClustering(
                method="kmeans",
                n_clusters=n_clusters,
            )
            clusters = temp_clusterer.cluster(embeddings, metadata)

            hierarchy[level] = clusters

        return hierarchy


class ConceptTreeBuilder:
    """Build a hierarchical tree of concepts from embeddings.

    Creates a tree structure where:
    - Root: All data
    - Level 1: Broad categories
    - Level 2: Subcategories
    - Level 3+: Fine-grained concepts
    """

    def __init__(self, encoder: Callable | None = None):
        """Initialize tree builder.

        Args:
            encoder: Optional encoder for generating concept descriptions
        """
        self.encoder = encoder

    def build_tree(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Build concept tree.

        Args:
            embeddings: All embeddings
            metadata: All metadata
            max_depth: Maximum tree depth

        Returns:
            Tree structure as nested dictionary
        """
        clusterer = SemanticClustering(method="agglomerative")

        def build_level(embs, metas, depth):
            if depth >= max_depth or len(embs) < 10:
                return {
                    "type": "leaf",
                    "size": len(embs),
                    "samples": [m.get("name", str(i)) for i, m in enumerate(metas[:5])],
                }

            # Cluster at this level
            n_clusters = min(4, len(embs) // 5)
            if n_clusters < 2:
                return {
                    "type": "leaf",
                    "size": len(embs),
                    "samples": [m.get("name", str(i)) for i, m in enumerate(metas[:5])],
                }

            clusterer.n_clusters = n_clusters
            clusters = clusterer.cluster(embs, metas)

            children = []
            for cluster in clusters:
                # Get embeddings for this cluster
                indices = [m["index"] for m in cluster.members]
                cluster_embs = embs[indices]
                cluster_metas = [metas[i] for i in indices]

                child = {
                    "type": "node",
                    "label": cluster.label,
                    "size": cluster.size,
                    "coherence": cluster.coherence,
                    "children": build_level(cluster_embs, cluster_metas, depth + 1),
                }
                children.append(child)

            return children

        tree = {
            "type": "root",
            "size": len(embeddings),
            "children": build_level(embeddings, metadata, 0),
        }

        return tree
