"""Concept Interpolation - Discover intermediate concepts between two queries.

This module implements semantic interpolation in the embedding space,
allowing users to explore the "semantic path" between two concepts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class InterpolationPoint:
    """A single point along the interpolation path.

    Attributes:
        step: Position along the path (0.0 to 1.0)
        description: Human-readable description of this point
        embedding: The interpolated embedding vector
        neighbors: Top-k nearest neighbors from the index
    """

    step: float
    description: str
    embedding: np.ndarray
    neighbors: list[dict]


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) between two vectors.

    SLERP maintains constant angular velocity, providing smooth
    transitions in semantic space compared to linear interpolation.

    Args:
        v0: Starting vector (normalized)
        v1: Ending vector (normalized)
        t: Interpolation factor (0.0 = v0, 1.0 = v1)

    Returns:
        Interpolated vector (normalized)

    Example:
        >>> v0 = np.array([1, 0, 0])
        >>> v1 = np.array([0, 1, 0])
        >>> v_mid = slerp(v0, v1, 0.5)
    """
    # Ensure inputs are normalized
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    # Compute dot product (cosine of angle)
    dot = np.dot(v0, v1)

    # Clamp to avoid numerical errors
    dot = np.clip(dot, -1.0, 1.0)

    # Calculate angle
    theta = np.arccos(dot) * t

    # Calculate orthogonal vector
    v2 = v1 - v0 * dot
    v2 = v2 / np.linalg.norm(v2)

    # Interpolate
    result = v0 * np.cos(theta) + v2 * np.sin(theta)

    return result


def lerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two vectors.

    Simpler than SLERP but may not preserve semantic relationships
    as well in high-dimensional spaces.

    Args:
        v0: Starting vector
        v1: Ending vector
        t: Interpolation factor (0.0 = v0, 1.0 = v1)

    Returns:
        Interpolated vector (normalized)
    """
    result = (1 - t) * v0 + t * v1
    return result / np.linalg.norm(result)


class ConceptInterpolator:
    """Interpolate between semantic concepts to discover intermediate ideas.

    This class enables "conceptual blending" - finding images that represent
    the transition between two different concepts (e.g., "vintage" → "futuristic"
    might reveal "steampunk" aesthetics).

    Example:
        >>> interpolator = ConceptInterpolator(engine.clip, engine.index)
        >>> path = interpolator.interpolate("cat", "tiger", steps=5)
        >>> for point in path:
        ...     print(f"{point.description}: {point.neighbors[0]['metadata']['name']}")

    Attributes:
        encoder: Function to encode text/images to embeddings
        searcher: Function to search index with embedding
        method: Interpolation method ("slerp" or "lerp")
    """

    def __init__(
        self,
        encoder: Callable,
        searcher: Callable,
        method: str = "slerp",
    ):
        """Initialize interpolator.

        Args:
            encoder: Function that takes text/image and returns embedding
            searcher: Function that takes embedding and returns search results
            method: "slerp" (default) or "lerp"
        """
        self.encoder = encoder
        self.searcher = searcher
        self.method = method
        self._interp_fn = slerp if method == "slerp" else lerp

    def interpolate(
        self,
        concept_a: str,
        concept_b: str,
        steps: int = 5,
        top_k: int = 3,
    ) -> list[InterpolationPoint]:
        """Create interpolation path between two concepts.

        Args:
            concept_a: Starting concept (text description)
            concept_b: Ending concept (text description)
            steps: Number of interpolation steps (including endpoints)
            top_k: Number of neighbors to find at each step

        Returns:
            List of InterpolationPoint along the path
        """
        # Encode concepts
        emb_a = self.encoder([concept_a])
        emb_b = self.encoder([concept_b])

        # Handle batch dimension
        if emb_a.ndim > 1:
            emb_a = emb_a[0]
        if emb_b.ndim > 1:
            emb_b = emb_b[0]

        results = []

        # Generate interpolation steps
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0

            # Interpolate
            emb_interp = self._interp_fn(emb_a, emb_b, t)

            # Create description
            if i == 0:
                desc = f"100% {concept_a}"
            elif i == steps - 1:
                desc = f"100% {concept_b}"
            else:
                pct_a = int((1 - t) * 100)
                pct_b = int(t * 100)
                desc = f"{pct_a}% {concept_a} + {pct_b}% {concept_b}"

            # Search for neighbors
            neighbors = self.searcher(emb_interp, top_k=top_k)

            point = InterpolationPoint(
                step=t,
                description=desc,
                embedding=emb_interp,
                neighbors=neighbors,
            )
            results.append(point)

        return results

    def find_interesting_intermediates(
        self,
        concept_a: str,
        concept_b: str,
        n_candidates: int = 20,
        diversity_threshold: float = 0.3,
    ) -> list[InterpolationPoint]:
        """Find semantically interesting intermediate points.

        Instead of evenly spaced steps, this method finds points along
        the interpolation that have diverse, high-quality neighbors.

        Args:
            concept_a: Starting concept
            concept_b: Ending concept
            n_candidates: Number of candidate points to sample
            diversity_threshold: Minimum diversity score for selection

        Returns:
            List of interesting InterpolationPoint (sparse)
        """
        # Get dense samples
        candidates = self.interpolate(concept_a, concept_b, steps=n_candidates, top_k=5)

        selected = [candidates[0]]  # Always include start

        for i, candidate in enumerate(candidates[1:-1], 1):
            # Calculate diversity of neighbors
            neighbors = candidate.neighbors
            if len(neighbors) < 2:
                continue

            # Simple diversity: variance of scores
            scores = [n["score"] for n in neighbors]
            score_variance = np.var(scores)

            # Check if this point offers something new compared to selected
            is_diverse = True
            for sel in selected:
                sim = np.dot(candidate.embedding, sel.embedding)
                if sim > 0.95:  # Too similar to already selected
                    is_diverse = False
                    break

            if is_diverse and score_variance > diversity_threshold:
                selected.append(candidate)

        selected.append(candidates[-1])  # Always include end

        return selected

    def visualize_path(
        self,
        path: list[InterpolationPoint],
        method: str = "pca",
    ) -> np.ndarray:
        """Reduce path embeddings to 2D for visualization.

        Args:
            path: List of interpolation points
            method: Dimensionality reduction method ("pca" or "tsne")

        Returns:
            2D coordinates array of shape (len(path), 2)
        """
        embeddings = np.array([p.embedding for p in path])

        if method == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
        elif method == "tsne":
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2, perplexity=min(len(path) - 1, 5))
            coords = tsne.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

        return coords


class MultiConceptBlend:
    """Blend multiple concepts together.

    Extends interpolation to more than two concepts, enabling
    complex semantic mixtures like "80% vintage + 15% neon + 5% grunge".
    """

    def __init__(self, encoder: Callable, searcher: Callable):
        """Initialize blender.

        Args:
            encoder: Encoding function
            searcher: Search function
        """
        self.encoder = encoder
        self.searcher = searcher

    def blend(
        self,
        concepts: list[tuple],
        top_k: int = 5,
    ) -> list[dict]:
        """Blend multiple concepts with weights.

        Args:
            concepts: List of (concept, weight) tuples
            top_k: Number of results to return

        Returns:
            Search results matching the blend
        """
        # Normalize weights
        total_weight = sum(w for _, w in concepts)
        normalized = [(c, w / total_weight) for c, w in concepts]

        # Encode all concepts
        concept_texts = [c for c, _ in normalized]
        embeddings = self.encoder(concept_texts)

        # Weighted combination
        result_emb = np.zeros(embeddings.shape[1])
        for (_, weight), emb in zip(normalized, embeddings):
            result_emb += weight * emb

        # Normalize
        result_emb = result_emb / np.linalg.norm(result_emb)

        # Search
        return self.searcher(result_emb, top_k=top_k)
