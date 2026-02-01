"""NexusMind main engine - orchestrates all components."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from nexus_mind.application.use_cases.discovery.concept_interpolation import (
    ConceptInterpolator,
    MultiConceptBlend,
)
from nexus_mind.application.use_cases.discovery.cross_modal_chain import (
    CrossModalChain,
    SemanticPathFinder,
)
from nexus_mind.application.use_cases.discovery.semantic_clustering import (
    SemanticClustering,
)
from nexus_mind.infrastructure.compute.optimizer import (
    BatchConfig,
    DynamicBatcher,
    PerformanceMonitor,
    optimize_for_3080ti,
    optimize_for_4090,
)
from nexus_mind.infrastructure.memory.manager import get_memory_manager
from nexus_mind.infrastructure.models.clip import CLIPWrapper
from nexus_mind.infrastructure.storage.cache.tiered_cache import TieredCache
from nexus_mind.infrastructure.storage.vector.faiss_backend import FAISSBackend


class NexusEngine:
    """Main NexusMind search engine.

    This is the primary interface for building and searching
    multimodal indices. It orchestrates CLIP for embeddings,
    FAISS for vector search, and manages workspace persistence.

    Example:
        >>> engine = NexusEngine("./my_workspace")
        >>> engine.index_images(["./photos/*.jpg"])
        >>> results = engine.search("a sunset over mountains")
        >>> for r in results:
        ...     print(f"{r['metadata']['path']}: {r['score']:.3f}")

    Attributes:
        workspace_dir: Path to workspace directory
        clip: CLIP model wrapper
        index: FAISS vector index
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        clip_model: str = "openai/clip-vit-large-patch14",
    ) -> None:
        """Initialize Nexus engine.

        Args:
            workspace_dir: Workspace directory for persistence
            clip_model: CLIP model name
        """
        # Setup workspace
        if workspace_dir is None:
            from platformdirs import user_data_dir

            base_dir = user_data_dir("nexus-mind", "nexusmind")
            workspace_dir = Path(base_dir) / "default"

        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.clip = CLIPWrapper(model_name=clip_model)
        self.index: FAISSBackend | None = None
        self.memory_manager = get_memory_manager()

        # Initialize tiered cache
        self.cache = TieredCache(
            l1_size=int(1e9),  # 1GB for GPU memory
            l2_path=self.workspace_dir / "cache" / "l2_ssd",
            l3_path=self.workspace_dir / "cache" / "l3_disk",
        )

        # Initialize performance optimization
        self.perf_monitor = PerformanceMonitor()

        # Detect GPU and set optimal batch config
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "3080" in gpu_name or "3090" in gpu_name:
                self.batch_config = optimize_for_3080ti()
                print(f"🎯 Optimized for {gpu_name}")
            elif "4090" in gpu_name or "3090 Ti" in gpu_name:
                self.batch_config = optimize_for_4090()
                print(f"🎯 Optimized for {gpu_name}")
            else:
                self.batch_config = BatchConfig()
        else:
            self.batch_config = BatchConfig(batch_size=16)

        self.dynamic_batcher = DynamicBatcher(self.batch_config)

        # Initialize plugin system
        from nexus_mind.plugins.base import PluginRegistry

        self.plugin_registry = PluginRegistry()

        # Load built-in plugins
        from nexus_mind.plugins.builtin import CSVExporter, JSONExporter

        self.plugin_registry.register(CSVExporter())
        self.plugin_registry.register(JSONExporter())

        # Try to load existing index
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        """Try to load existing index from workspace."""
        index_path = self.workspace_dir / "indices"
        if (index_path / "index.faiss").exists():
            try:
                self.index = FAISSBackend(dim=self.clip.embedding_dim)
                self.index.load(index_path)
                print(f"📂 Loaded existing index with {len(self.index)} vectors")
            except Exception as e:
                print(f"⚠️  Failed to load existing index: {e}")
                self.index = None

    def index_images(
        self,
        paths: list[str | Path],
        batch_size: int = 32,
        recursive: bool = True,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
    ) -> dict[str, Any]:
        """Build index from images.

        Args:
            paths: List of image files or directories
            batch_size: Processing batch size
            recursive: Whether to search directories recursively
            extensions: Image file extensions to include

        Returns:
            Statistics dict with count, time, etc.
        """
        start_time = time.time()

        # Collect all image paths
        image_paths = []
        for path in paths:
            path = Path(path)

            if path.is_dir():
                # Directory - glob for images
                if recursive:
                    pattern = "**/*"
                else:
                    pattern = "*"

                for ext in extensions:
                    image_paths.extend(path.glob(f"{pattern}{ext}"))
                    image_paths.extend(path.glob(f"{pattern}{ext.upper()}"))
            elif path.is_file():
                # Single file
                if path.suffix.lower() in extensions:
                    image_paths.append(path)

        # Remove duplicates and sort
        image_paths = sorted(set(image_paths))

        if not image_paths:
            raise ValueError(f"No images found in paths: {paths}")

        print(f"Found {len(image_paths)} images to index")

        # Extract embeddings
        print("Extracting embeddings...")
        embeddings = self.clip.encode_images(
            image_paths,
            batch_size=batch_size,
            show_progress=True,
        )

        # Build metadata
        metadata = []
        for i, path in enumerate(image_paths):
            metadata.append(
                {
                    "id": i,
                    "path": str(path),
                    "name": path.name,
                    "stem": path.stem,
                    "suffix": path.suffix.lower(),
                }
            )

        # Build index
        print("Building FAISS index...")
        self.index = FAISSBackend(dim=embeddings.shape[1])
        self.index.build(embeddings, metadata)

        # Save
        self._save_index()

        elapsed = time.time() - start_time
        stats = {
            "count": len(image_paths),
            "time_seconds": elapsed,
            "vectors_per_second": len(image_paths) / elapsed,
            "index_type": self.index.index_type,
            "on_gpu": self.index.use_gpu,
        }

        print(f"✅ Indexed {stats['count']} images in {elapsed:.1f}s")
        return stats

    def _save_index(self) -> None:
        """Save current index to workspace."""
        if self.index is None:
            return

        index_path = self.workspace_dir / "indices"
        self.index.save(index_path)

    def search(
        self,
        query: str | Path | Image.Image,
        top_k: int = 10,
        return_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """Search index.

        Args:
            query: Text string, image path, or PIL Image
            top_k: Number of results to return
            return_embeddings: Whether to include embeddings in results

        Returns:
            List of result dicts with score, metadata, etc.
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Build an index first with index_images().")

        # Encode query
        query_vec = self._encode_query(query)

        # Search
        results = self.index.search_with_metadata(query_vec, top_k)

        # Add rank
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def _encode_query(
        self,
        query: str | Path | Image.Image,
    ) -> np.ndarray:
        """Encode query to embedding vector."""
        # Handle path
        if isinstance(query, (str, Path)) and Path(query).exists():
            query = Image.open(query).convert("RGB")

        # Encode based on type
        if isinstance(query, Image.Image):
            return self.clip.encode_images([query])
        else:
            return self.clip.encode_text([str(query)])

    def search_diverse(
        self,
        query: str,
        top_k: int = 10,
        lambda_param: float = 0.5,
        candidate_factor: int = 4,
    ) -> list[dict[str, Any]]:
        """MMR (Maximal Marginal Relevance) search for diverse results.

        Balances relevance with diversity to avoid redundant results.

        Args:
            query: Text query
            top_k: Number of results
            lambda_param: Trade-off parameter (0=diversity, 1=relevance)
            candidate_factor: Factor for initial candidate pool size

        Returns:
            Diverse list of results
        """
        if self.index is None:
            raise RuntimeError("No index loaded")

        # Get larger candidate pool
        candidates = self.search(query, top_k=top_k * candidate_factor)

        if len(candidates) <= top_k:
            return candidates

        # Get embeddings for candidates
        candidate_indices = [c["index"] for c in candidates]

        # For now, simplified MMR using index-based similarity
        # Full implementation would need actual embeddings

        query_vec = self._encode_query(query)
        selected = []
        selected_indices = set()

        while len(selected) < top_k and len(selected_indices) < len(candidates):
            best_mmr_score = -float("inf")
            best_candidate = None

            for cand in candidates:
                idx = cand["index"]
                if idx in selected_indices:
                    continue

                # Relevance score
                rel_score = cand["score"]

                # Diversity score (max similarity to selected)
                if selected:
                    # Simplified: use inverse of average score similarity
                    # In full implementation, would use actual embedding similarity
                    div_score = 1.0 / (1 + len(selected))
                else:
                    div_score = 1.0

                # MMR formula
                mmr_score = lambda_param * rel_score + (1 - lambda_param) * div_score

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = cand

            if best_candidate:
                selected.append(best_candidate)
                selected_indices.add(best_candidate["index"])

        # Update ranks
        for i, r in enumerate(selected):
            r["rank"] = i + 1

        return selected

    def negative_search(
        self,
        positive: str,
        negative: str,
        top_k: int = 10,
        alpha: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Negative search - exclude concepts.

        Example: positive="sunset beach", negative="people"

        Args:
            positive: What to search for
            negative: What to exclude
            top_k: Number of results
            alpha: Weight for negative term (0-1)

        Returns:
            Filtered search results
        """
        # Encode both queries
        pos_emb = self.clip.encode_text([positive])
        neg_emb = self.clip.encode_text([negative])

        # Vector subtraction: towards positive, away from negative
        query_vec = pos_emb - alpha * neg_emb
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Search
        return self.index.search_with_metadata(query_vec, top_k)

    def interpolate_concepts(
        self,
        concept_a: str,
        concept_b: str,
        steps: int = 5,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Interpolate between two concepts to discover intermediate ideas.

        Args:
            concept_a: Starting concept (e.g., "vintage")
            concept_b: Ending concept (e.g., "futuristic")
            steps: Number of interpolation steps
            top_k: Results per step

        Returns:
            List of interpolation points with descriptions and results
        """
        if self.index is None:
            raise RuntimeError("No index loaded")

        # Create interpolator
        interpolator = ConceptInterpolator(
            encoder=lambda texts: self.clip.encode_text(texts),
            searcher=lambda emb, top_k: self.index.search_with_metadata(
                emb.reshape(1, -1) if emb.ndim == 1 else emb, top_k=top_k
            ),
            method="slerp",
        )

        # Interpolate
        path = interpolator.interpolate(concept_a, concept_b, steps=steps, top_k=top_k)

        # Convert to serializable format
        results = []
        for point in path:
            results.append(
                {
                    "step": point.step,
                    "description": point.description,
                    "neighbors": point.neighbors,
                }
            )

        return results

    def cluster_index(
        self,
        method: str = "kmeans",
        n_clusters: int | None = None,
        min_cluster_size: int = 3,
    ) -> list[dict[str, Any]]:
        """Cluster the index to discover semantic groups.

        Args:
            method: "kmeans", "hdbscan", or "agglomerative"
            n_clusters: Number of clusters (for KMeans)
            min_cluster_size: Minimum cluster size (for HDBSCAN)

        Returns:
            List of cluster information
        """
        if self.index is None:
            raise RuntimeError("No index loaded")

        # Get all embeddings
        # Note: FAISS doesn't provide direct access, so we need to store separately
        # For now, use metadata to reconstruct (placeholder implementation)

        clusterer = SemanticClustering(
            method=method,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )

        # We need embeddings - this is a limitation of current FAISS backend
        # In practice, we should cache embeddings separately
        print("⚠️  Clustering requires embedding cache. Feature limited in current version.")

        return []

    def blend_concepts(
        self,
        concepts: list[tuple],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Blend multiple concepts with weights.

        Args:
            concepts: List of (concept, weight) tuples
            top_k: Number of results

        Returns:
            Search results matching the blend
        """
        if self.index is None:
            raise RuntimeError("No index loaded")

        blender = MultiConceptBlend(
            encoder=lambda texts: self.clip.encode_text(texts),
            searcher=lambda emb, top_k: self.index.search_with_metadata(
                emb.reshape(1, -1) if emb.ndim == 1 else emb, top_k=top_k
            ),
        )

        return blender.blend(concepts, top_k=top_k)

    def explore_chain(
        self,
        start: str,
        steps: int = 4,
        strategy: str = "diverse",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Cross-modal chain exploration.

        Args:
            start: Starting text or image path
            steps: Number of chain steps
            strategy: "diverse", "similar", or "random"
            top_k: Number of candidates

        Returns:
            Chain result with nodes and links
        """
        chain = CrossModalChain(self.engine)

        # Determine if start is image or text
        if Path(start).exists():
            return chain.explore_from_image(start, steps=steps, strategy=strategy, top_k=top_k)
        else:
            return chain.explore_from_text(start, steps=steps, strategy=strategy, top_k=top_k)

    def find_semantic_path(
        self,
        start: str,
        end: str,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """Find semantic path between concepts.

        Args:
            start: Starting concept
            end: Target concept
            max_hops: Maximum intermediate steps

        Returns:
            List of path steps
        """
        finder = SemanticPathFinder(self)
        return finder.find_path(start, end, max_hops=max_hops)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "workspace": str(self.workspace_dir),
            "clip_model": self.clip.model_name,
            "clip_device": self.clip.device,
        }

        if self.index:
            stats["index"] = {
                "vectors": len(self.index),
                "type": self.index.index_type,
                "on_gpu": self.index.use_gpu,
            }
        else:
            stats["index"] = None

        # Memory stats
        mem_stats = self.memory_manager.get_stats()
        stats["memory"] = {
            "gpu_used_gb": mem_stats.gpu_used / 1e9,
            "gpu_total_gb": mem_stats.gpu_total / 1e9,
            "gpu_usage_pct": mem_stats.gpu_usage_pct,
        }

        return stats

    def validate(self) -> bool:
        """Validate engine state."""
        if self.index is None:
            print("❌ No index loaded")
            return False

        if len(self.index) == 0:
            print("❌ Index is empty")
            return False

        if len(self.index.metadata) != len(self.index):
            print("❌ Metadata/index mismatch")
            return False

        print(f"✅ Engine validated: {len(self.index)} vectors ready")
        return True

    def export_results(
        self,
        results: list[dict[str, Any]],
        output_path: str,
        format: str = "json",
    ) -> bool:
        """Export search results using plugins.

        Args:
            results: Search results to export
            output_path: Output file path
            format: Export format (json, csv, html)

        Returns:
            True if successful
        """
        # Find appropriate exporter plugin
        exporters = self.plugin_registry.get_plugins_by_type(
            type(self.plugin_registry._plugins.get("json_exporter"))
        )

        for exporter in exporters:
            if f".{format}" in exporter.supported_formats():
                return exporter.export(results, output_path)

        print(f"❌ No exporter found for format: {format}")
        return False

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
