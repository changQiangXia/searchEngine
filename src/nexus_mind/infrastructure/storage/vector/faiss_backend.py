"""FAISS vector storage backend with hybrid GPU/CPU support.

Optimized for consumer GPUs (RTX 3080ti 12GB) with automatic
fallback to CPU for large indices.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from nexus_mind.infrastructure.memory.manager import (
    MemoryPressureLevel,
    get_memory_manager,
)


class FAISSBackend:
    """FAISS vector index backend with memory-aware GPU/CPU handling.

    This backend automatically selects the optimal index type based on
    dataset size and available GPU memory:

    - < 10K vectors: IndexFlatIP (exact, GPU if possible)
    - 10K - 1M vectors: IndexIVFFlat (fast, GPU if possible)
    - > 1M vectors: IndexIVFPQ (compressed) or CPU index

    The backend monitors GPU memory and automatically falls back to CPU
    if the index would exceed available VRAM.

    Example:
        >>> backend = FAISSBackend(dim=768)
        >>> backend.build(embeddings, metadata)
        >>> scores, indices = backend.search(query_vector, top_k=10)

    Attributes:
        dim: Dimensionality of vectors
        index: The underlying FAISS index
        index_type: Type of index being used
        use_gpu: Whether index is currently on GPU
        metadata: List of metadata dicts for each vector
    """

    # Memory estimates for different index types (bytes per vector)
    MEMORY_PER_VECTOR = {
        "flat": 4 * 768,  # 4 bytes per float
        "ivf": 4 * 768 * 1.1,  # ~10% overhead
        "ivfpq": 32,  # 32 bytes with m=32, nbits=8
    }

    def __init__(
        self,
        dim: int = 768,
        index_type: str = "auto",
        use_gpu: bool = True,
        gpu_id: int = 0,
    ) -> None:
        """Initialize FAISS backend.

        Args:
            dim: Vector dimensionality (768 for CLIP-L, 512 for CLIP-B)
            index_type: Index type ("auto", "flat", "ivf", "ivfpq", "hnsw")
            use_gpu: Whether to use GPU if available
            gpu_id: GPU device ID
        """
        self.dim = dim
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.gpu_id = gpu_id
        self.memory_manager = get_memory_manager()

        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._is_trained = False
        self._original_index_type = index_type  # Remember original preference

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
        force_cpu: bool = False,
    ) -> None:
        """Build index from embeddings.

        Args:
            embeddings: Array of shape (N, dim)
            metadata: Optional metadata for each vector
            force_cpu: Force CPU index even if GPU is available
        """
        n = len(embeddings)

        # Validate input
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {embeddings.shape[1]}")

        # Auto-select index type
        if self.index_type == "auto":
            self.index_type = self._select_index_type(n)

        print(f"Building {self.index_type} index for {n} vectors ({self.dim}D)...")

        # Create index with vector count for optimal parameter calculation
        self.index = self._create_index(self.index_type, n_vectors=n)

        # Train if needed
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            print("Training index...")
            self.index.train(embeddings)
            self._is_trained = True

        # Check if we should use GPU
        if self.use_gpu and not force_cpu:
            self._try_gpu_transfer(embeddings)

        # Add vectors
        print(f"Adding {n} vectors to index...")
        self.index.add(embeddings)

        # Store metadata
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = [{"id": i} for i in range(n)]

        # Print stats
        device = "GPU" if self.use_gpu and hasattr(self.index, "getDevice") else "CPU"
        print(f"✅ Index built: {self.index.ntotal} vectors on {device}")

    def _select_index_type(self, n: int) -> str:
        """Select optimal index type based on dataset size."""
        if n < 10000:
            return "flat"
        elif n < 1000000:
            return "ivf"
        else:
            return "ivfpq"

    def _create_index(self, index_type: str, n_vectors: int | None = None) -> faiss.Index:
        """Create FAISS index of specified type.

        Args:
            index_type: Type of index to create
            n_vectors: Number of vectors (for calculating IVF nlist)
        """
        if index_type == "flat":
            # Exact search with inner product
            return faiss.IndexFlatIP(self.dim)

        elif index_type == "ivf":
            # Inverted file index
            # Better nlist calculation: 4 * sqrt(n) for good recall/speed balance
            if n_vectors is None:
                n_vectors = 10000
            # nlist should be at least 100 for good clustering
            nlist = min(4096, max(100, int(4 * np.sqrt(n_vectors))))
            self._nlist = nlist  # Store for reference
            quantizer = faiss.IndexFlatIP(self.dim)
            return faiss.IndexIVFFlat(quantizer, self.dim, nlist)

        elif index_type == "ivfpq":
            # Product quantization for memory efficiency
            if n_vectors is None:
                n_vectors = 100000
            nlist = min(4096, max(100, int(4 * np.sqrt(n_vectors))))
            m = 32  # Number of subquantizers
            nbits = 8  # Bits per subquantizer
            quantizer = faiss.IndexFlatIP(self.dim)
            return faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits)

        elif index_type == "hnsw":
            # Hierarchical NSW graph
            return faiss.IndexHNSWFlat(self.dim, 32)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def _try_gpu_transfer(self, embeddings: np.ndarray) -> None:
        """Attempt to transfer index to GPU with memory check."""
        # Estimate memory needed
        n = len(embeddings)
        memory_per_vec = self.MEMORY_PER_VECTOR.get(self.index_type, 4 * self.dim)
        estimated_memory = n * memory_per_vec

        # Check current memory pressure
        pressure = self.memory_manager.check_pressure()

        if pressure in [MemoryPressureLevel.CRITICAL, MemoryPressureLevel.EMERGENCY]:
            print("⚠️  GPU memory pressure high, keeping index on CPU")
            self.use_gpu = False
            return

        # Check if we have enough memory
        if not self.memory_manager.ensure_available(int(estimated_memory), aggressive_cleanup=True):
            warnings.warn(
                f"Insufficient GPU memory for index ({estimated_memory/1e9:.2f}GB needed). "
                "Using CPU index.",
                ResourceWarning,
                stacklevel=2,
            )
            self.use_gpu = False
            return

        # Try GPU transfer
        try:
            res = faiss.StandardGpuResources()
            # Limit temporary memory to 2GB
            res.setTempMemory(2 * 1024 * 1024 * 1024)

            # Configure for 3080ti
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True  # Use FP16 for memory efficiency

            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index, co)
            print("✅ Index moved to GPU (FP16 mode)")

        except Exception as e:
            warnings.warn(f"GPU transfer failed: {e}. Using CPU.", RuntimeWarning, stacklevel=2)
            self.use_gpu = False

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        nprobe: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search index for nearest neighbors.

        Args:
            query: Query vector(s) of shape (D,) or (N, D)
            top_k: Number of results to return
            nprobe: Number of clusters to search (IVF only)

        Returns:
            Tuple of (scores, indices) arrays
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Ensure 2D array
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Ensure correct dtype
        if query.dtype != np.float32:
            query = query.astype("float32")

        # Set nprobe for IVF indexes
        if nprobe and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        # Search
        scores, indices = self.index.search(query, top_k)
        return scores, indices

    def search_with_metadata(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search and return results with metadata.

        Args:
            query: Query vector
            top_k: Number of results

        Returns:
            List of result dicts with score, index, and metadata
        """
        scores, indices = self.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx >= 0 and idx < len(self.metadata):
                results.append(
                    {
                        "score": float(score),
                        "index": int(idx),
                        "metadata": self.metadata[idx],
                    }
                )

        return results

    def batch_search(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        batch_size: int = 100,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Search multiple queries in batches.

        Args:
            queries: Query vectors of shape (N, D)
            top_k: Number of results per query
            batch_size: Batch size for processing

        Returns:
            List of (scores, indices) tuples
        """
        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            scores, indices = self.search(batch, top_k)
            results.append((scores, indices))
        return results

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add new vectors to existing index.

        Args:
            embeddings: New vectors to add
            metadata: Metadata for new vectors
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        self.index.add(embeddings)

        if metadata:
            self.metadata.extend(metadata)
        else:
            start_id = len(self.metadata)
            self.metadata.extend([{"id": start_id + i} for i in range(len(embeddings))])

    def save(self, path: str | Path) -> None:
        """Save index and metadata to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # If index is on GPU, move to CPU for saving
        index_to_save = self.index
        if self.use_gpu and hasattr(self.index, "getDevice"):
            print("Moving index to CPU for saving...")
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        # Save index
        faiss.write_index(index_to_save, str(path / "index.faiss"))

        # Save metadata
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Save config
        config = {
            "dim": self.dim,
            "index_type": self.index_type,
            "ntotal": int(self.index.ntotal) if self.index else 0,
            "use_gpu": self.use_gpu,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Index saved to {path}")

    def load(self, path: str | Path, use_gpu: bool | None = None) -> None:
        """Load index and metadata from disk.

        Args:
            path: Directory path to load from
            use_gpu: Override GPU usage (None = auto-detect)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")

        # Load index
        index_path = path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.dim = config.get("dim", self.dim)
                self.index_type = config.get("index_type", self.index_type)

        # Try GPU transfer if requested
        if use_gpu is not None:
            self.use_gpu = use_gpu

        if self.use_gpu and faiss.get_num_gpus() > 0:
            self._try_gpu_transfer(np.zeros((self.index.ntotal, self.dim), dtype="float32"))

        print(f"✅ Loaded index with {self.index.ntotal} vectors from {path}")

    @property
    def ntotal(self) -> int:
        """Get total number of vectors in index."""
        return self.index.ntotal if self.index else 0

    @property
    def nlist(self) -> int | None:
        """Get number of clusters for IVF index."""
        if self.index and hasattr(self.index, "nlist"):
            return int(self.index.nlist)  # type: ignore[return-value]
        return None

    def recommend_nprobe(self, target_recall: float = 0.95) -> int:
        """Recommend nprobe value for target recall.

        This is a heuristic based on empirical testing with clustered data.
        For accurate tuning for your specific dataset, use tools/nprobe_tuner.py.

        Empirical recall estimates (clustered data, 50K-100K vectors):
        - nprobe ~4% of nlist → ~50% recall
        - nprobe ~8% of nlist → ~60% recall
        - nprobe ~16% of nlist → ~70% recall
        - nprobe ~32% of nlist → ~80% recall
        - nprobe ~64% of nlist → ~95% recall
        - nprobe ~100% of nlist → ~100% recall

        Args:
            target_recall: Target recall value (0-1)

        Returns:
            Recommended nprobe value
        """
        if not self.index or not hasattr(self.index, "nlist"):
            return 1  # Not an IVF index

        nlist: int = int(self.index.nlist)  # type: ignore[arg-type]

        # Empirical mapping based on clustered data tests
        # Using higher percentages for conservative estimates
        if target_recall >= 0.99:
            pct_of_nlist = 0.80
        elif target_recall >= 0.95:
            pct_of_nlist = 0.64  # ~2/3 of clusters
        elif target_recall >= 0.90:
            pct_of_nlist = 0.48  # ~1/2 of clusters
        elif target_recall >= 0.80:
            pct_of_nlist = 0.32  # ~1/3 of clusters
        elif target_recall >= 0.70:
            pct_of_nlist = 0.16  # ~1/6 of clusters
        elif target_recall >= 0.60:
            pct_of_nlist = 0.08
        else:
            pct_of_nlist = 0.04

        recommended = max(1, int(nlist * pct_of_nlist))

        # Clamp to valid range
        recommended = max(1, min(recommended, nlist))

        return recommended

    def set_nprobe(self, nprobe: int) -> None:
        """Set nprobe for IVF index.

        Args:
            nprobe: Number of clusters to search
        """
        if self.index and hasattr(self.index, "nlist"):
            nprobe = max(1, min(nprobe, self.index.nlist))
            self.index.nprobe = nprobe
            print(f"Set nprobe={nprobe} (nlist={self.index.nlist})")
        else:
            print("Warning: Not an IVF index, nprobe has no effect")

    def __len__(self) -> int:
        """Get total number of vectors."""
        return self.ntotal
