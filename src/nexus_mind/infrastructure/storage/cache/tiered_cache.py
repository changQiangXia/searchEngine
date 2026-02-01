"""Tiered Cache System - L1 (GPU) / L2 (SSD) / L3 (Disk).

Provides multi-level caching for embeddings and intermediate results,
optimizing for both speed and capacity.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    """Cache entry metadata.

    Attributes:
        key: Cache key
        level: Cache level (1, 2, or 3)
        size_bytes: Size of cached data
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        access_count: Number of accesses
        ttl: Time to live in seconds
    """

    key: str
    level: int
    size_bytes: int
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: int | None = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


class TieredCache:
    """Multi-level cache system.

    Levels:
    - L1: GPU memory (fastest, smallest)
    - L2: SSD/NVMe (fast, medium)
    - L3: HDD/Network (slow, largest)

    Example:
        >>> cache = TieredCache(
        ...     l1_size=1e9,  # 1GB GPU
        ...     l2_path="./cache/l2",
        ...     l3_path="./cache/l3",
        ... )
        >>> cache.put("embedding_1", data, level=1)
        >>> data = cache.get("embedding_1")  # From L1
    """

    def __init__(
        self,
        l1_size: int = int(1e9),  # 1GB default
        l2_path: str = "./data/cache/l2_ssd",
        l3_path: str = "./data/cache/l3_disk",
        l2_size: int = int(10e9),  # 10GB
        l3_size: int = int(100e9),  # 100GB
    ):
        """Initialize tiered cache.

        Args:
            l1_size: Maximum L1 cache size in bytes
            l2_path: Path for L2 (SSD) cache
            l3_path: Path for L3 (Disk) cache
            l2_size: Maximum L2 cache size
            l3_size: Maximum L3 cache size
        """
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size

        self.l2_path = Path(l2_path)
        self.l3_path = Path(l3_path)

        # Create directories
        self.l2_path.mkdir(parents=True, exist_ok=True)
        self.l3_path.mkdir(parents=True, exist_ok=True)

        # L1: In-memory cache
        self._l1_cache: dict[str, Any] = {}
        self._l1_metadata: dict[str, CacheEntry] = {}

        # L2/L3: Metadata tracking
        self._l2_metadata: dict[str, CacheEntry] = self._load_metadata(self.l2_path)
        self._l3_metadata: dict[str, CacheEntry] = self._load_metadata(self.l3_path)

        self._current_l1_size = 0

    def _load_metadata(self, path: Path) -> dict[str, CacheEntry]:
        """Load cache metadata from disk."""
        metadata_file = path / ".metadata.json"
        if not metadata_file.exists():
            return {}

        try:
            with open(metadata_file) as f:
                data = json.load(f)

            return {k: CacheEntry(**v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_metadata(self, path: Path, metadata: dict[str, CacheEntry]) -> None:
        """Save cache metadata to disk."""
        metadata_file = path / ".metadata.json"

        data = {
            k: {
                "key": v.key,
                "level": v.level,
                "size_bytes": v.size_bytes,
                "created_at": v.created_at,
                "accessed_at": v.accessed_at,
                "access_count": v.access_count,
                "ttl": v.ttl,
            }
            for k, v in metadata.items()
        }

        with open(metadata_file, "w") as f:
            json.dump(data, f)

    def _make_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            return hashlib.md5(data.tobytes()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()

    def _get_size(self, data: Any) -> int:
        """Get size of data in bytes."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (str, bytes)):
            return len(data)
        else:
            return len(str(data).encode())

    def put(
        self,
        key: str,
        data: Any,
        level: int = 2,
        ttl: int | None = None,
    ) -> bool:
        """Put data into cache.

        Args:
            key: Cache key
            data: Data to cache
            level: Cache level (1, 2, or 3)
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        size = self._get_size(data)

        entry = CacheEntry(
            key=key,
            level=level,
            size_bytes=size,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=0,
            ttl=ttl,
        )

        if level == 1:
            return self._put_l1(key, data, entry)
        elif level == 2:
            return self._put_l2(key, data, entry)
        elif level == 3:
            return self._put_l3(key, data, entry)
        else:
            raise ValueError(f"Invalid level: {level}")

    def _put_l1(self, key: str, data: Any, entry: CacheEntry) -> bool:
        """Put into L1 (memory) cache."""
        # Check if we need to evict
        while self._current_l1_size + entry.size_bytes > self.l1_size:
            if not self._evict_l1():
                break

        if entry.size_bytes > self.l1_size:
            # Too big for L1, try L2
            return self._put_l2(key, data, entry)

        self._l1_cache[key] = data
        self._l1_metadata[key] = entry
        self._current_l1_size += entry.size_bytes

        return True

    def _put_l2(self, key: str, data: Any, entry: CacheEntry) -> bool:
        """Put into L2 (SSD) cache."""
        path = self.l2_path / f"{key}.npy"

        # Save data
        try:
            if isinstance(data, np.ndarray):
                np.save(path, data)
            else:
                # Save as JSON for non-array data
                with open(path.with_suffix(".json"), "w") as f:
                    json.dump(data, f)

            self._l2_metadata[key] = entry
            self._save_metadata(self.l2_path, self._l2_metadata)

            return True
        except Exception as e:
            print(f"L2 cache write failed: {e}")
            return False

    def _put_l3(self, key: str, data: Any, entry: CacheEntry) -> bool:
        """Put into L3 (Disk) cache."""
        path = self.l3_path / f"{key}.npy"

        try:
            if isinstance(data, np.ndarray):
                np.save(path, data)
            else:
                with open(path.with_suffix(".json"), "w") as f:
                    json.dump(data, f)

            self._l3_metadata[key] = entry
            self._save_metadata(self.l3_path, self._l3_metadata)

            return True
        except Exception as e:
            print(f"L3 cache write failed: {e}")
            return False

    def _evict_l1(self) -> bool:
        """Evict least recently used item from L1."""
        if not self._l1_metadata:
            return False

        # Find LRU item
        lru_key = min(self._l1_metadata.keys(), key=lambda k: self._l1_metadata[k].accessed_at)

        entry = self._l1_metadata[lru_key]

        # Try to move to L2
        data = self._l1_cache[lru_key]
        if self._put_l2(lru_key, data, entry):
            # Remove from L1
            del self._l1_cache[lru_key]
            del self._l1_metadata[lru_key]
            self._current_l1_size -= entry.size_bytes
            return True

        return False

    def get(self, key: str) -> Any | None:
        """Get data from cache.

        Tries L1 -> L2 -> L3 in order.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        # Try L1 first
        if key in self._l1_cache:
            self._l1_metadata[key].touch()
            return self._l1_cache[key]

        # Try L2
        if key in self._l2_metadata:
            entry = self._l2_metadata[key]
            if not entry.is_expired():
                data = self._load_from_disk(self.l2_path, key)
                if data is not None:
                    # Promote to L1 if possible
                    self._promote_to_l1(key, data, entry)
                    entry.touch()
                    return data
            else:
                # Expired, remove
                self._remove_l2(key)

        # Try L3
        if key in self._l3_metadata:
            entry = self._l3_metadata[key]
            if not entry.is_expired():
                data = self._load_from_disk(self.l3_path, key)
                if data is not None:
                    # Promote to L2 (and possibly L1)
                    self._promote_to_l2(key, data, entry)
                    entry.touch()
                    return data
            else:
                self._remove_l3(key)

        return None

    def _load_from_disk(self, base_path: Path, key: str) -> Any | None:
        """Load data from disk cache."""
        npy_path = base_path / f"{key}.npy"
        json_path = base_path / f"{key}.json"

        try:
            if npy_path.exists():
                return np.load(npy_path)
            elif json_path.exists():
                with open(json_path) as f:
                    return json.load(f)
        except Exception as e:
            print(f"Cache load failed: {e}")

        return None

    def _promote_to_l1(self, key: str, data: Any, entry: CacheEntry) -> None:
        """Promote L2/L3 entry to L1."""
        new_entry = CacheEntry(
            key=key,
            level=1,
            size_bytes=entry.size_bytes,
            created_at=entry.created_at,
            accessed_at=time.time(),
            access_count=entry.access_count,
            ttl=entry.ttl,
        )
        self._put_l1(key, data, new_entry)

    def _promote_to_l2(self, key: str, data: Any, entry: CacheEntry) -> None:
        """Promote L3 entry to L2."""
        new_entry = CacheEntry(
            key=key,
            level=2,
            size_bytes=entry.size_bytes,
            created_at=entry.created_at,
            accessed_at=time.time(),
            access_count=entry.access_count,
            ttl=entry.ttl,
        )
        self._put_l2(key, data, new_entry)

    def _remove_l2(self, key: str) -> None:
        """Remove entry from L2."""
        path = self.l2_path / f"{key}"
        for ext in [".npy", ".json"]:
            if path.with_suffix(ext).exists():
                path.with_suffix(ext).unlink()

        if key in self._l2_metadata:
            del self._l2_metadata[key]
            self._save_metadata(self.l2_path, self._l2_metadata)

    def _remove_l3(self, key: str) -> None:
        """Remove entry from L3."""
        path = self.l3_path / f"{key}"
        for ext in [".npy", ".json"]:
            if path.with_suffix(ext).exists():
                path.with_suffix(ext).unlink()

        if key in self._l3_metadata:
            del self._l3_metadata[key]
            self._save_metadata(self.l3_path, self._l3_metadata)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "l1": {
                "items": len(self._l1_cache),
                "size_bytes": self._current_l1_size,
                "max_size_bytes": self.l1_size,
                "usage": self._current_l1_size / self.l1_size if self.l1_size > 0 else 0,
            },
            "l2": {
                "items": len(self._l2_metadata),
                "path": str(self.l2_path),
            },
            "l3": {
                "items": len(self._l3_metadata),
                "path": str(self.l3_path),
            },
        }

    def clear(self, level: int | None = None) -> None:
        """Clear cache.

        Args:
            level: Specific level to clear, or None for all
        """
        if level is None or level == 1:
            self._l1_cache.clear()
            self._l1_metadata.clear()
            self._current_l1_size = 0

        if level is None or level == 2:
            shutil.rmtree(self.l2_path, ignore_errors=True)
            self.l2_path.mkdir(parents=True, exist_ok=True)
            self._l2_metadata.clear()

        if level is None or level == 3:
            shutil.rmtree(self.l3_path, ignore_errors=True)
            self.l3_path.mkdir(parents=True, exist_ok=True)
            self._l3_metadata.clear()
