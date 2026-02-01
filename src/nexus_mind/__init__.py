"""NexusMind - Next-gen multimodal semantic search engine.

A powerful AI search engine built on CLIP and FAISS, designed to run efficiently
on consumer GPUs like RTX 3080ti (12GB) with intelligent memory management.

Example:
    >>> from nexus_mind import NexusEngine
    >>> engine = NexusEngine()
    >>> engine.index_images(["./photos"])
    >>> results = engine.search("a cute cat")
"""

__version__ = "0.1.0"
__author__ = "NexusMind Team"

from nexus_mind.core.engine import NexusEngine
from nexus_mind.infrastructure.memory.manager import GPUMemoryManager, MemoryPressureLevel

__all__ = [
    "NexusEngine",
    "GPUMemoryManager",
    "MemoryPressureLevel",
    "__version__",
]