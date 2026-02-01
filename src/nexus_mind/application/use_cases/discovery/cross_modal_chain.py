"""Cross-Modal Chain Reasoning - Explore semantic paths through modalities.

This module implements chain-of-thought reasoning across modalities:
Image → Text → Image → Text → ...
Enabling discovery of semantic associations and creative exploration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class ModalityType(Enum):
    """Types of modalities in the chain."""

    IMAGE = auto()
    TEXT = auto()


@dataclass
class ChainNode:
    """A single node in the cross-modal chain.

    Attributes:
        step: Step number in the chain
        modality: Type of modality (IMAGE or TEXT)
        content: The actual content (image path or text string)
        embedding: Vector representation
        description: Human-readable description
        metadata: Additional metadata
    """

    step: int
    modality: ModalityType
    content: str | Image.Image
    embedding: np.ndarray
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainLink:
    """A transition between two nodes.

    Attributes:
        from_node: Source node
        to_node: Target node
        transition_type: How the transition was made
        similarity: Similarity score between nodes
    """

    from_node: ChainNode
    to_node: ChainNode
    transition_type: str
    similarity: float


class CrossModalChain:
    """Build and explore cross-modal reasoning chains.

    A chain starts with an image or text, then alternates:
    - Image → Generate caption → Search similar images
    - Text → Search images → Generate caption of result

    This creates a "semantic journey" exploring conceptual associations.

    Example:
        >>> chain = CrossModalChain(engine)
        >>> result = chain.explore(
        ...     start_image="cat.jpg",
        ...     steps=4,
        ...     strategy="diverse"
        ... )
        >>> for node in result.nodes:
        ...     print(f"{node.modality.name}: {node.description}")
    """

    def __init__(
        self,
        engine: Any,
        caption_generator: Callable | None = None,
    ):
        """Initialize chain explorer.

        Args:
            engine: NexusEngine instance
            caption_generator: Optional function to generate captions from images
        """
        self.engine = engine
        self.caption_generator = caption_generator
        self.nodes: list[ChainNode] = []
        self.links: list[ChainLink] = []

    def explore_from_image(
        self,
        image_path: str | Path | Image.Image,
        steps: int = 4,
        strategy: str = "diverse",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Start exploration from an image.

        Args:
            image_path: Starting image
            steps: Number of chain steps
            strategy: "diverse", "similar", or "random"
            top_k: Number of candidates at each step

        Returns:
            Chain result with nodes and links
        """
        self.nodes = []
        self.links = []

        # Step 0: Encode starting image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        emb = self.engine.clip.encode_images([image])

        node0 = ChainNode(
            step=0,
            modality=ModalityType.IMAGE,
            content=image,
            embedding=emb[0],
            description="Starting image",
            metadata={"source": "user_input"},
        )
        self.nodes.append(node0)

        # Build chain
        for step in range(1, steps + 1):
            prev_node = self.nodes[-1]

            if prev_node.modality == ModalityType.IMAGE:
                # Image → Text (generate caption)
                next_node = self._image_to_text(prev_node, step, strategy)
            else:
                # Text → Image (search)
                next_node = self._text_to_image(prev_node, step, strategy, top_k)

            if next_node is None:
                break

            # Create link
            similarity = float(
                np.dot(
                    prev_node.embedding / np.linalg.norm(prev_node.embedding),
                    next_node.embedding / np.linalg.norm(next_node.embedding),
                )
            )

            link = ChainLink(
                from_node=prev_node,
                to_node=next_node,
                transition_type="caption" if prev_node.modality == ModalityType.IMAGE else "search",
                similarity=similarity,
            )

            self.nodes.append(next_node)
            self.links.append(link)

        return self._format_result()

    def explore_from_text(
        self,
        text: str,
        steps: int = 4,
        strategy: str = "diverse",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Start exploration from text.

        Args:
            text: Starting text query
            steps: Number of chain steps
            strategy: Exploration strategy
            top_k: Number of candidates

        Returns:
            Chain result
        """
        self.nodes = []
        self.links = []

        # Step 0: Encode starting text
        emb = self.engine.clip.encode_text([text])

        node0 = ChainNode(
            step=0,
            modality=ModalityType.TEXT,
            content=text,
            embedding=emb[0],
            description=text,
            metadata={"source": "user_input"},
        )
        self.nodes.append(node0)

        # Build chain
        for step in range(1, steps + 1):
            prev_node = self.nodes[-1]

            if prev_node.modality == ModalityType.TEXT:
                # Text → Image (search)
                next_node = self._text_to_image(prev_node, step, strategy, top_k)
            else:
                # Image → Text (caption)
                next_node = self._image_to_text(prev_node, step, strategy)

            if next_node is None:
                break

            # Create link
            similarity = float(
                np.dot(
                    prev_node.embedding / np.linalg.norm(prev_node.embedding),
                    next_node.embedding / np.linalg.norm(next_node.embedding),
                )
            )

            link = ChainLink(
                from_node=prev_node,
                to_node=next_node,
                transition_type="search" if prev_node.modality == ModalityType.TEXT else "caption",
                similarity=similarity,
            )

            self.nodes.append(next_node)
            self.links.append(link)

        return self._format_result()

    def _image_to_text(
        self,
        image_node: ChainNode,
        step: int,
        strategy: str,
    ) -> ChainNode | None:
        """Generate caption from image."""
        # If we have a caption generator, use it
        if self.caption_generator:
            caption = self.caption_generator(image_node.content)
        else:
            # Fallback: use CLIP to find similar text
            # In practice, this would use BLIP/LLaVA
            caption = self._simulate_caption(image_node)

        emb = self.engine.clip.encode_text([caption])

        return ChainNode(
            step=step,
            modality=ModalityType.TEXT,
            content=caption,
            embedding=emb[0],
            description=f"Caption: {caption}",
            metadata={"generated": True},
        )

    def _text_to_image(
        self,
        text_node: ChainNode,
        step: int,
        strategy: str,
        top_k: int,
    ) -> ChainNode | None:
        """Search image from text."""
        if self.engine.index is None:
            return None

        # Search for images
        results = self.engine.index.search_with_metadata(
            text_node.embedding.reshape(1, -1),
            top_k=top_k,
        )

        if not results:
            return None

        # Select based on strategy
        if strategy == "diverse" and len(results) > 1:
            # Pick something different from previous
            idx = min(step % len(results), len(results) - 1)
        elif strategy == "random":
            idx = np.random.randint(0, len(results))
        else:  # similar (default)
            idx = 0

        result = results[idx]
        img_path = result["metadata"].get("path", "")

        try:
            image = Image.open(img_path).convert("RGB")
            emb = self.engine.clip.encode_images([image])

            return ChainNode(
                step=step,
                modality=ModalityType.IMAGE,
                content=image,
                embedding=emb[0],
                description=f"Image: {Path(img_path).name}",
                metadata={
                    "path": img_path,
                    "score": result["score"],
                    "search_query": text_node.description,
                },
            )
        except Exception:
            return None

    def _simulate_caption(self, image_node: ChainNode) -> str:
        """Simulate caption generation using CLIP similarity.

        In a real implementation, this would use BLIP-2 or LLaVA.
        For now, search for best matching concept.
        """
        # Search with common descriptive templates
        templates = [
            "a photo of a {}",
            "an image showing {}",
            "a picture of {}",
            "{} in the scene",
        ]

        concepts = ["object", "scene", "person", "landscape", "building", "animal", "vehicle"]

        best_score = -1
        best_caption = "an image"

        for concept in concepts:
            for template in templates:
                text = template.format(concept)
                emb = self.engine.clip.encode_text([text])

                score = float(
                    np.dot(
                        image_node.embedding / np.linalg.norm(image_node.embedding),
                        emb[0] / np.linalg.norm(emb[0]),
                    )
                )

                if score > best_score:
                    best_score = score
                    best_caption = text

        return best_caption

    def _format_result(self) -> dict[str, Any]:
        """Format chain result for output."""
        return {
            "nodes": [
                {
                    "step": n.step,
                    "modality": n.modality.name,
                    "description": n.description,
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
            "links": [
                {
                    "from_step": l.from_node.step,
                    "to_step": l.to_node.step,
                    "transition": l.transition_type,
                    "similarity": l.similarity,
                }
                for l in self.links
            ],
            "length": len(self.nodes),
            "path": " → ".join([n.modality.name for n in self.nodes]),
        }


class SemanticPathFinder:
    """Find semantic paths between two concepts through intermediate steps.

    Similar to cross-modal chain but with explicit start and end goals.
    """

    def __init__(self, engine: Any):
        """Initialize path finder.

        Args:
            engine: NexusEngine instance
        """
        self.engine = engine

    def find_path(
        self,
        start: str,
        end: str,
        max_hops: int = 3,
        min_similarity: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find semantic path from start to end concept.

        Args:
            start: Starting concept (text)
            end: Target concept (text)
            max_hops: Maximum intermediate steps
            min_similarity: Minimum similarity threshold

        Returns:
            List of path steps
        """
        # Encode endpoints
        start_emb = self.engine.clip.encode_text([start])[0]
        end_emb = self.engine.clip.encode_text([end])[0]

        path = [{"step": 0, "concept": start, "type": "start"}]

        current_emb = start_emb

        for hop in range(1, max_hops + 1):
            # Interpolate towards end
            t = hop / (max_hops + 1)
            target_emb = (1 - t) * current_emb + t * end_emb
            target_emb = target_emb / np.linalg.norm(target_emb)

            # Search for intermediate concept
            if self.engine.index:
                results = self.engine.index.search_with_metadata(
                    target_emb.reshape(1, -1),
                    top_k=1,
                )

                if results and results[0]["score"] >= min_similarity:
                    intermediate = results[0]["metadata"].get("name", f"step_{hop}")
                    path.append(
                        {
                            "step": hop,
                            "concept": intermediate,
                            "type": "intermediate",
                            "similarity": results[0]["score"],
                        }
                    )

                    # Update current
                    current_emb = target_emb

            # Check if we're close enough to end
            similarity_to_end = float(
                np.dot(
                    current_emb / np.linalg.norm(current_emb),
                    end_emb / np.linalg.norm(end_emb),
                )
            )

            if similarity_to_end > 0.9:
                break

        path.append({"step": len(path), "concept": end, "type": "end"})

        return path
