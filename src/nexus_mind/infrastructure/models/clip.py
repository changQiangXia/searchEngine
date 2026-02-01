"""CLIP model wrapper with memory management and 3080ti optimizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from nexus_mind.infrastructure.memory.manager import (
    MemoryPressureLevel,
    get_memory_manager,
)


class CLIPWrapper:
    """CLIP model wrapper with automatic memory management.

    This wrapper provides a unified interface for encoding images and text
    using OpenAI's CLIP model. It includes automatic memory management for
    GPUs with limited VRAM (e.g., RTX 3080ti with 12GB).

    The wrapper automatically:
    - Uses FP16 precision on GPU to save memory
    - Falls back to CPU if GPU OOM
    - Monitors memory pressure during batch processing
    - Automatically adjusts batch size if needed

    Example:
        >>> clip = CLIPWrapper()
        >>> embeddings = clip.encode_images(["photo1.jpg", "photo2.jpg"])
        >>> text_emb = clip.encode_text(["a cute cat"])

    Attributes:
        model: The CLIP model instance
        processor: The CLIP processor for preprocessing
        device: Current device ("cuda" or "cpu")
        model_name: Name of the CLIP model being used
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize CLIP wrapper.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("cuda", "cpu", or None for auto)
            dtype: Data type for model weights (None for auto)
        """
        self.model_name = model_name
        self.memory_manager = get_memory_manager()

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine dtype based on device
        if dtype is None:
            if self.device == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load CLIP model with memory safety."""
        print(f"Loading CLIP model: {self.model_name}")

        def loader() -> tuple[CLIPModel, CLIPProcessor]:
            model = CLIPModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=None,  # We'll handle device placement
            )
            processor = CLIPProcessor.from_pretrained(self.model_name)
            return model, processor

        try:
            # CLIP-L is ~1.5GB in FP16
            estimated_memory = 1.5e9 if self.dtype == torch.float16 else 3e9

            loaded_model, loaded_processor = self.memory_manager.load_safe(
                loader,
                model_name="clip",
                estimated_memory=int(estimated_memory),
                persistent=True,  # CLIP is core model, keep in GPU
            )
            self.model = loaded_model
            self.processor = loaded_processor

            # Ensure correct device placement
            # Cast to Any to bypass mypy confusion about the wrapped type
            _model: Any = self.model
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = _model.cuda()
            else:
                self.device = "cpu"
                self.model = _model.cpu()

            self.model.eval()
            print(f"✅ CLIP loaded on {self.device} with {self.dtype}")

        except Exception as e:
            print(f"❌ Failed to load CLIP: {e}")
            raise

    def encode_images(
        self,
        images: list[str | Path | Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode images to embedding vectors.

        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (N, D) where D is embedding dimension (768 for CLIP-L)
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        embeddings = []

        # Adjust batch size based on memory pressure
        if self.memory_manager.check_pressure() == MemoryPressureLevel.WARNING:
            batch_size = max(8, batch_size // 2)
        elif self.memory_manager.check_pressure() == MemoryPressureLevel.CRITICAL:
            batch_size = 4

        iterator = range(0, len(images), batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Encoding images", total=len(images) // batch_size + 1)

        for i in iterator:
            batch = images[i : i + batch_size]

            # Load and preprocess images
            processed_images = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert("RGB")
                processed_images.append(img)

            # Process with CLIP
            inputs = self.processor(
                images=processed_images,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

            # Encode with memory safety
            try:
                with torch.no_grad():
                    # Use pixel_values directly for CLIP
                    pixel_values = inputs["pixel_values"]
                    image_features = self.model.get_image_features(pixel_values=pixel_values)
                    # Ensure it's a tensor (not a model output object)
                    if hasattr(image_features, "pooler_output"):
                        image_features = image_features.pooler_output
                    elif hasattr(image_features, "last_hidden_state"):
                        image_features = image_features.last_hidden_state[:, 0, :]
                    # Normalize
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embeddings.append(image_features.cpu().float().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Emergency fallback: process one by one
                    self.memory_manager.auto_clean(aggressive=True)
                    for img in processed_images:
                        single_emb = self._encode_single_image(img)
                        embeddings.append(single_emb)
                else:
                    raise

        return np.vstack(embeddings).astype("float32")

    def _encode_single_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single image (fallback for OOM)."""
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not loaded")

        inputs = self.processor(images=image, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            pixel_values = inputs["pixel_values"]
            features = self.model.get_image_features(pixel_values=pixel_values)
            # Handle model output object if needed
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "last_hidden_state"):
                features = features.last_hidden_state[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)

        return cast(np.ndarray, features.cpu().float().numpy())

    def encode_text(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text queries to embedding vectors.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (N, D) where D is embedding dimension
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Encoding text")

        for i in iterator:
            batch = texts[i : i + batch_size]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # CLIP's max context length
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                text_features = self.model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                # Handle model output object if needed
                if hasattr(text_features, "pooler_output"):
                    text_features = text_features.pooler_output
                elif hasattr(text_features, "last_hidden_state"):
                    text_features = text_features.last_hidden_state[:, 0, :]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.cpu().float().numpy())

        return np.vstack(embeddings).astype("float32")

    def encode(
        self,
        data: str | Image.Image | Path | list,
    ) -> np.ndarray:
        """Universal encoding interface.

        Automatically detects input type and encodes accordingly.

        Args:
            data: Text string, image path, PIL Image, or list of either

        Returns:
            Embedding vector(s)
        """
        # Handle single items
        if not isinstance(data, list):
            data = [data]

        if len(data) == 0:
            raise ValueError("Empty input")

        # Detect type from first element
        first = data[0]

        if isinstance(first, str):
            # Could be text or file path
            if Path(first).exists():
                return self.encode_images(data)
            else:
                return self.encode_text([str(d) for d in data])

        elif isinstance(first, (Path, Image.Image)):
            return self.encode_images(data)

        else:
            raise ValueError(f"Unsupported input type: {type(first)}")

    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings (N, D)
            embeddings2: Second set of embeddings (M, D)

        Returns:
            Similarity matrix (N, M)
        """
        # Normalize if needed
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        return cast(np.ndarray, np.dot(embeddings1, embeddings2.T))

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        # CLIP-L is 768, CLIP-B is 512
        if "large" in self.model_name:
            return 768
        elif "base" in self.model_name:
            return 512
        else:
            return 768  # Default
