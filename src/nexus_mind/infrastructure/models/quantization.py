"""Model Quantization Support - INT8/INT4 for memory efficiency.

Provides quantization utilities to reduce model memory footprint
by 2-4x with minimal accuracy loss.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class QuantizationType(Enum):
    """Quantization precision levels."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        quant_type: Quantization precision
        device: Target device
        compute_dtype: Compute dtype (for mixed precision)
        use_double_quant: Use nested quantization for 4-bit
        quant_storage_dtype: Storage dtype for quantized weights
    """

    quant_type: QuantizationType = QuantizationType.FP16
    device: str = "cuda"
    compute_dtype: torch.dtype = torch.float16
    use_double_quant: bool = True
    quant_storage_dtype: torch.dtype = torch.uint8

    @property
    def memory_reduction(self) -> float:
        """Expected memory reduction factor."""
        reductions = {
            QuantizationType.FP16: 2.0,  # vs FP32
            QuantizationType.INT8: 4.0,  # vs FP32
            QuantizationType.INT4: 8.0,  # vs FP32
        }
        return reductions.get(self.quant_type, 2.0)


def quantize_clip_model(
    model: CLIPModel,
    config: QuantizationConfig,
) -> CLIPModel:
    """Quantize CLIP model to lower precision.

    Args:
        model: Original CLIP model
        config: Quantization configuration

    Returns:
        Quantized model
    """
    if config.quant_type == QuantizationType.FP16:
        # FP16 is straightforward
        return cast(CLIPModel, model.half())

    elif config.quant_type == QuantizationType.INT8:
        return quantize_int8(model, config)

    elif config.quant_type == QuantizationType.INT4:
        return quantize_int4(model, config)

    else:
        raise ValueError(f"Unsupported quantization type: {config.quant_type}")


def quantize_int8(
    model: CLIPModel,
    config: QuantizationConfig,
) -> CLIPModel:
    """Apply INT8 dynamic quantization.

    Note: INT8 quantization works best on CPU.
    For GPU, we use FP16 as fallback.
    """
    if config.device == "cuda":
        warnings.warn(
            "INT8 quantization on CUDA is limited. "
            "Using FP16 as fallback for better compatibility.",
            stacklevel=2,
        )
        return cast(CLIPModel, model.half())

    # Dynamic quantization for CPU
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    return cast(CLIPModel, quantized_model)


def quantize_int4(
    model: CLIPModel,
    config: QuantizationConfig,
) -> CLIPModel:
    """Apply INT4 quantization using bitsandbytes.

    This requires the bitsandbytes library.
    Falls back to INT8 if not available.
    """
    try:
        from bitsandbytes.nn import Linear4bit

        # Replace linear layers with 4-bit versions
        def replace_linear(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Create 4-bit linear layer
                    quantized = Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=config.compute_dtype,
                        compress_statistics=config.use_double_quant,
                    )
                    setattr(module, name, quantized)
                else:
                    replace_linear(child)

        replace_linear(model)

        return model

    except ImportError:
        warnings.warn(
            "bitsandbytes not installed. " "Falling back to FP16 quantization.", stacklevel=2
        )
        return cast(CLIPModel, model.half())


class QuantizedCLIPWrapper:
    """Wrapper for quantized CLIP model.

    Provides the same interface as CLIPWrapper but with
    quantization support for memory efficiency.

    Example:
        >>> config = QuantizationConfig(quant_type=QuantizationType.INT8)
        >>> wrapper = QuantizedCLIPWrapper(config=config)
        >>> embeddings = wrapper.encode_images(images)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        config: QuantizationConfig | None = None,
        device: str | None = None,
    ):
        """Initialize quantized CLIP wrapper.

        Args:
            model_name: CLIP model name
            config: Quantization config (default: FP16)
            device: Device override
        """
        self.model_name = model_name
        self.config = config or QuantizationConfig(
            quant_type=QuantizationType.FP16,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        if device:
            self.config.device = device

        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load and quantize model."""
        from transformers import CLIPProcessor

        print(f"Loading CLIP with {self.config.quant_type.value} quantization...")

        # Load model
        model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        # Apply quantization
        model = quantize_clip_model(model, self.config)

        # Move to device
        model = cast(CLIPModel, model.to(self.config.device))  # type: ignore[arg-type]
        model.eval()

        self.model = model

        # Print memory info
        if self.config.device == "cuda":
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            print(f"✅ Model loaded (GPU memory: {mem_allocated:.2f} GB)")
            print(f"   Quantization: {self.config.quant_type.value}")
            print(f"   Expected reduction: {self.config.memory_reduction:.1f}x")

    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode images (same interface as CLIPWrapper)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            assert self.processor is not None
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().float().numpy())

        import numpy as np

        return np.vstack(embeddings).astype("float32")

    def encode_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode text (same interface as CLIPWrapper)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        import numpy as np

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            assert self.processor is not None
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().float().numpy())

        return np.vstack(embeddings).astype("float32")


def estimate_quantized_memory(
    model_name: str = "openai/clip-vit-large-patch14",
    quant_type: QuantizationType = QuantizationType.FP16,
) -> dict[str, float]:
    """Estimate memory usage for different quantization levels.

    Args:
        model_name: CLIP model name
        quant_type: Quantization type

    Returns:
        Memory estimates in GB
    """
    # Base sizes (approximate)
    base_sizes = {
        "openai/clip-vit-base-patch32": 600e6,  # 600MB
        "openai/clip-vit-base-patch16": 600e6,
        "openai/clip-vit-large-patch14": 1500e6,  # 1.5GB
    }

    base_size = base_sizes.get(model_name, 1500e6)

    # FP32 baseline
    fp32_size = base_size

    # Quantized sizes
    quant_sizes = {
        QuantizationType.FP16: fp32_size / 2,
        QuantizationType.INT8: fp32_size / 4,
        QuantizationType.INT4: fp32_size / 8,
    }

    quantized_size = quant_sizes.get(quant_type, fp32_size / 2)

    return {
        "fp32_gb": fp32_size / 1e9,
        "quantized_gb": quantized_size / 1e9,
        "reduction_factor": fp32_size / quantized_size,
        "savings_gb": (fp32_size - quantized_size) / 1e9,
    }
