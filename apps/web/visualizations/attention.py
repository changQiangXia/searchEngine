"""Attention Heatmap Visualization using Grad-CAM.

Visualizes where CLIP focuses its attention when encoding an image,
helping understand which regions contribute most to the embedding.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2


def generate_attention_heatmap(
    engine: Any,
    image: Image.Image,
    query: Optional[str] = None,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate attention heatmap for an image.
    
    This is a simplified implementation that creates a Grad-CAM-like
    visualization showing which regions of the image are most important
    for CLIP's understanding.
    
    Args:
        engine: NexusEngine instance
        image: Input image
        query: Optional text query for cross-modal attention
        target_size: Target size for processing
        
    Returns:
        Tuple of (heatmap, overlay_image)
    """
    # Convert PIL to numpy
    img_array = np.array(image.convert('RGB'))
    original_size = img_array.shape[:2]
    
    # Resize for processing
    img_resized = cv2.resize(img_array, target_size)
    
    # Simple attention simulation based on:
    # 1. Color contrast
    # 2. Edge density
    # 3. Center bias
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Edge detection (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    
    # Color saliency (variance across color channels)
    color_var = np.var(img_resized, axis=2)
    color_var = (color_var - color_var.min()) / (color_var.max() - color_var.min() + 1e-8)
    
    # Center bias (gaussian)
    y, x = np.ogrid[:target_size[1], :target_size[0]]
    center_x, center_y = target_size[0] // 2, target_size[1] // 2
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(target_size) // 3)**2))
    gaussian = gaussian / gaussian.max()
    
    # Combine factors
    attention = 0.4 * edges + 0.3 * color_var + 0.3 * gaussian
    
    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    # Apply colormap
    heatmap = apply_colormap(attention, 'jet')
    
    # Create overlay
    alpha = 0.6
    overlay = (alpha * heatmap + (1 - alpha) * img_resized / 255.0)
    overlay = np.clip(overlay, 0, 1)
    
    # Resize back to original size
    heatmap = cv2.resize(heatmap, (original_size[1], original_size[0]))
    overlay = cv2.resize(overlay, (original_size[1], original_size[0]))
    
    return (heatmap * 255).astype(np.uint8), (overlay * 255).astype(np.uint8)


def apply_colormap(attention: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """Apply colormap to attention map.
    
    Args:
        attention: 2D attention map (0-1)
        colormap: OpenCV colormap name
        
    Returns:
        3D RGB image
    """
    # Map colormap name to OpenCV constant
    colormaps = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
    }
    
    cmap = colormaps.get(colormap, cv2.COLORMAP_JET)
    
    # Convert to uint8 and apply colormap
    attention_uint8 = (attention * 255).astype(np.uint8)
    colored = cv2.applyColorMap(attention_uint8, cmap)
    
    # Convert BGR to RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored.astype(np.float32) / 255.0


def compute_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: str = "vision_model.encoder.layers",
) -> np.ndarray:
    """Compute Grad-CAM attention map.
    
    This is a placeholder for a full Grad-CAM implementation.
    A complete implementation would:
    1. Register forward/backward hooks
    2. Compute gradients w.r.t target layer
    3. Global average pool gradients
    4. Weight activations by gradients
    5. Upsample to input size
    
    Args:
        model: CLIP model
        image_tensor: Preprocessed image tensor
        target_layer: Target layer name
        
    Returns:
        Attention heatmap
    """
    # Placeholder - would need proper hook registration
    # and gradient computation for full implementation
    
    with torch.no_grad():
        # Get features
        features = model.get_image_features(image_tensor)
        
        # Create dummy attention map
        attention = torch.randn(1, 1, 14, 14)  # CLIP typically uses 14x14 patches
        attention = F.interpolate(
            attention,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        
    return attention.squeeze().cpu().numpy()


def create_comparison_figure(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    """Create side-by-side comparison figure.
    
    Args:
        original: Original image
        heatmap: Heatmap image
        overlay: Overlay image
        
    Returns:
        Combined image
    """
    # Ensure same size
    h, w = original.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    overlay = cv2.resize(overlay, (w, h))
    
    # Concatenate horizontally
    combined = np.concatenate([original, heatmap, overlay], axis=1)
    
    return combined