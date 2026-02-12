"""Post-processing: downscale and palette quantization."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixelsmith._palettes import Palette


def downscale(image: Image.Image, size: int) -> Image.Image:
    """Nearest-neighbor downscale to a square target size."""
    return image.resize((size, size), Image.Resampling.NEAREST)


def quantize_palette(image: Image.Image, palette: Palette) -> Image.Image:
    """Snap every pixel to the nearest color in the palette (RGB Euclidean distance)."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)  # (H, W, 3)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)  # (H*W, 3)

    pal = palette.as_array().astype(np.float32)  # (N, 3)

    # Vectorized nearest-color: broadcast distance computation
    # pixels[:, None, :] is (H*W, 1, 3), pal[None, :, :] is (1, N, 3)
    diffs = pixels[:, None, :] - pal[None, :, :]  # (H*W, N, 3)
    dists = np.sum(diffs**2, axis=2)  # (H*W, N)
    nearest = np.argmin(dists, axis=1)  # (H*W,)

    result = pal[nearest].reshape(h, w, 3).astype(np.uint8)
    return Image.fromarray(result, "RGB")
