"""pixelsmith — Generate 8-bit pixel art from text prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pixelsmith._config import GenerationConfig
from pixelsmith._palettes import C64, GAMEBOY, NES, PICO8, Palette, resolve_palette
from pixelsmith._pipeline import run_pipeline, unload_pipeline
from pixelsmith._postprocess import downscale as _downscale
from pixelsmith._postprocess import quantize_palette as _quantize
from pixelsmith.exceptions import GenerationError, ModelLoadError, PaletteError, PixelsmithError

if TYPE_CHECKING:
    from PIL import Image

__version__ = "0.1.0"
__all__ = [
    "C64",
    "GAMEBOY",
    "GenerationConfig",
    "GenerationError",
    "ModelLoadError",
    "NES",
    "PICO8",
    "Palette",
    "PaletteError",
    "PixelsmithError",
    "downscale",
    "generate",
    "quantize_palette",
    "unload_pipeline",
]

_DEFAULT_NEGATIVE = "3d render, realistic, blurry, photograph, smooth shading"


def generate(
    prompt: str,
    *,
    size: int = 64,
    negative_prompt: str = _DEFAULT_NEGATIVE,
    palette: str | Palette | None = None,
    seed: int | None = None,
    config: GenerationConfig | None = None,
) -> Image.Image:
    """Generate pixel art from a text prompt.

    Args:
        prompt: Text description of the desired image.
        size: Output pixel dimensions (square). Default 64.
        negative_prompt: Things to avoid in the generation.
        palette: Optional palette name ("nes", "gameboy", "pico8", "c64") or Palette object.
        seed: Optional seed for reproducibility.
        config: Optional GenerationConfig for advanced settings.

    Returns:
        PIL Image with the generated pixel art.
    """
    cfg = config or GenerationConfig()
    resolved_pal = resolve_palette(palette)

    raw = run_pipeline(
        prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        config=cfg,
    )

    result = _downscale(raw, size)

    if resolved_pal is not None:
        result = _quantize(result, resolved_pal)

    return result


def downscale(image: Image.Image, size: int) -> Image.Image:
    """Nearest-neighbor downscale to a square target size."""
    return _downscale(image, size)


def quantize_palette(image: Image.Image, palette: str | Palette) -> Image.Image:
    """Snap every pixel to the nearest color in the given palette."""
    resolved = resolve_palette(palette)
    if resolved is None:
        msg = "palette cannot be None for quantize_palette()"
        raise PaletteError(msg)
    return _quantize(image, resolved)
