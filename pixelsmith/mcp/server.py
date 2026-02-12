"""MCP server exposing pixel art generation tools."""

from __future__ import annotations

import io
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage

mcp = FastMCP("pixelsmith")


def _generate_pixel_art(
    prompt: str,
    size: int = 64,
    palette: str | None = None,
    seed: int | None = None,
) -> MCPImage:
    """Generate pixel art from a text prompt.

    Args:
        prompt: Description of the pixel art to generate.
        size: Output pixel dimensions (square). Default 64.
        palette: Optional retro palette: "nes", "gameboy", "pico8", "c64".
        seed: Optional seed for reproducibility.

    Returns:
        Generated pixel art as a PNG image.
    """
    from pixelsmith import generate

    img = generate(prompt, size=size, palette=palette, seed=seed)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return MCPImage(data=buf.getvalue(), format="png")


def _quantize_to_palette(
    image_path: str,
    palette: str,
    size: int | None = None,
) -> MCPImage:
    """Quantize an existing image to a retro color palette.

    Args:
        image_path: Path to the source image file.
        palette: Retro palette name: "nes", "gameboy", "pico8", "c64".
        size: Optional target size to downscale to.

    Returns:
        Quantized image as a PNG.
    """
    from PIL import Image

    from pixelsmith import downscale, quantize_palette

    img = Image.open(Path(image_path))

    if size is not None:
        img = downscale(img, size)

    result = quantize_palette(img, palette)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return MCPImage(data=buf.getvalue(), format="png")


# Register tools with MCP server (names without underscore prefix)
mcp.tool(name="generate_pixel_art")(_generate_pixel_art)
mcp.tool(name="quantize_to_palette")(_quantize_to_palette)


def run() -> None:
    """Entry point for the pixelsmith-mcp console script."""
    mcp.run()
