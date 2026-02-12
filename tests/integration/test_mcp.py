"""Integration tests for the MCP server tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


def _make_test_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a random test image at the given path."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


class TestQuantizeToPalette:
    """quantize_to_palette works without GPU — only needs PIL."""

    def test_quantize_existing_image(self):
        from pixelsmith.mcp.server import _quantize_to_palette

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _make_test_image(img_path)

            result = _quantize_to_palette(
                image_path=str(img_path),
                palette="gameboy",
            )
            # MCPImage has data attribute with PNG bytes
            assert result.data is not None
            assert len(result.data) > 0

    def test_quantize_with_downscale(self):
        from pixelsmith.mcp.server import _quantize_to_palette

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _make_test_image(img_path, size=(256, 256))

            result = _quantize_to_palette(
                image_path=str(img_path),
                palette="pico8",
                size=32,
            )
            assert result.data is not None


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU required")
class TestGeneratePixelArt:
    @pytest.fixture(autouse=True)
    def _unload_after(self):
        yield
        from pixelsmith._pipeline import unload_pipeline

        unload_pipeline()

    def test_generate_returns_image(self):
        from pixelsmith.mcp.server import _generate_pixel_art

        result = _generate_pixel_art(prompt="a tiny red gem", size=32, seed=42)
        assert result.data is not None
        assert len(result.data) > 100  # Non-trivial PNG


class TestMCPServerDefinition:
    """Verify the MCP server is properly configured (no GPU needed)."""

    def test_server_has_tools(self):
        from pixelsmith.mcp.server import mcp

        assert mcp.name == "pixelsmith"

    def test_generate_tool_callable(self):
        from pixelsmith.mcp.server import _generate_pixel_art

        assert callable(_generate_pixel_art)

    def test_quantize_tool_callable(self):
        from pixelsmith.mcp.server import _quantize_to_palette

        assert callable(_quantize_to_palette)
