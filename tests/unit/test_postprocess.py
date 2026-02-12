"""Tests for post-processing functions."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixelsmith._palettes import GAMEBOY, Palette
from pixelsmith._postprocess import downscale, quantize_palette


class TestDownscale:
    def test_downscale_dimensions(self):
        img = Image.new("RGB", (1024, 1024), "red")
        result = downscale(img, 64)
        assert result.size == (64, 64)

    def test_downscale_preserves_solid_color(self):
        img = Image.new("RGB", (256, 256), (42, 100, 200))
        result = downscale(img, 32)
        arr = np.array(result)
        assert np.all(arr[:, :, 0] == 42)
        assert np.all(arr[:, :, 1] == 100)
        assert np.all(arr[:, :, 2] == 200)

    def test_downscale_uses_nearest_neighbor(self):
        """Nearest-neighbor should not blend colors — only exact source pixels."""
        img = Image.new("RGB", (4, 4))
        pixels = img.load()
        # Top-left quadrant red, rest blue
        for y in range(4):
            for x in range(4):
                pixels[x, y] = (255, 0, 0) if x < 2 and y < 2 else (0, 0, 255)

        result = downscale(img, 2)
        arr = np.array(result)
        # Top-left pixel should be red, bottom-right blue — no blending
        assert tuple(arr[0, 0]) == (255, 0, 0)
        assert tuple(arr[1, 1]) == (0, 0, 255)


class TestQuantizePalette:
    def test_pure_black_maps_to_darkest(self):
        img = Image.new("RGB", (8, 8), (0, 0, 0))
        result = quantize_palette(img, GAMEBOY)
        arr = np.array(result)
        # Darkest Game Boy color is (15, 56, 15)
        assert tuple(arr[0, 0]) == (15, 56, 15)

    def test_pure_white_maps_to_lightest(self):
        img = Image.new("RGB", (8, 8), (255, 255, 255))
        result = quantize_palette(img, GAMEBOY)
        arr = np.array(result)
        # Lightest Game Boy color is (155, 188, 15)
        assert tuple(arr[0, 0]) == (155, 188, 15)

    def test_output_only_contains_palette_colors(self):
        # Random noise image
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")

        palette = Palette("bw", ((0, 0, 0), (255, 255, 255)))
        result = quantize_palette(img, palette)
        result_arr = np.array(result)

        unique = {tuple(c) for c in result_arr.reshape(-1, 3)}
        assert unique <= {(0, 0, 0), (255, 255, 255)}

    def test_preserves_image_dimensions(self):
        img = Image.new("RGB", (32, 48), (128, 128, 128))
        result = quantize_palette(img, GAMEBOY)
        assert result.size == (32, 48)

    def test_rgba_input_converted_to_rgb(self):
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        result = quantize_palette(img, GAMEBOY)
        assert result.mode == "RGB"
