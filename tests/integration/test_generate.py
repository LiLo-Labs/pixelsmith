"""Integration tests for end-to-end pixel art generation (requires GPU)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU required")


@pytest.fixture(autouse=True)
def _unload_after():
    """Ensure pipeline is freed after each test."""
    yield
    from pixelsmith._pipeline import unload_pipeline

    unload_pipeline()


class TestGenerate:
    def test_basic_generation(self):
        from pixelsmith import generate

        img = generate("a small red mushroom", size=32, seed=42)
        assert img.size == (32, 32)
        assert img.mode == "RGB"

    def test_with_palette(self):
        from pixelsmith import PICO8, generate

        img = generate("a blue slime", size=64, palette="pico8", seed=123)
        arr = np.array(img)
        palette_colors = {tuple(c) for c in PICO8.as_array()}
        unique = {tuple(c) for c in arr.reshape(-1, 3)}
        assert unique <= palette_colors

    def test_seed_reproducibility(self):
        from pixelsmith import generate

        img1 = generate("a tiny sword", size=32, seed=999)
        img2 = generate("a tiny sword", size=32, seed=999)
        assert np.array_equal(np.array(img1), np.array(img2))

    def test_custom_config(self):
        from pixelsmith import GenerationConfig, generate

        cfg = GenerationConfig(num_inference_steps=15, guidance_scale=5.0)
        img = generate("a green tree", size=48, seed=7, config=cfg)
        assert img.size == (48, 48)
