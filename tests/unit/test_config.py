"""Tests for GenerationConfig."""

from __future__ import annotations

from pixelsmith._config import GenerationConfig


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.num_inference_steps == 30
        assert cfg.guidance_scale == 7.5
        assert cfg.lora_weight == 1.2
        assert cfg.render_size == 1024
        assert cfg.device == "auto"
        assert cfg.dtype == "float16"
        assert cfg.enable_cpu_offload is True

    def test_frozen(self):
        cfg = GenerationConfig()
        try:
            cfg.device = "cpu"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_custom_values(self):
        cfg = GenerationConfig(
            num_inference_steps=50,
            guidance_scale=5.0,
            device="cpu",
        )
        assert cfg.num_inference_steps == 50
        assert cfg.guidance_scale == 5.0
        assert cfg.device == "cpu"

    def test_resolved_device_explicit(self):
        cfg = GenerationConfig(device="cuda")
        assert cfg.resolved_device() == "cuda"

    def test_cache_dir_default_not_empty(self):
        cfg = GenerationConfig()
        assert cfg.cache_dir  # non-empty string
        assert "pixelsmith" in cfg.cache_dir
