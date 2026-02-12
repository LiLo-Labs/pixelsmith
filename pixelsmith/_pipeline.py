"""Diffusers pipeline loading, caching, and inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pixelsmith._config import GenerationConfig
from pixelsmith.exceptions import GenerationError, ModelLoadError

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
_LORA_REPO = "nerijs/pixel-art-xl"

# Module-level pipeline cache
_cached_pipeline: object | None = None
_cached_config: GenerationConfig | None = None


def _resolve_torch_dtype(dtype_str: str):  # noqa: ANN202
    """Convert string dtype to torch dtype."""
    import torch

    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[
        dtype_str
    ]


def _load_pipeline(config: GenerationConfig):  # noqa: ANN202
    """Load SDXL base + pixel-art-xl LoRA, with caching."""
    global _cached_pipeline, _cached_config  # noqa: PLW0603

    if _cached_pipeline is not None and _cached_config == config:
        return _cached_pipeline

    try:
        import torch  # noqa: F401 — required by diffusers at runtime
        from diffusers import StableDiffusionXLPipeline

        dtype = _resolve_torch_dtype(config.dtype)
        device = config.resolved_device()

        logger.info("Loading SDXL base model from %s", _BASE_MODEL)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            _BASE_MODEL,
            torch_dtype=dtype,
            cache_dir=config.cache_dir,
            use_safetensors=True,
        )

        logger.info("Loading LoRA weights from %s", _LORA_REPO)
        pipe.load_lora_weights(_LORA_REPO, cache_dir=config.cache_dir)
        pipe.fuse_lora(lora_scale=config.lora_weight)

        if config.enable_cpu_offload and device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        _cached_pipeline = pipe
        _cached_config = config
        return pipe

    except Exception as exc:
        raise ModelLoadError(f"Failed to load pipeline: {exc}") from exc


def run_pipeline(
    prompt: str,
    *,
    negative_prompt: str,
    seed: int | None = None,
    config: GenerationConfig,
) -> Image.Image:
    """Run the SDXL + LoRA pipeline and return the raw generated image."""
    pipe = _load_pipeline(config)

    try:
        import torch

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            width=config.render_size,
            height=config.render_size,
            generator=generator,
        )
        return result.images[0]

    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc


def unload_pipeline() -> None:
    """Free the cached pipeline and GPU memory."""
    global _cached_pipeline, _cached_config  # noqa: PLW0603
    _cached_pipeline = None
    _cached_config = None

    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
