"""Generation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from platformdirs import user_cache_dir


def _default_cache_dir() -> str:
    return user_cache_dir("pixelsmith")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Settings for the SDXL + LoRA pixel art pipeline."""

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    lora_weight: float = 1.2
    render_size: int = 1024
    device: str = "auto"  # "auto" | "cuda" | "cpu" | "mps"
    dtype: str = "float16"
    enable_cpu_offload: bool = True
    cache_dir: str = field(default_factory=_default_cache_dir)

    def resolved_device(self) -> str:
        """Return the actual device string, resolving 'auto'."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
