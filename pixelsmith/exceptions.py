"""Exception hierarchy for pixelsmith."""


class PixelsmithError(Exception):
    """Base exception for all pixelsmith errors."""


class ModelLoadError(PixelsmithError):
    """Failed to load or download the diffusion model."""


class GenerationError(PixelsmithError):
    """Failed during image generation."""


class PaletteError(PixelsmithError):
    """Invalid palette name or configuration."""
