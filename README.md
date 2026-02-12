# pixelsmith

Generate 8-bit pixel art from text prompts using SDXL + LoRA.

## Quick Start

```python
from pixelsmith import generate

img = generate("a tiny wizard casting a spell", size=64, palette="nes")
img.save("wizard.png")
```

## Installation

```bash
uv add pixelsmith
```

For MCP server support:

```bash
uv add "pixelsmith[mcp]"
```

## API

### `generate(prompt, *, size=64, negative_prompt=..., palette=None, seed=None, config=None)`

Generate pixel art from a text prompt.

- **prompt**: Text description of the image
- **size**: Output pixel dimensions (square), default 64
- **palette**: Optional palette name (`"nes"`, `"gameboy"`, `"pico8"`, `"c64"`) or `Palette` object
- **seed**: Optional seed for reproducibility
- **config**: Optional `GenerationConfig` for advanced settings

### `downscale(image, size)`

Nearest-neighbor downscale to target size.

### `quantize_palette(image, palette)`

Quantize image colors to a retro palette.

## MCP Server

Run as an MCP server:

```bash
uvx pixelsmith-mcp
```

Tools:
- `generate_pixel_art` — Generate pixel art from a prompt
- `quantize_to_palette` — Quantize an existing image to a retro palette
