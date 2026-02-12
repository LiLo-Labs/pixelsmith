"""Built-in retro color palettes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from pixelsmith.exceptions import PaletteError


@dataclass(frozen=True, slots=True)
class Palette:
    """A named color palette for pixel art quantization."""

    name: str
    colors: tuple[tuple[int, int, int], ...]

    def as_array(self) -> NDArray[np.uint8]:
        """Return palette as (N, 3) uint8 array."""
        return np.array(self.colors, dtype=np.uint8)


# fmt: off
NES = Palette("nes", (
    (0, 0, 0), (252, 252, 252), (188, 188, 188), (124, 124, 124),
    (168, 16, 0), (228, 92, 16), (248, 56, 0), (228, 0, 88),
    (104, 68, 0), (172, 124, 0), (248, 184, 0), (248, 120, 88),
    (0, 120, 0), (0, 168, 0), (0, 168, 68), (88, 216, 84),
    (0, 0, 168), (0, 88, 248), (104, 136, 252), (0, 120, 248),
    (148, 0, 132), (216, 0, 204), (248, 120, 248), (120, 120, 248),
    (0, 88, 0), (0, 168, 0), (184, 248, 24), (172, 224, 0),
    (0, 64, 88), (0, 136, 136), (0, 232, 216), (88, 248, 152),
    (248, 164, 0), (232, 208, 124), (248, 216, 168), (248, 184, 108),
    (44, 44, 44), (116, 116, 116), (188, 188, 188), (252, 252, 252),
))

GAMEBOY = Palette("gameboy", (
    (15, 56, 15),
    (48, 98, 48),
    (139, 172, 15),
    (155, 188, 15),
))

PICO8 = Palette("pico8", (
    (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
    (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
    (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
    (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170),
))

C64 = Palette("c64", (
    (0, 0, 0), (255, 255, 255), (136, 0, 0), (170, 255, 238),
    (204, 68, 204), (0, 204, 85), (0, 0, 170), (238, 238, 119),
    (221, 136, 85), (102, 68, 0), (255, 119, 119), (51, 51, 51),
    (119, 119, 119), (170, 255, 102), (0, 136, 255), (187, 187, 187),
))
# fmt: on

_BUILTIN: ClassVar[dict[str, Palette]] = {
    "nes": NES,
    "gameboy": GAMEBOY,
    "pico8": PICO8,
    "c64": C64,
}


def get_palette(name: str) -> Palette:
    """Look up a built-in palette by name (case-insensitive)."""
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in _BUILTIN:
        valid = ", ".join(sorted(_BUILTIN))
        raise PaletteError(f"Unknown palette {name!r}. Available: {valid}")
    return _BUILTIN[key]


def resolve_palette(palette: str | Palette | None) -> Palette | None:
    """Accept a palette name string, Palette object, or None."""
    if palette is None:
        return None
    if isinstance(palette, Palette):
        return palette
    return get_palette(palette)
