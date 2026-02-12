"""Tests for palette definitions and lookup."""

from __future__ import annotations

import numpy as np
import pytest

from pixelsmith._palettes import (
    C64,
    GAMEBOY,
    NES,
    PICO8,
    Palette,
    get_palette,
    resolve_palette,
)
from pixelsmith.exceptions import PaletteError


class TestPaletteDefinitions:
    def test_nes_has_40_colors(self):
        assert len(NES.colors) == 40

    def test_gameboy_has_4_colors(self):
        assert len(GAMEBOY.colors) == 4

    def test_pico8_has_16_colors(self):
        assert len(PICO8.colors) == 16

    def test_c64_has_16_colors(self):
        assert len(C64.colors) == 16

    def test_all_colors_in_valid_range(self):
        for palette in [NES, GAMEBOY, PICO8, C64]:
            for r, g, b in palette.colors:
                assert 0 <= r <= 255
                assert 0 <= g <= 255
                assert 0 <= b <= 255

    def test_as_array_shape(self):
        arr = PICO8.as_array()
        assert arr.shape == (16, 3)
        assert arr.dtype == np.uint8


class TestGetPalette:
    @pytest.mark.parametrize("name", ["nes", "NES", "Nes"])
    def test_case_insensitive(self, name: str):
        assert get_palette(name) is NES

    @pytest.mark.parametrize("name", ["gameboy", "game_boy", "game-boy"])
    def test_strip_separators(self, name: str):
        assert get_palette(name) is GAMEBOY

    def test_unknown_palette_raises(self):
        with pytest.raises(PaletteError, match="Unknown palette"):
            get_palette("sega_genesis")


class TestResolvePalette:
    def test_none_returns_none(self):
        assert resolve_palette(None) is None

    def test_string_resolves(self):
        assert resolve_palette("pico8") is PICO8

    def test_palette_passthrough(self):
        custom = Palette("custom", ((0, 0, 0), (255, 255, 255)))
        assert resolve_palette(custom) is custom
