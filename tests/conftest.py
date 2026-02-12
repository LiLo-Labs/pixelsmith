"""Shared test fixtures."""

from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    """Free GPU memory after each test that might load models."""
    yield
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
