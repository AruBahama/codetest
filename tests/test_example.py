"""Test suite for example functions."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.example import add  # noqa: E402 pylint: disable=wrong-import-position


def test_add():
    """Ensure add returns correct results."""
    assert add(1, 2) == 3
