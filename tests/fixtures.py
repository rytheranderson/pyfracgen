"""Fixtures for testing pyfracgen."""
from pathlib import Path

import pytest

TEST_FILES = Path(__file__).parent / "files"


@pytest.fixture()
def mandelbrot_integration_answer() -> Path:
    return TEST_FILES / "mandelbrot_integration_answer.png"


@pytest.fixture()
def julia_integration_answer() -> Path:
    return TEST_FILES / "julia_integration_answer.png"
