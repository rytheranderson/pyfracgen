"""Tests for the mandelbrot module."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib import colormaps
from matplotlib import pyplot as plt

from pyfracgen.images.images import image
from pyfracgen.iterfuncs.funcs import power
from pyfracgen.mandelbrot import Mandelbrot, _mandelbrot_paint, mandelbrot
from tests.assertions import assert_pngs_equal

MAXITER = 1000


def point_escapes(c: complex) -> bool:
    threshold = 2
    if abs(c) > threshold:
        return True
    z = c
    for _ in range(MAXITER):
        z = power(z, c)
        if abs(z) > threshold:
            return True
    return False


@pytest.mark.parametrize(
    "x, y, expected_color", [(3, 0, 0), (2, 0, 1), (1, 0, 2), (0.5, 0, 4)]
)
def test_mandelbrot_paint_colors_correctly(
    x: float, y: float, expected_color: float
) -> None:
    """Test _mandelbrot_paint assigns the expected color for a given point.

    Args:
        x: The point x-value (real part).
        y: The point y-value (imaginary part).
        expected_color: The expected color for the point (x, y).
    """
    lattice = np.zeros((1, 1), dtype=np.float64)
    _mandelbrot_paint(
        xvals=(x,),
        yvals=(y,),
        lattice=lattice,
        update_func=power,
        maxiter=MAXITER,
        horizon=2,
        log_smooth=False,
    )
    assert lattice[0, 0] == expected_color


@settings(deadline=None)
@given(c=st.complex_numbers(max_magnitude=4.0, allow_subnormal=False))
def test_single_point_escape(c: complex) -> None:
    """Test that a single point escapes or does not escape as expected.

    If any Q(z) > 2, where Q is the orbit of z, then z escapes.
    https://en.wikipedia.org/wiki/Mandelbrot_set#Basic_properties

    Args:
        c: The single point to test.
    """
    res = Mandelbrot(1, 1, 1, (c.real, c.real), (c.imag, c.imag))
    res.paint(power, MAXITER, 2.0**40, False)
    color = res.result.image_array[0, 0]
    if point_escapes(c):
        # No c^2 + c > the horizon => minimun 1 iteration required for escape
        assert 0.0 < color < MAXITER
    else:
        assert color == 0.0


def test_integration_mandelbrot_image_creation(
    mandelbrot_integration_answer: Path,
    tmp_path: Path,
) -> None:
    """Test mandelbrot image creation by comparing with an answer image.

    Args:
        mandelbrot_integration_answer: Path to the answer image.
        tmp_path: Temporary path where the result image is saved.
    """
    res = mandelbrot(
        (-2.2, 0.5), (-1, 1), power, width=4, height=3, dpi=50, maxiter=MAXITER
    )
    image(res, cmap=colormaps["binary"])
    result_path = tmp_path / "mandelbrot_integration_answer.png"
    plt.savefig(result_path)
    assert_pngs_equal(result_path, mandelbrot_integration_answer)
