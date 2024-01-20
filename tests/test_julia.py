"""Tests for the julia module."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib import colormaps
from matplotlib import pyplot as plt

from pyfracgen.images.images import image
from pyfracgen.iterfuncs.funcs import power
from pyfracgen.julia import Julia, _julia_paint, julia
from tests.assertions import assert_pngs_equal

MAXITER = 1000


def point_escapes(z: complex, c: complex) -> bool:
    crit = max([abs(c), 2])
    if abs(z) > crit:
        return True
    for _ in range(MAXITER):
        z = power(z, c)
        if abs(z) > crit:
            return True
    return False


@pytest.mark.parametrize(
    "x, y, c, expected_color",
    [
        (3, 0, 0, 0),
        (2, 0, 0, 1),
        (1, 0, 1, 2),
        (0.5, 0, 0.5, 4),
        (0.5, 0, 0.3, 8),
        (0.3, 0, 0.3, 11),
    ],
)
def test_julia_paint_colors_correctly(
    x: float, y: float, c: complex, expected_color: float
) -> None:
    """Test _julia_paint assigns the expected color for a given point.

    Args:
        x: The point x-value (real part).
        y: The point y-value (imaginary part).
        expected_color: The expected color for the point (x, y).
    """
    lattice = np.zeros((1, 1), dtype=np.float64)
    _julia_paint(
        c=c,
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
@given(
    z=st.complex_numbers(max_magnitude=4.0, allow_subnormal=False),
    c=st.complex_numbers(max_magnitude=4.0, allow_subnormal=False),
)
def test_single_point_escape(z: complex, c: complex) -> None:
    """Test that a single point escapes or does not escape as expected.

    If any Q(z) > max(2, abs(c)), where Q is the orbit of z, then z escapes.
    https://www.marksmath.org/classes/Spring2019ComplexDynamics/text/section-filled_julia_set.html  # noqa: E501

    Args:
        c: The single point to test.
    """
    res = Julia(1, 1, 1, (z.real, z.real), (z.imag, z.imag))
    res.paint(c, power, MAXITER, 2.0**40, False)
    color = res.result.image_array[0, 0]
    if point_escapes(z, c):
        # No c^2 + c > the horizon => minimun 1 iteration required for escape
        assert 0.0 < color < MAXITER
    else:
        assert color == 0.0


def test_integration_julia_image_creation(
    julia_integration_answer: Path,
    tmp_path: Path,
) -> None:
    """Test julia image creation by comparing with an answer image.

    Args:
        julia_integration_answer: Path to the answer image.
        tmp_path: Temporary path where the result image is saved.
    """
    res = julia(
        [-0.8 + 0.156j],
        (-1.6, 1.6),
        (-0.9, 0.9),
        power,
        width=4,
        height=3,
        dpi=50,
        maxiter=MAXITER,
    )
    image(next(res), cmap=colormaps["binary"])
    result_path = tmp_path / "julia_integration_answer.png"
    plt.savefig(result_path)
    assert_pngs_equal(result_path, julia_integration_answer)
