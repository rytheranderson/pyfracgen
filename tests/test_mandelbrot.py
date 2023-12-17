"""Tests for the mandelbrot module."""

from hypothesis import given
from hypothesis import strategies as st

from pyfracgen.iterfuncs.funcs import power
from pyfracgen.mandelbrot import Mandelbrot

MAXITER = 1000
HORIZON = 2.0**40


def point_escapes(c: complex) -> bool:
    if abs(c) > 2:
        return True
    z = c
    for _ in range(MAXITER):
        z = power(z, c)
        if abs(z) > 2:
            return True
    return False


@given(c=st.complex_numbers(max_magnitude=4.0, allow_subnormal=False))
def test_single_point_escape(c: complex) -> None:
    """Test that a point (complex) escapes or does not escape as expected.

    TODO: Add escape criteria and citation.

    Args:
        c: The single point to test.
    """
    mandelbrot = Mandelbrot(1, 1, 1, (c.real, c.real), (c.imag, c.imag))
    mandelbrot.paint(power, MAXITER, HORIZON, False)
    color = mandelbrot.result.image_array[0, 0]
    if point_escapes(c):
        # No c^2 + c > the horizon => minimun 1 iteration required for escape
        assert 0.0 < color < MAXITER
    else:
        assert color == 0.0


def test_points_outside_mandelbrot_are_colored() -> None:
    """Test points outside the Mandelbrot set are colored."""
    mandelbrot = Mandelbrot(1, 1, 100, (4.0, 5.0), (4.0, 5.0))
    mandelbrot.paint(power, MAXITER, HORIZON, False)
    assert mandelbrot.result.image_array.all()


def test_points_inside_mandelbrot_are_not_colored() -> None:
    """Test points in the Mandelbrot set are not colored."""
    mandelbrot = Mandelbrot(1, 1, 100, (-0.2, 0.2), (-0.2, 0.2))
    mandelbrot.paint(power, MAXITER, 2, False)
    assert not mandelbrot.result.image_array.any()
