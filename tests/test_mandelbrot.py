"""Tests for the mandelbrot module."""

from hashlib import md5
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from matplotlib import colormaps
from matplotlib import pyplot as plt

from pyfracgen.images.images import image
from pyfracgen.iterfuncs.funcs import power
from pyfracgen.mandelbrot import Mandelbrot, mandelbrot

MAXITER = 1000


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
    """Test that a single point escapes or does not escape as expected.

    TODO: Add escape criteria and citation.

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
        (-2.2, 0.5), (-1, 1), power, width=4, height=3, dpi=100, maxiter=MAXITER
    )
    image(res, cmap=colormaps["binary"])
    result_path = tmp_path / "mandelbrot_integration_answer.png"
    plt.savefig(result_path)
    with open(result_path, "rb") as res_file:
        result_bytes = res_file.read()
    with open(mandelbrot_integration_answer, "rb") as answer_file:
        answer_bytes = answer_file.read()
    assert md5(result_bytes).digest() == md5(answer_bytes).digest()
