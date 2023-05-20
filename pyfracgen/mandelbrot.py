from __future__ import annotations

from typing import Any, Sequence

from numba import jit
from numpy import log

from pyfracgen.common import Canvas, Result
from pyfracgen.types import Array64, Bound, UpdateFunc


@jit  # type: ignore[misc]
def _mandelbrot_paint(
    xvals: Sequence[float],
    yvals: Sequence[float],
    lattice: Array64,
    update_func: UpdateFunc,
    maxiter: int = 100,
    horizon: float = 2.0**40,
    log_smooth: bool = True,
) -> None:

    logh = log(log(horizon)) / log(2)
    for iy, yval in enumerate(yvals):
        for ix, xval in enumerate(xvals):
            c = xval + 1j * yval
            z = c
            for it in range(maxiter):
                az = abs(z)
                if az > horizon:
                    if log_smooth:
                        lattice[iy, ix] = it - log(log(az)) / log(2) + logh
                    else:
                        lattice[iy, ix] = it
                    break
                z = update_func(z, c)


class Mandelbrot(Canvas):
    def paint(self, **kwargs: Any) -> None:

        _mandelbrot_paint(
            self.xvals,
            self.yvals,
            self.lattice,
            **kwargs,
        )


def mandelbrot(
    xbound: Bound,
    ybound: Bound,
    update_func: UpdateFunc,
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 2.0**40,
    log_smooth: bool = True,
) -> Result:

    canvas = Mandelbrot(xbound, ybound, width, height, dpi)
    canvas.paint(
        update_func=update_func, maxiter=maxiter, horizon=horizon, log_smooth=log_smooth
    )
    return canvas.result
