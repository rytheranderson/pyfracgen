from __future__ import annotations

from typing import Iterator, Sequence

from numba import jit
from numpy import log

from pyfracgen.common import CanvasBounded, Result
from pyfracgen.types import Array64, Bound, UpdateFunc
from pyfracgen.updaters.funcs import power


@jit  # type: ignore[misc]
def _julia_paint(
    c: float,
    xvals: Sequence[float],
    yvals: Sequence[float],
    lattice: Array64,
    update_func: UpdateFunc,
    maxiter: int,
    horizon: float,
    log_smooth: bool,
) -> None:

    log_horizon = log(log(horizon)) / log(2)
    for iy, yval in enumerate(yvals):
        for ix, xval in enumerate(xvals):
            z = xval + 1j * yval
            for iteration in range(maxiter):
                az = abs(z)
                if az > horizon:
                    if log_smooth:
                        lattice[iy, ix] = (
                            iteration - log(log(az)) / log(2) + log_horizon
                        )
                    else:
                        lattice[iy, ix] = iteration
                    break
                z = update_func(z, c)


class Julia(CanvasBounded):
    def paint(
        self,
        c: complex,
        update_func: UpdateFunc,
        maxiter: int,
        horizon: float,
        log_smooth: bool,
    ) -> None:

        _julia_paint(
            c,
            self.xvals,
            self.yvals,
            self.lattice,
            update_func,
            maxiter,
            horizon,
            log_smooth,
        )


def julia(
    cvals: Sequence[complex],
    xbound: Bound,
    ybound: Bound,
    update_func: UpdateFunc = power,
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 2.0**40,
    log_smooth: bool = True,
) -> Iterator[Result]:

    for c in cvals:
        canvas = Julia(width, height, dpi, xbound, ybound)
        canvas.paint(c, update_func, maxiter, horizon, log_smooth)
        yield canvas.result
