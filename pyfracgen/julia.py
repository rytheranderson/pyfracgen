from typing import Iterable, Iterator, Sequence

from numba import jit
from numpy import log

from pyfracgen.common import CanvasBounded, Result
from pyfracgen.iterfuncs.funcs import power
from pyfracgen.types import Bound, IterFunc, Lattice


@jit  # type: ignore[misc]
def _julia_paint(
    c: float,
    xvals: Sequence[float],
    yvals: Sequence[float],
    lattice: Lattice,
    update_func: IterFunc,
    maxiter: int,
    horizon: float,
    log_smooth: bool,
) -> None:

    logh = log(log(horizon)) / log(2)
    for iy, yval in enumerate(yvals):
        for ix, xval in enumerate(xvals):
            z = xval + 1j * yval
            for it in range(maxiter):
                az = abs(z)
                if az > horizon:
                    if log_smooth:
                        lattice[iy, ix] = it - log(log(az)) / log(2) + logh
                    else:
                        lattice[iy, ix] = it
                    break
                z = update_func(z, c)


class Julia(CanvasBounded):
    def paint(
        self,
        c: complex,
        update_func: IterFunc,
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
    cvals: Iterable[complex],
    xbound: Bound,
    ybound: Bound,
    update_func: IterFunc = power,
    width: int = 5,
    height: int = 4,
    dpi: int = 300,
    maxiter: int = 100,
    horizon: float = 2.0**40,
    log_smooth: bool = True,
) -> Iterator[Result]:

    for c in cvals:
        canvas = Julia(width, height, dpi, xbound, ybound)
        canvas.paint(c, update_func, maxiter, horizon, log_smooth)
        yield canvas.result
