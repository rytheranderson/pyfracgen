from typing import Sequence

import numpy as np
from numba import jit

from pyfracgen.common import CanvasBounded, Result
from pyfracgen.types import Bound, Lattice


@jit  # type: ignore[misc]
def _lyapunov_paint(
    string: str,
    xvals: Sequence[float],
    yvals: Sequence[float],
    lattice: Lattice,
    ninit: int,
    niter: int,
) -> None:

    instructions = [0 if s == "A" else 1 for s in string]
    length = len(instructions)
    for iy, yval in enumerate(yvals):
        for ix, xval in enumerate(xvals):
            coord = [xval, yval]
            count = 0
            x = 0.5
            for _ in range(ninit):
                rn = coord[instructions[count % length]]
                x = (rn * x) * (1 - x)
                count += 1
            lamd = 0.0
            for _ in range(niter):
                rn = coord[instructions[count % length]]
                x = (rn * x) * (1 - x)
                lamd += np.log(np.abs(rn - 2 * rn * x))
                count += 1
            lattice[iy, ix] += lamd / niter


class Lyapunov(CanvasBounded):
    def paint(
        self,
        string: str,
        ninit: int,
        niter: int,
    ) -> None:

        _lyapunov_paint(string, self.xvals, self.yvals, self.lattice, ninit, niter)


def lyapunov(
    string: str,
    xbound: Bound,
    ybound: Bound,
    ninit: int = 200,
    niter: int = 800,
    width: int = 5,
    height: int = 4,
    dpi: int = 300,
) -> Result:

    canvas = Lyapunov(width, height, dpi, xbound, ybound)
    canvas.paint(string, ninit, niter)
    return canvas.result
