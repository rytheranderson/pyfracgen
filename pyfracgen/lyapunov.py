from __future__ import annotations

import numpy as np
from numba import jit

from pyfracgen.result import Result
from pyfracgen.types import Bound, ResultArray


@jit  # type: ignore[misc]
def _lyapunov(
    string: str,
    xbound: Bound,
    ybound: Bound,
    n_init: int = 200,
    n_iter: int = 800,
    width: int = 3,
    height: int = 3,
    dpi: int = 100,
) -> tuple[ResultArray, int, int, int]:

    instructions = [0 if s == "A" else 1 for s in string]
    xmin, xmax = [float(xbound[0]), float(xbound[1])]
    ymin, ymax = [float(ybound[0]), float(ybound[1])]
    nx = width * dpi
    ny = height * dpi
    xvals = np.array(
        [xmin + i * (xmax - xmin) / nx for i in range(nx)], dtype=np.float64
    )
    yvals = np.array(
        [ymin + i * (ymax - ymin) / ny for i in range(ny)], dtype=np.float64
    )
    length = len(instructions)
    lattice = np.zeros((int(nx), int(ny)), dtype=np.float64)

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            coord = [xvals[i], yvals[j]]
            n = 0
            x = 0.5
            for _ in range(n_init):
                rn = coord[instructions[n % length]]
                x = (rn * x) * (1 - x)
                n += 1
            lamd = 0.0
            for _ in range(n_iter):
                rn = coord[instructions[n % length]]
                x = (rn * x) * (1 - x)
                lamd += np.log(np.abs(rn - 2 * rn * x))
                n += 1
            lattice[i, j] += lamd / n_iter

    return (lattice.T, width, height, dpi)


def lyapunov(
    string: str,
    xbound: Bound,
    ybound: Bound,
    n_init: int = 200,
    n_iter: int = 800,
    width: int = 3,
    height: int = 3,
    dpi: int = 100,
) -> Result:

    res = _lyapunov(
        string,
        xbound,
        ybound,
        n_init=n_init,
        n_iter=n_iter,
        width=width,
        height=height,
        dpi=dpi,
    )

    return Result(*res)
