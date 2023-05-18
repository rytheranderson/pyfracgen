from __future__ import annotations

from typing import Callable

import numpy as np
from numba import jit
from numpy import log

from pyfracgen.result import Result


@jit  # type: ignore[misc]
def _julia(
    c: float,
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    update_func: Callable[[complex, complex], complex],
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 2.0**40,
    log_smooth: bool = True,
) -> tuple[np.ndarray, int, int, int]:

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
    lattice = np.zeros((int(nx), int(ny)), dtype=np.float64)
    log_horizon = log(log(horizon)) / log(2)

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            z = xvals[i] + 1j * yvals[j]
            for iteration in range(maxiter):
                az = abs(z)
                if az > horizon:
                    if log_smooth:
                        lattice[i, j] = iteration - log(log(az)) / log(2) + log_horizon
                    else:
                        lattice[i, j] = iteration
                    break
                z = update_func(z, c)

    return (lattice.T, width, height, dpi)


def julia_series(
    c_vals: np.ndarray,
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    update_func: Callable[[complex, complex], complex],
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 2.0**40,
    log_smooth: bool = True,
) -> list[Result]:

    series = []
    for c in c_vals:
        res = _julia(
            c,
            xbound,
            ybound,
            update_func,
            width=width,
            height=height,
            dpi=dpi,
            maxiter=maxiter,
            horizon=horizon,
            log_smooth=log_smooth,
        )
        res = Result(*res)
        series.append(res)

    return series


def julia(
    c: float,
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    update_func: Callable[[complex, complex], complex],
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 2.0**40,
    log_smooth: bool = True,
) -> Result:

    res = _julia(
        c,
        xbound,
        ybound,
        update_func,
        width=width,
        height=height,
        dpi=dpi,
        maxiter=maxiter,
        horizon=horizon,
        log_smooth=log_smooth,
    )
    return Result(*res)
