from __future__ import annotations

from typing import Callable

import numpy as np
from numba import jit
from numpy.random import random

from pyfracgen.mandelbrot import _mandelbrot
from pyfracgen.result import Result


@jit  # type: ignore[misc]
def compute_cvals(
    n_cvals: int,
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    update_func: Callable[[complex, complex], complex],
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    importance_weight: float = 0.75,
    transpose_energy_grid: bool = False,
) -> np.ndarray:

    xmin, xmax = [float(xbound[0]), float(xbound[1])]
    ymin, ymax = [float(ybound[0]), float(ybound[1])]
    nx = width * dpi
    ny = height * dpi
    xvals = np.array(
        [xmin + i * (xmax - xmin) / nx for i in range(nx)], dtype=np.float32
    )
    yvals = np.array(
        [ymin + i * (ymax - ymin) / ny for i in range(ny)], dtype=np.float32
    )
    xboxs = [(xvals[i], xvals[i + 1]) for i in range(len(xvals) - 1)]
    yboxs = [(yvals[i], yvals[i + 1]) for i in range(len(yvals) - 1)]
    xboxs = xboxs + [(xboxs[-1][1], xmax)]
    yboxs = yboxs + [(yboxs[-1][1], ymax)]
    nr = int(round(n_cvals * (1 - importance_weight)))
    cvals = []

    # randomly sampled starting points
    for k in range(nr):
        c = xmin + (random() * (xmax - xmin)) + 1j * (ymin + (random() * (ymax - ymin)))
        cvals.append(c)

    # energy grid sampled starting points
    if importance_weight > 0.0:
        ni = int(round(n_cvals * importance_weight))
        energy_grid = _mandelbrot(
            xbound,
            ybound,
            update_func,
            width=width,
            height=height,
            dpi=dpi,
            maxiter=1000,
            horizon=2.5,
            log_smooth=False,
        )[0].T
        energy_grid = (energy_grid / energy_grid.sum()) * ni
        energy_grid = energy_grid.T if transpose_energy_grid else energy_grid
        for i in range(nx):
            for j in range(ny):
                num = int(round(energy_grid[i, j]))
                xlo, xhi = xboxs[i]
                ylo, yhi = yboxs[j]
                cs = (
                    xlo
                    + (random(num) * (xhi - xlo))
                    + 1j * (ylo + (random(num) * (yhi - ylo)))
                )
                cvals.extend(list(cs))

    return np.array(cvals)


@jit  # type: ignore[misc]
def _buddhabrot(
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    cvals: np.ndarray,
    update_func: Callable[[complex, complex], complex],
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 1.0e6,
) -> tuple[np.ndarray, int, int, int]:

    xmin, xmax = [float(xbound[0]), float(xbound[1])]
    ymin, ymax = [float(ybound[0]), float(ybound[1])]
    nx = width * dpi
    ny = height * dpi
    xvals = np.array(
        [xmin + i * (xmax - xmin) / nx for i in range(nx)], dtype=np.float32
    )
    yvals = np.array(
        [ymin + i * (ymax - ymin) / ny for i in range(ny)], dtype=np.float32
    )
    xboxs = [(xvals[i], xvals[i + 1]) for i in range(len(xvals) - 1)]
    yboxs = [(yvals[i], yvals[i + 1]) for i in range(len(yvals) - 1)]
    xboxs = xboxs + [(xboxs[-1][1], xmax)]
    yboxs = yboxs + [(yboxs[-1][1], ymax)]
    lattice = np.zeros((int(width * dpi), int(height * dpi)), dtype=np.float32)

    for c in cvals:
        z = c
        trial_sequence = []
        sequence = []
        for _ in range(maxiter):
            az = np.abs(z)
            trial_sequence.append(z)
            if az > horizon:
                sequence.extend(trial_sequence)
                break
            z = update_func(z, c)
        for c in sequence:
            indx = 0
            indy = 0
            for bx in range(nx):
                if xboxs[bx][0] < c.real < xboxs[bx][1]:
                    indx += bx
                    break
            for by in range(ny):
                if yboxs[by][0] < c.imag < yboxs[by][1]:
                    indy += by
                    break
            if indx != 0 and indy != 0:
                lattice[indx, indy] += 1

    return (lattice.T, width, height, dpi)


def buddhabrot(
    xbound: tuple[float, float],
    ybound: tuple[float, float],
    cvals: np.ndarray,
    update_func: Callable[[complex, complex], complex],
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 1.0e6,
) -> Result:

    res = _buddhabrot(
        xbound,
        ybound,
        cvals,
        update_func,
        width=width,
        height=height,
        dpi=dpi,
        maxiter=maxiter,
        horizon=horizon,
    )
    return Result(*res)
