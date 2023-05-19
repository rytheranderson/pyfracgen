from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import random

from pyfracgen.mandelbrot import _mandelbrot
from pyfracgen.result import Result
from pyfracgen.types import Bound, ResultArray, UpdateFunc


@jit  # type: ignore[misc]
def compute_cvals(
    n_cvals: int,
    xbound: Bound,
    ybound: Bound,
    update_func: UpdateFunc,
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    importance_weight: float = 0.75,
    transpose_energy_grid: bool = False,
) -> npt.NDArray[np.float32]:

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

    # Randomly sampled starting points
    for _ in range(nr):
        c = xmin + (random() * (xmax - xmin)) + 1j * (ymin + (random() * (ymax - ymin)))
        cvals.append(c)

    # Energy grid sampled starting points
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
    xbound: Bound,
    ybound: Bound,
    cvals: npt.NDArray[np.float32],
    update_func: UpdateFunc,
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiter: int = 100,
    horizon: float = 1.0e6,
) -> tuple[ResultArray, int, int, int]:

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
    xbound: Bound,
    ybound: Bound,
    cvals: npt.NDArray[np.float32],
    update_func: UpdateFunc,
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
