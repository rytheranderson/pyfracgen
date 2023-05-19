from __future__ import annotations

import math
from typing import Any, Iterator, Sequence

import numpy as np
from numba import jit
from numpy.random import random

from pyfracgen.common import Canvas, Result
from pyfracgen.mandelbrot import _mandelbrot_paint
from pyfracgen.types import Array64, Bound, ComplexArray128, UpdateFunc


@jit  # type: ignore[misc]
def threshold_round_array(arr: Array64, threshold: float = 0.5) -> None:

    w, h = arr.shape
    for i in range(w):
        for j in range(h):
            current = arr[i, j]
            if current - int(current) < threshold:
                rounded = math.floor(current)
            else:
                rounded = math.ceil(current)
            arr[i, j] = rounded


@jit  # type: ignore[misc]
def round_array_preserving_sum(arr: Array64) -> None:

    target = arr.sum()
    best = (math.inf, 0.5)
    # Simply checking 100 values between 0 and 1 works well enough
    for thresh in np.linspace(0, 1, 100):
        check = arr.copy()
        threshold_round_array(check, thresh)
        dist = abs(check.sum() - target)
        if dist < best[0]:
            best = (dist, thresh)
    threshold_round_array(arr, best[-1])


@jit  # type: ignore[misc]
def _compute_cvals(
    ncvals: int,
    update_func: UpdateFunc,
    bounds: tuple[Bound, Bound],
    xvals: Sequence[float],
    yvals: Sequence[float],
    boxes: tuple[Array64, Array64],
    importance_weight: float = 0.75,
) -> ComplexArray128:

    nr = round(ncvals * (1 - importance_weight))
    cvals = []
    (xmin, xmax), (ymin, ymax) = bounds
    # Randomly sampled starting points
    for _ in range(nr):
        c = xmin + (random() * (xmax - xmin)) + 1j * (ymin + (random() * (ymax - ymin)))
        cvals.append(c)

    # Energy grid sampled starting points
    if importance_weight > 0.0:
        ni = round(ncvals * importance_weight)
        energy_grid = np.zeros((len(yvals), len(xvals)))
        _mandelbrot_paint(
            xvals,
            yvals,
            energy_grid,
            update_func,
            maxiter=1000,
            horizon=2.5,
            log_smooth=False,
        )
        energy_grid = (energy_grid / energy_grid.sum()) * ni
        round_array_preserving_sum(energy_grid)
        xboxes, yboxes = boxes

        for iy in range(len(yboxes)):
            for ix in range(len(xboxes)):
                ylo, yhi = yboxes[iy]
                xlo, xhi = xboxes[ix]
                nsamples = int(energy_grid[iy, ix])
                cs = (
                    xlo
                    + (random(nsamples) * (xhi - xlo))
                    + 1j * (ylo + (random(nsamples) * (yhi - ylo)))
                )
                cvals.extend(list(cs))
    print(ni, nr)
    return np.array(cvals)


@jit  # type: ignore[misc]
def _buddhabrot_paint(
    boxes: tuple[Array64, Array64],
    lattice: Array64,
    update_func: UpdateFunc,
    cvals: Array64 | None = None,
    maxiter: int = 100,
    horizon: float = 1.0e6,
) -> None:

    if cvals is None:
        return
    xboxes, yboxes = boxes
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
            for bx in range(len(xboxes)):
                if xboxes[bx][0] < c.real < xboxes[bx][1]:
                    indx += bx
                    break
            for by in range(len(yboxes)):
                if yboxes[by][0] < c.imag < yboxes[by][1]:
                    indy += by
                    break
            if indx != 0 and indy != 0:
                lattice[indy, indx] += 1


class Buddhabrot(Canvas):
    @property
    def boxes(self) -> tuple[Array64, Array64]:

        xboxes = [
            (self.xvals[ix], self.xvals[ix + 1]) for ix in range(len(self.xvals) - 1)
        ]
        yboxes = [
            (self.yvals[ix], self.yvals[ix + 1]) for ix in range(len(self.yvals) - 1)
        ]
        *_, (_, lastx) = xboxes
        *_, (_, lasty) = yboxes
        (_, xmax), (_, ymax) = self.bounds
        xboxes += [(lastx, xmax)]
        yboxes += [(lasty, ymax)]
        return np.array(xboxes), np.array(yboxes)

    def compute_cvals(
        self,
        ncvals: int,
        update_func: UpdateFunc,
        importance_weight: float = 0.75,
    ) -> ComplexArray128:

        cvals: ComplexArray128 = _compute_cvals(
            ncvals,
            update_func,
            self.bounds,
            self.xvals,
            self.yvals,
            self.boxes,
            importance_weight=importance_weight,
        )
        return cvals

    def paint(self, **kwargs: Any) -> None:

        _buddhabrot_paint(
            self.boxes,
            self.lattice,
            **kwargs,
        )


def buddhabrot(
    xbound: Bound,
    ybound: Bound,
    ncvals: int,
    update_func: UpdateFunc,
    width: int = 5,
    height: int = 5,
    dpi: int = 100,
    maxiters: Sequence[int] = [100],
    horizon: float = 1.0e6,
) -> Iterator[Result]:

    canvases = {m: Buddhabrot(xbound, ybound, width, height, dpi) for m in maxiters}
    (_, first), *_ = canvases.items()
    cvals = first.compute_cvals(ncvals, update_func)
    for maxiter, canvas in canvases.items():
        canvas.paint(
            cvals=cvals,
            update_func=update_func,
            maxiter=maxiter,
            horizon=horizon,
        )
        yield canvas.result
