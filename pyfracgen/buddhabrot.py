import math
from typing import Iterable, Iterator

import numpy as np
from numba import jit
from numpy.random import random

from pyfracgen.common import CanvasBounded, Result
from pyfracgen.iterfuncs.funcs import power
from pyfracgen.mandelbrot import mandelbrot
from pyfracgen.types import Bound, Boxes, ComplexSequence, IterFunc, Lattice


@jit  # type: ignore[misc]
def threshold_round_array(arr: Lattice, threshold: float = 0.5) -> None:

    w, h = arr.shape
    for iy in range(w):
        for ix in range(h):
            current = arr[iy, ix]
            if current - int(current) < threshold:
                rounded = math.floor(current)
            else:
                rounded = math.ceil(current)
            arr[iy, ix] = rounded


@jit  # type: ignore[misc]
def round_array_preserving_sum(arr: Lattice) -> None:

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
    bounds: tuple[Bound, Bound],
    boxes: tuple[Boxes, Boxes],
    energy_grid: Lattice,
    random_fraction: float,
) -> ComplexSequence:

    nr = round(ncvals * random_fraction)
    cvals = []
    (xmin, xmax), (ymin, ymax) = bounds
    # Randomly sampled starting points
    for _ in range(nr):
        c = xmin + (random() * (xmax - xmin)) + 1j * (ymin + (random() * (ymax - ymin)))
        cvals.append(c)

    # Energy grid sampled starting points
    if random_fraction < 1.0:
        ni = round(ncvals * (1 - random_fraction))
        energy_grid = (energy_grid / energy_grid.sum()) * ni
        round_array_preserving_sum(energy_grid)

        xboxes, yboxes = boxes
        for iy, (ylo, yhi) in enumerate(yboxes):
            for ix, (xlo, xhi) in enumerate(xboxes):
                nsamples = int(energy_grid[iy, ix])
                xadd = xlo + (random(nsamples) * (xhi - xlo))
                yadd = ylo + (random(nsamples) * (yhi - ylo))
                cvals.extend(xadd + 1j * yadd)

    return np.array(cvals)


@jit  # type: ignore[misc]
def _buddhabrot_paint(
    bounds: tuple[Bound, Bound],
    lattice: Lattice,
    cvals: ComplexSequence,
    update_func: IterFunc,
    maxiter: int,
    horizon: float,
) -> None:

    (xmin, xmax), (ymin, ymax) = bounds
    height, width = lattice.shape
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
            indx = int((c.real - xmin) / (xmax - xmin) * width)
            indy = int((c.imag - ymin) / (ymax - ymin) * height)
            if (indx < 0 or indx >= width) or (indy < 0 or indy >= height):
                continue
            lattice[indy, indx] += 1


class Buddhabrot(CanvasBounded):
    @property
    def boxes(self) -> tuple[Boxes, Boxes]:

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
        energy_grid: Lattice,
        random_fraction: float = 0.25,
    ) -> ComplexSequence:
        cvals: ComplexSequence = _compute_cvals(
            ncvals,
            self.bounds,
            self.boxes,
            energy_grid,
            random_fraction,
        )
        return cvals

    def paint(
        self,
        cvals: ComplexSequence,
        update_func: IterFunc,
        maxiter: int,
        horizon: float,
    ) -> None:

        _buddhabrot_paint(
            self.bounds,
            self.lattice,
            cvals,
            update_func,
            maxiter,
            horizon,
        )


def buddhabrot(
    xbound: Bound,
    ybound: Bound,
    ncvals: int,
    update_func: IterFunc = power,
    width: int = 5,
    height: int = 4,
    dpi: int = 300,
    maxiters: Iterable[int] = (100,),
    horizon: float = 1.0e6,
    random_fraction: float = 0.25,
) -> Iterator[Result]:

    mdbres = mandelbrot(
        xbound=xbound,
        ybound=ybound,
        update_func=update_func,
        width=width,
        height=height,
        dpi=dpi,
        maxiter=500,
        horizon=2.5,
        log_smooth=False,
    )
    canvi = [(m, Buddhabrot(width, height, dpi, xbound, ybound)) for m in maxiters]
    (_, first), *_ = canvi
    cvals = first.compute_cvals(
        ncvals, mdbres.image_array, random_fraction=random_fraction
    )
    for maxiter, canvas in canvi:
        canvas.paint(cvals, update_func, maxiter, horizon)
        yield canvas.result
