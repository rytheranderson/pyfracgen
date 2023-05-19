from __future__ import annotations

import itertools as itt

import numpy as np
from numba import jit
from numpy import array
from numpy.random import randint

from pyfracgen.common import Result
from pyfracgen.types import Array64


def construct_moves(basis: Array64) -> Array64:

    basis = np.r_[basis, -1 * basis, [array([0, 0, 0])]]
    nonnull = list(
        itt.filterfalse(
            lambda x: not np.any(x), (b0 + b1 for b0, b1 in itt.combinations(basis, 2))
        )
    )
    moves: Array64 = np.unique(nonnull, axis=0)
    return moves


@jit  # type: ignore[misc]
def _randomwalk(
    moves: Array64,
    niter: int,
    width: int = 5,
    height: int = 5,
    depth: int = 1,
    dpi: int = 100,
    tracking: str = "visitation",
) -> tuple[Array64, int, int, int]:

    lattice = np.zeros(
        (int(height * dpi), int(width * dpi), int(depth)), dtype=np.float64
    )
    shape = array([height * dpi, width * dpi, depth])
    nmoves = len(moves)
    l0, l1, l2 = shape
    indices = array([height * dpi, width * dpi, depth]) / 2.0

    for iteration in range(niter):
        move = moves[randint(0, nmoves)]
        indices += move
        i, j, k = int(indices[0] % l0), int(indices[1] % l1), int(indices[2] % l2)
        if tracking == "visitation":
            lattice[i, j, k] += 1.0
        elif tracking == "temporal":
            lattice[i, j, k] = iteration

    lattice /= np.amax(lattice)
    return (lattice, width, height, dpi)


def randomwalk(
    moves: Array64,
    niter: int,
    width: int = 5,
    height: int = 5,
    depth: int = 1,
    dpi: int = 100,
    tracking: str = "visitation",
) -> Result:

    res = _randomwalk(
        moves,
        niter,
        width=width,
        height=height,
        depth=depth,
        dpi=dpi,
        tracking=tracking,
    )
    return Result(*res)
