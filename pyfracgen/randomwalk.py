from __future__ import annotations

from itertools import combinations

import numpy as np
from numba import jit
from numpy import array
from numpy.random import randint

from pyfracgen.result import Result


def construct_moves(basis: np.ndarray) -> np.ndarray:

    basis = np.r_[basis, -1 * basis, [array([0, 0, 0])]]
    moves = np.unique(array([b0 + b1 for b0, b1 in combinations(basis, 2)]), axis=0)
    moves = array([m for m in moves if np.any(m)])
    return moves


@jit  # type: ignore[misc]
def _randomwalk(
    moves: np.ndarray,
    niter: int,
    width: int = 5,
    height: int = 5,
    depth: int = 1,
    dpi: int = 100,
    tracking: str = "visitation",
) -> tuple[np.ndarray, int, int, int]:

    lattice = np.zeros(
        (int(height * dpi), int(width * dpi), int(depth)), dtype=np.float32
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
    moves: np.ndarray,
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
