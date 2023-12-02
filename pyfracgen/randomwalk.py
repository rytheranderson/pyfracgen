import itertools as itt

import numpy as np
from numba import jit
from numpy import array
from numpy.random import randint

from pyfracgen.common import Canvas3D, Result
from pyfracgen.types import Lattice3D, Moves3D


def construct_moves(basis: Moves3D) -> Moves3D:

    basis = np.r_[basis, -1 * basis, [array([0, 0, 0])]]
    nonnull = list(
        itt.filterfalse(
            lambda x: not np.any(x), (b0 + b1 for b0, b1 in itt.combinations(basis, 2))
        )
    )
    moves: Moves3D = np.unique(nonnull, axis=0)
    return moves


@jit  # type: ignore[misc]
def _randomwalk_paint(
    lattice: Lattice3D,
    moves: Moves3D,
    niter: int,
) -> None:

    nmoves = len(moves)
    h, w, d = lattice.shape
    indices = array([h, w, d]) / 2.0
    for iteration in range(niter):
        move = moves[randint(0, nmoves)]
        indices += move
        iy, ix, iz = int(indices[0] % h), int(indices[1] % w), int(indices[2] % d)
        lattice[iy, ix, iz] = iteration

    lattice /= np.amax(lattice)


class RandomWalk(Canvas3D):
    def paint(self, moves: Moves3D, niter: int) -> None:
        _randomwalk_paint(self.lattice, moves, niter)


def randomwalk(
    moves: Moves3D,
    niter: int,
    width: int = 5,
    height: int = 4,
    depth: int = 1,
    dpi: int = 300,
) -> Result:

    canvas = RandomWalk(width, height, depth, dpi)
    canvas.paint(moves, niter)
    return canvas.result
