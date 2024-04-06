import itertools as itt

import numpy as np
from numba import jit
from numpy import array
from numpy.random import randint

from pyfracgen.common import Result, Canvas
from pyfracgen.types import Moves, Lattice


def construct_moves(*vectors: tuple[int, int]) -> Moves:
    basis = []
    for vec in vectors:
        arr = np.array(vec)
        basis.extend([arr, -1 * arr])
    nonnull = list(
        filter(lambda x: np.any(x), (b0 + b1 for b0, b1 in itt.combinations(basis, 2)))
    )
    moves: Moves = np.unique(nonnull, axis=0)
    return moves


@jit(nopython=True)  # type: ignore[misc]
def _randomwalk_paint(
    lattice: Lattice,
    moves: Moves,
    niter: int,
) -> None:
    nmoves = len(moves)
    h, w = lattice.shape
    indices = array([h, w]) / 2.0
    for iteration in range(niter):
        move = moves[randint(0, nmoves)]
        indices += move
        iy, ix = int(indices[0] % h), int(indices[1] % w)
        lattice[iy, ix] = iteration


class RandomWalk(Canvas):
    def paint(self, moves: Canvas, niter: int) -> None:
        _randomwalk_paint(self.lattice, moves, niter)


def randomwalk(
    moves: Moves,
    niter: int,
    width: int = 5,
    height: int = 4,
    dpi: int = 300,
) -> Result:
    canvas = RandomWalk(width, height, dpi)
    canvas.paint(moves, niter)
    return canvas.result
