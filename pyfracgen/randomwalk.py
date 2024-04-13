import itertools as itt

import numpy as np
from numba import jit
from numpy import array
from numpy.random import randint

from pyfracgen.common import Result, Canvas
from pyfracgen.types import Moves, Lattice


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
    moves = np.array(list(itt.product([1, -1, 0], [1, -1, 0])))

    def paint(self, niter: int) -> None:
        _randomwalk_paint(self.lattice, self.moves, niter)


def randomwalk(
    niter: int,
    width: int = 5,
    height: int = 4,
    dpi: int = 300,
) -> Result:
    canvas = RandomWalk(width, height, dpi)
    canvas.paint(niter)
    return canvas.result
