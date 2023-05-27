"""Utility objects and methods used across the library."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyfracgen.types import Bound, Lattice, Lattice3D

RESULT_DEFAULT_SAVE = Path("save.pickle")


@dataclass(frozen=True)
class Result:

    image_array: Lattice
    width_inches: int
    height_inches: int
    dpi: int

    @classmethod
    def load(cls, file: Path) -> Result:
        with open(file, "rb") as f:
            res = pickle.load(f)
        return cls(*res)

    def save(self, name: Path = RESULT_DEFAULT_SAVE) -> None:
        res = [self.image_array, self.width_inches, self.height_inches, self.dpi]
        with open(name, "wb") as f:
            pickle.dump(res, f)


class Canvas:
    def __init__(self, width: int, height: int, dpi: int):
        self.lattice: Lattice = np.zeros((height * dpi, width * dpi), dtype=np.float64)
        self.width = width
        self.height = height
        self.dpi = dpi

    @property
    def result(self) -> Result:
        return Result(self.lattice, self.width, self.height, self.dpi)

    def paint(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


class Canvas3D(Canvas):
    def __init__(self, width: int, height: int, depth: int, dpi: int) -> None:
        super().__init__(width, height, dpi)
        self.lattice: Lattice3D = np.dstack(
            [np.zeros(self.lattice.shape) for _ in range(depth)]
        )


class CanvasBounded(Canvas):
    def __init__(
        self, width: int, height: int, dpi: int, xbound: Bound, ybound: Bound
    ) -> None:
        super().__init__(width, height, dpi)
        ny, nx = self.lattice.shape
        self.xbound = xbound
        self.ybound = ybound
        (xmin, xmax), (ymin, ymax) = xbound, ybound
        self.xvals = np.array(
            [xmin + i * (xmax - xmin) / nx for i in range(nx)], dtype=np.float64
        )
        self.yvals = np.array(
            [ymin + i * (ymax - ymin) / ny for i in range(ny)], dtype=np.float64
        )

    @property
    def bounds(self) -> tuple[Bound, Bound]:
        return (self.xbound, self.ybound)
