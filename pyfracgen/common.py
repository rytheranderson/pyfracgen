"""Utility objects and methods used across the library."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyfracgen.types import Array64, Bound


@dataclass
class Result:

    image_array: Array64
    width_inches: int
    height_inches: int
    dpi: int

    @classmethod
    def load(cls, file: Path) -> Result:
        with open(file, "rb") as f:
            res = pickle.load(f)
        return cls(*res)

    def save(self, name: Path = Path("save")) -> None:
        res = [self.image_array, self.width_inches, self.height_inches, self.dpi]
        with open(name, "wb") as f:
            pickle.dump(res, f)


class Canvas:
    def __init__(
        self, xbound: Bound, ybound: Bound, width: int, height: int, dpi: int
    ) -> None:

        xmin, xmax = xbound
        ymin, ymax = ybound
        nx, ny = width * dpi, height * dpi

        self.bounds = (xbound, ybound)
        self.width = width
        self.height = height
        self.dpi = dpi
        self.xvals = np.array(
            [xmin + i * (xmax - xmin) / nx for i in range(nx)], dtype=np.float64
        )
        self.yvals = np.array(
            [ymin + i * (ymax - ymin) / ny for i in range(ny)], dtype=np.float64
        )
        self.lattice: Array64 = np.zeros((ny, nx), dtype=np.float64)

    @property
    def result(self) -> Result:
        return Result(self.lattice, self.width, self.height, self.dpi)

    def paint(self, **kwargs: Any) -> None:
        raise NotImplementedError
