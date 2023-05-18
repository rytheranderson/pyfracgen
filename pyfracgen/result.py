from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Result:

    image_array: np.ndarray
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
