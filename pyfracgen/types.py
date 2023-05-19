from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

Array64 = npt.NDArray[np.float64]
ComplexArray128 = npt.NDArray[np.complex128]
Bound = Tuple[float, float]
UpdateFunc = Callable[[complex, complex], complex]
