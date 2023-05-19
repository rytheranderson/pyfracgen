from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

ResultArray = npt.NDArray[np.float32 | np.float64]
Bound = Tuple[float, float]
UpdateFunc = Callable[[complex, complex], complex]
