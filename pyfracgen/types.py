
from __future__ import annotations

import numpy.typing as npt
import numpy as np
from typing import Tuple, Callable


ResultArray = npt.NDArray[np.float32 | np.float64]
Bound = Tuple[float, float]
UpdateFunc = Callable[[complex, complex], complex]