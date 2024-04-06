from typing import Callable, NewType, Tuple

from nptyping import Complex128, Float64, Int64, NDArray, Shape

Width = NewType("Width", int)
Height = NewType("Height", int)
Depth = NewType("Depth", int)

Lattice = NDArray[Shape["Height, Width"], Float64]
Moves = NDArray[Shape["*, 2"], Int64]
Boxes = NDArray[Shape["*, 2"], Float64]
ComplexSequence = NDArray[Shape["*"], Complex128]

Bound = Tuple[float, float]
IterFunc = Callable[[complex, complex], complex]
