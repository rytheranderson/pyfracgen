from cmath import cos, exp, sin

from numba import jit
from numpy import conj


@jit  # type: ignore[misc]
def power(z: complex, c: complex, n: int = 2) -> complex:
    return z**n + c


@jit  # type: ignore[misc]
def conj_power(z: complex, c: complex, n: int = 2) -> complex:
    val: complex = conj(z) ** n + c
    return val


@jit  # type: ignore[misc]
def cosine(z: complex, c: complex) -> complex:
    return c * cos(z)


@jit  # type: ignore[misc]
def sine(z: complex, c: float) -> complex:
    return c * sin(z)


@jit  # type: ignore[misc]
def exponential(z: complex, c: complex) -> complex:
    return c * exp(z)


@jit  # type: ignore[misc]
def magnetic_1(z: complex, c: complex) -> complex:
    t0 = (z * z + c - 1) / (2 * z + c - 2)
    t1 = (z * z + c - 1) / (2 * z + c - 2)
    return t0 * t1


@jit  # type: ignore[misc]
def magnetic_2(z: complex, c: complex) -> complex:
    t0 = (z * z * z * 3 * (c - 1) * z + (c - 1) * (c - 2)) / (
        3 * z * z + 3 * (c - 2) * z + (c - 1) * (c - 2) + 1
    )
    t1 = (z * z * z * 3 * (c - 1) * z + (c - 1) * (c - 2)) / (
        3 * z * z + 3 * (c - 2) * z + (c - 1) * (c - 2) + 1
    )
    return t0 * t1
