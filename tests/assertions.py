from hashlib import md5
from pathlib import Path


def assert_pngs_equal(result: Path, answer: Path) -> None:
    """Assert two PNGs, likely a test result and an answer, are equal.

    Compare md5 hashes to keep error prints small.

    Args:
        result: The left PNG in the comparison
        answer: The right PNG in the comparison.
    """
    with open(result, "rb") as res_file:
        res_bytes = res_file.read()
    with open(answer, "rb") as ans_file:
        ans_bytes = ans_file.read()
    assert md5(res_bytes).digest() == md5(ans_bytes).digest()
