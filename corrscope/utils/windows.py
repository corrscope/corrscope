import numpy as np

from corrscope.utils.scipy import windows
from corrscope.wave_common import f32


def leftpad(data: np.ndarray, n: int) -> np.ndarray:
    if not n > 0:
        raise ValueError(f"leftpad(n={n}) must be > 0")

    data = data[-n:]
    data = np.pad(data, (n - len(data), 0), "constant")

    return data


def midpad(data: np.ndarray, n: int) -> np.ndarray:
    if not n > 0:
        raise ValueError(f"midpad(n={n}) must be > 0")

    shrink = len(data) - n
    if shrink > 0:
        half = shrink // 2
        data = data[half : -(shrink - half)]
        return data

    expand = n - len(data)
    if expand > 0:
        half = expand // 2
        data = np.pad(data, (half, expand - half), "constant")
        return data

    return data


def rightpad(data: np.ndarray, n: int, constant_values=1) -> np.ndarray:
    if not n > 0:
        raise ValueError(f"rightpad(n={n}) must be > 0")

    data = data[:n]

    # _validate_lengths() raises error on negative values.
    data = np.pad(data, (0, n - len(data)), "constant", constant_values=constant_values)

    return data


def gaussian_or_zero(M: int, std: float, sym: bool = True) -> np.ndarray:
    """
    Sometimes `std` is computed based on period.

    If period is zero (cannot be estimated), return all zeros.
    """
    if std == 0:
        return np.zeros(M, dtype=f32)
    else:
        return windows.gaussian(M, std, sym)
