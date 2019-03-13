import numpy as np

from corrscope.utils.scipy import windows
from corrscope.wave import FLOAT


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


def cosine_flat(n: int, diameter: int, falloff: int) -> np.ndarray:
    cosine = windows.hann(falloff * 2)
    # assert cosine.dtype == FLOAT
    left, right = cosine[:falloff], cosine[falloff:]

    window = np.concatenate([left, np.ones(diameter, dtype=FLOAT), right])

    padded = midpad(window, n)
    # assert padded.dtype == FLOAT
    return padded
