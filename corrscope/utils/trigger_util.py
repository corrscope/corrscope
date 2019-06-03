from typing import Union

import numpy as np

from corrscope.utils.scipy import signal as signal


# get_trigger()
def get_period(data: np.ndarray) -> int:
    """
    Use autocorrelation to estimate the period of a signal.
    Loosely inspired by https://github.com/endolith/waveform_analysis
    """
    corr = signal.correlate(data, data)
    corr = corr[len(corr) // 2 :]

    # Remove the zero-correlation peak
    zero_crossings = np.where(corr < 0)[0]

    if len(zero_crossings) == 0:
        # This can happen given an array of all zeros. Anything else?
        return len(data)

    crossX = zero_crossings[0]
    peakX = crossX + np.argmax(corr[crossX:])
    return int(peakX)


# update_buffer()
MIN_AMPLITUDE = 0.01


def normalize_buffer(data: np.ndarray) -> None:
    """
    Rescales `data` in-place.
    """
    peak = np.amax(abs(data))
    data /= max(peak, MIN_AMPLITUDE)


Arithmetic = Union[np.ndarray, float]


def lerp(x: Arithmetic, y: Arithmetic, a: float) -> Arithmetic:
    return x * (1 - a) + y * a


def abs_max(data, offset=0):
    return np.amax(np.abs(data)) + offset
