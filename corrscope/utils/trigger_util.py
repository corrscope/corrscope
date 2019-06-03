from typing import Union

import numpy as np

import corrscope.triggers as t
import corrscope.utils.scipy.signal as signal
from corrscope.wave import FLOAT

# 0 = no amplification (estimated period too short).
# 1 = full area compensation (estimated period too long).
EDGE_COMPENSATION = 0.9

MAX_AMPLIFICATION = 2


# get_trigger()
def get_period(data: np.ndarray, self: "t.CorrelationTrigger" = None) -> int:
    """
    Use autocorrelation to estimate the period (AKA pitch) of a signal.
    Loosely inspired by https://github.com/endolith/waveform_analysis

    Design principles:
    - It is better to overestimate the period than underestimate.
        - Underestimation leads to bad triggering.
        - Overestimation only leads to slightly increased x-distance.
    - When the wave is exiting the field of view,
        do NOT estimate an egregiously large period.

    Return value:
    - Returns 0 if period cannot be estimated.
        This is a good placeholder value
        since it causes buffers/etc. to be basically not updated.
    """
    UNKNOWN_PERIOD = 0

    N = len(data)

    # If no input, return period of 1.
    if np.add.reduce(np.abs(data)) < MIN_AMPLITUDE * N:
        return UNKNOWN_PERIOD

    # Begin.
    corr_symmetric = signal.correlate(data, data)
    mid = len(corr_symmetric) // 2
    corr = corr_symmetric[mid:]
    assert len(corr) == len(data)

    # Remove the central peak.
    zero_crossings = np.where(corr < 0)[0]
    if len(zero_crossings) == 0:
        # This can happen given an array of all zeros. Anything else?
        return UNKNOWN_PERIOD
    crossX = zero_crossings[0]

    # Remove the zero-correlation peak.
    def calc_peak():
        return crossX + np.argmax(corr[crossX:])

    temp_peakX = calc_peak()
    # In the case of uncorrelated noise,
    # corr[temp_peakX] can be tiny (smaller than N * MIN_AMPLITUDE^2).
    # But don't return 0 since it's not silence.

    is_long_period = temp_peakX > 0.1 * N
    self.custom_line("owo", np.full(500, is_long_period), False, False)
    if is_long_period:
        # If a long-period wave has strong harmonics,
        # the true peak will be attenuated below the harmonic peaks.
        # Compensate for that.
        divisor = np.linspace(1, 1 - EDGE_COMPENSATION, N, endpoint=False, dtype=FLOAT)
        divisor = np.maximum(divisor, 1 / MAX_AMPLIFICATION)
        corr /= divisor
        peakX = calc_peak()

    else:
        peakX = temp_peakX

    self.custom_vline("peakPeriod", peakX - N // 2, False)
    self.custom_line("autocorrelation", corr, False, False)

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
