from typing import Union, Optional, TYPE_CHECKING

import numpy as np

import corrscope.utils.scipy.signal as signal
from corrscope.util import iround
from corrscope.wave_common import f32

if TYPE_CHECKING:
    import corrscope.triggers as t

# 0 = no amplification (estimated period too short).
# 1 = full area compensation (estimated period too long).
EDGE_COMPENSATION = 0.9
MAX_AMPLIFICATION = 2

# Return value of get_period.
UNKNOWN_PERIOD = 0


# get_trigger()
def get_period(
    data: np.ndarray,
    subsmp_per_s: float,
    max_freq: float,
    self: "Optional[t.CorrelationTrigger]" = None,
) -> int:
    """
    Use tweaked autocorrelation to estimate the period (AKA pitch) of a signal.
    Loosely inspired by https://github.com/endolith/waveform_analysis

    Design principles:
    - It is better to overestimate the period than underestimate.
        - Underestimation leads to bad triggering.
        - Overestimation only leads to slightly increased x-distance.
    - When the wave is exiting the field of view,
        do NOT estimate an egregiously large period.
    - Do not report a tiny period when faced with YM2612 FM feedback/noise.
        - See get_min_period() docstring.

    Return value:
    - Returns UNKNOWN_PERIOD (0) if period cannot be estimated.
        This is a good placeholder value since it causes buffers/etc. to be basically
        not updated.
    """
    N = len(data)

    # If no input, return period of 0.
    if np.max(np.abs(data)) < MIN_AMPLITUDE:
        return UNKNOWN_PERIOD

    # Begin.
    corr_symmetric = signal.correlate(data, data)
    mid = len(corr_symmetric) // 2
    corr = corr_symmetric[mid:]
    assert len(corr) == len(data)

    def get_min_period() -> int:
        """
        Avoid picking periods shorter than `max_freq`.
        - Yamaha FM feedback produces nearly inaudible high frequencies,
          which tend to produce erroneously short period estimates,
          causing correlation to fail.
        - Most music does not go this high.
        - Overestimating period of high notes is mostly harmless.
        """
        max_cyc_per_s = max_freq
        min_s_per_cyc = 1 / max_cyc_per_s
        min_subsmp_per_cyc = subsmp_per_s * min_s_per_cyc
        return iround(min_subsmp_per_cyc)

    def get_zero_crossing() -> int:
        """Remove the central peak."""
        zero_crossings = np.where(corr < 0)[0]
        if len(zero_crossings) == 0:
            # This can happen given an array of all zeros. Anything else?
            return UNKNOWN_PERIOD
        return zero_crossings[0]

    min_period = get_min_period()
    zero_crossing = get_zero_crossing()
    if zero_crossing == UNKNOWN_PERIOD:
        return UNKNOWN_PERIOD

    # [minX..) = [min_period..) & [zero_crossing..)
    minX = max(min_period, zero_crossing)

    # Remove the zero-correlation peak.
    def calc_peak():
        return minX + np.argmax(corr[minX:])

    temp_peakX = calc_peak()
    # In the case of uncorrelated noise,
    # corr[temp_peakX] can be tiny (smaller than N * MIN_AMPLITUDE^2).
    # But don't return 0 since it's not silence.

    is_long_period = temp_peakX > 0.1 * N
    if is_long_period:
        # If a long-period wave has strong harmonics,
        # the true peak will be attenuated below the harmonic peaks.
        # Compensate for that.
        divisor = np.linspace(1, 1 - EDGE_COMPENSATION, N, endpoint=False, dtype=f32)
        divisor = np.maximum(divisor, 1 / MAX_AMPLIFICATION)
        corr /= divisor
        peakX = calc_peak()

    else:
        peakX = temp_peakX

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
