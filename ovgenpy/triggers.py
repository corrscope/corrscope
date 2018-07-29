from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type

import numpy as np
from scipy import signal

from ovgenpy.config import register_config, OvgenError
from ovgenpy.util import find
from ovgenpy.wave import FLOAT


if TYPE_CHECKING:
    from ovgenpy.wave import Wave


class ITriggerConfig:
    cls: Type['Trigger']

    def __call__(self, wave: 'Wave', nsamp: int, subsampling: int):
        return self.cls(wave, cfg=self, nsamp=nsamp, subsampling=subsampling)


def register_trigger(config_t: Type[ITriggerConfig]):
    """ @register_trigger(FooTriggerConfig)
    def FooTrigger(): ...
    """
    def inner(trigger_t: Type[Trigger]):
        config_t.cls = trigger_t
        return trigger_t

    return inner


class Trigger(ABC):
    def __init__(self, wave: 'Wave', cfg: ITriggerConfig, nsamp: int, subsampling: int):
        self.cfg = cfg
        self._wave = wave

        self._nsamp = nsamp
        self._subsampling = subsampling

    @abstractmethod
    def get_trigger(self, index: int) -> int:
        """
        :param index: sample index
        :return: new sample index, corresponding to rising edge
        """
        ...


def lerp(x: np.ndarray, y: np.ndarray, a: float):
    return x * (1 - a) + y * a


@register_config
class CorrelationTriggerConfig(ITriggerConfig):
    # get_trigger
    trigger_strength: float
    use_edge_trigger: bool

    # _update_buffer
    responsiveness: float
    falloff_width: float


@register_trigger(CorrelationTriggerConfig)
class CorrelationTrigger(Trigger):
    MIN_AMPLITUDE = 0.01
    ZERO_CROSSING_SCAN = 256
    cfg: CorrelationTriggerConfig

    def __init__(self, *args, **kwargs):
        """
        Correlation-based trigger which looks at a window of `trigger_nsamp` samples.
        it's complicated
        """
        Trigger.__init__(self, *args, **kwargs)
        self._buffer_nsamp = self._nsamp

        # Create correlation buffer (containing a series of old data)
        self._buffer = np.zeros(self._buffer_nsamp, dtype=FLOAT)    # type: np.ndarray[FLOAT]

        # Create zero crossing trigger, for postprocessing results
        self._zero_trigger = ZeroCrossingTrigger(
            self._wave,
            ITriggerConfig(),
            nsamp=self.ZERO_CROSSING_SCAN,
            subsampling=1,
        )

    def get_trigger(self, index: int) -> int:
        """
        :param index: sample index
        :return: new sample index, corresponding to rising edge
        """
        trigger_strength = self.cfg.trigger_strength
        use_edge_trigger = self.cfg.use_edge_trigger

        N = self._buffer_nsamp
        halfN = N // 2

        # data = windowed
        data = self._wave.get_around(index, N, self._subsampling)
        data *= signal.gaussian(N, std = halfN / np.sqrt(self._subsampling))

        # prev_buffer = windowed step function + self._buffer
        step = np.empty(N, dtype=FLOAT)     # type: np.ndarray[FLOAT]
        step[:halfN] = -trigger_strength / 2
        step[halfN:] = trigger_strength / 2

        step *= signal.gaussian(N, std = halfN / 3)

        prev_buffer = self._buffer + step

        # Calculate correlation
        """
        If offset < optimal, we need to `offset += positive`.
        - The peak will appear near the right of `data`.

        Either we must slide prev_buffer to the right:
        - correlate(data, prev_buffer)
        - trigger = offset + peak_offset

        Or we must slide data to the left (by sliding offset to the right):
        - correlate(prev_buffer, data)
        - trigger = offset - peak_offset
        """
        corr = signal.correlate(data, prev_buffer)
        assert len(corr) == 2*N - 1

        # Find optimal offset (within ±N//4)
        mid = N-1
        radius = N//4

        left = mid - radius
        right = mid + radius + 1

        corr = corr[left:right]
        mid = mid - left

        # argmax(corr) == mid + peak_offset == (data >> peak_offset)
        # peak_offset == argmax(corr) - mid
        peak_offset = np.argmax(corr) - mid   # type: int
        trigger = index + (self._subsampling * peak_offset)

        # Update correlation buffer (distinct from visible area)
        aligned = self._wave.get_around(trigger, self._buffer_nsamp, self._subsampling)
        self._update_buffer(aligned)

        if use_edge_trigger:
            return self._zero_trigger.get_trigger(trigger)
        else:
            return trigger

    def _update_buffer(self, data: np.ndarray) -> None:
        """
        Update self._buffer by adding `data` and a step function.
        Data is reshaped to taper away from the center.

        :param data: Wave data. WILL BE MODIFIED.
        """
        falloff_width = self.cfg.falloff_width
        responsiveness = self.cfg.responsiveness

        N = len(data)
        if N != self._buffer_nsamp:
            raise ValueError(f'invalid data length {len(data)} does not match '
                             f'CorrelationTrigger {self._buffer_nsamp}')

        # New waveform
        self._normalize_buffer(data)

        wave_period = get_period(data)
        window = signal.gaussian(N, std = wave_period * falloff_width)
        data *= window

        # Old buffer
        self._normalize_buffer(self._buffer)
        self._buffer = lerp(self._buffer, data, responsiveness)

    # const method
    def _normalize_buffer(self, data: np.ndarray) -> None:
        """
        Rescales `data` in-place.
        """
        peak = np.amax(abs(data))
        data /= max(peak, self.MIN_AMPLITUDE)


def get_period(data: np.ndarray) -> int:
    """
    Use autocorrelation to estimate the period of a signal.
    Loosely inspired by https://github.com/endolith/waveform_analysis
    """
    corr = signal.correlate(data, data, mode='full', method='fft')
    corr = corr[len(corr) // 2:]

    # Remove the zero-correlation peak
    zero_crossings = np.where(corr < 0)[0]

    if len(zero_crossings) == 0:
        # This can happen given an array of all zeros. Anything else?
        return len(data)

    crossX = zero_crossings[0]
    peakX = crossX + np.argmax(corr[crossX:])
    return peakX


class ZeroCrossingTrigger(Trigger):
    # TODO support subsampling
    def get_trigger(self, index: int):
        if self._subsampling != 1:
            raise OvgenError(
                f'ZeroCrossingTrigger with subsampling != 1 is not implemented '
                f'(supplied {self._subsampling})')
        nsamp = self._nsamp

        if not 0 <= index < self._wave.nsamp:
            return index

        if self._wave[index] < 0:
            direction = 1
            test = lambda a: a >= 0

        elif self._wave[index] > 0:
            direction = -1
            test = lambda a: a <= 0

        else:   # self._wave[sample] == 0
            return index + 1

        data = self._wave[index : index + (direction * nsamp) : direction]
        intercepts = find(data, test)
        try:
            (delta,), value = next(intercepts)
            return index + (delta * direction) + int(value <= 0)

        except StopIteration:   # No zero-intercepts
            return index

        # noinspection PyUnreachableCode
        """
        `value <= 0` produces poor results on on sine waves, since it erroneously
        increments the exact idx of the zero-crossing sample.

        `value < 0` produces poor results on impulse24000, since idx = 23999 which
        doesn't match CorrelationTrigger. (scans left looking for a zero-crossing)

        CorrelationTrigger tries to maximize @trigger - @(trigger-1). I think always
        incrementing zeros (impulse24000 = 24000) is acceptable.

        - To be consistent, we should increment zeros whenever we *start* there.
        """
