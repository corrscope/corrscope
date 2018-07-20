from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from dataclasses import dataclass
from scipy import signal

from ovgenpy.util import find
from ovgenpy.wave import FLOAT


if TYPE_CHECKING:
    from ovgenpy.wave import Wave


class Trigger(ABC):
    def __init__(self, wave: 'Wave', scan_nsamp: int):
        self._wave = wave
        self._scan_nsamp = scan_nsamp

    @abstractmethod
    def get_trigger(self, index: int) -> int:
        """
        :param index: sample index
        :return: new sample index, corresponding to rising edge
        """
        ...


class TriggerConfig:
    # NamedTuple inheritance does not work. Mark children @dataclass instead.
    # https://github.com/python/typing/issues/427
    def __call__(self, wave: 'Wave', scan_nsamp: int):
        # idea: __call__ return self.cls(wave, scan_nsamp, cfg=self)
        # problem: cannot reference XTrigger from within XTrigger
        raise NotImplementedError


def lerp(x: np.ndarray, y: np.ndarray, a: float):
    return x * (1 - a) + y * a


class CorrelationTrigger(Trigger):
    MIN_AMPLITUDE = 0.01

    @dataclass
    class Config(TriggerConfig):
        # get_trigger
        trigger_strength: float
        use_edge_trigger: bool

        # _update_buffer
        responsiveness: float
        falloff_width: float

        def __call__(self, wave: 'Wave', scan_nsamp: int):
            return CorrelationTrigger(wave, scan_nsamp, cfg=self)

    # get_trigger postprocessing: self._zero_trigger
    ZERO_CROSSING_SCAN = 256

    def __init__(self, wave: 'Wave', scan_nsamp: int, cfg: Config):
        """
        Correlation-based trigger which looks at a window of `scan_nsamp` samples.

        it's complicated

        :param wave: Wave file
        :param scan_nsamp: Number of samples used to align adjacent frames
        :param cfg: Correlation config
        """
        Trigger.__init__(self, wave, scan_nsamp)
        self._buffer_nsamp = self._scan_nsamp

        # Correlation config
        self.cfg = cfg

        # Create correlation buffer (containing a series of old data)
        self._buffer = np.zeros(scan_nsamp, dtype=FLOAT)    # type: np.ndarray[FLOAT]

        # Create zero crossing trigger, for postprocessing results
        self._zero_trigger = ZeroCrossingTrigger(wave, self.ZERO_CROSSING_SCAN)

    def get_trigger(self, index: int) -> int:
        """
        :param index: sample index
        :return: new sample index, corresponding to rising edge
        """
        trigger_strength = self.cfg.trigger_strength
        use_edge_trigger = self.cfg.use_edge_trigger

        N = self._buffer_nsamp
        data = self._wave.get_around(index, N)

        # prev_buffer = windowed step function + self._buffer
        halfN = N // 2
        step = np.empty(N, dtype=FLOAT)     # type: np.ndarray[FLOAT]
        step[:halfN] = -trigger_strength / 2
        step[halfN:] = trigger_strength / 2

        window = signal.gaussian(N, std = halfN // 3)
        step *= window

        prev_buffer = self._buffer + step

        # Find optimal offset (within ±N//4)
        mid = N-1
        radius = N//4

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

        left = mid - radius
        right = mid + radius + 1

        corr = corr[left:right]
        mid = mid - left

        # argmax(corr) == mid + peak_offset == (data >> peak_offset)
        # peak_offset == argmax(corr) - mid
        peak_offset = np.argmax(corr) - mid   # type: int
        trigger = index + peak_offset

        # Update correlation buffer (distinct from visible area)
        aligned = self._wave.get_around(trigger, self._buffer_nsamp)
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
    def __init__(self, wave: 'Wave', scan_nsamp: int):
        super().__init__(wave, scan_nsamp)

    def get_trigger(self, index: int):
        scan_nsamp = self._scan_nsamp

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

        data = self._wave[index : index + (direction * scan_nsamp) : direction]
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
