from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, Tuple

import numpy as np
from scipy import signal
from scipy.signal import windows

from ovgenpy.config import register_config, OvgenError
from ovgenpy.util import find
from ovgenpy.utils.windows import midpad, leftpad
from ovgenpy.wave import FLOAT


if TYPE_CHECKING:
    from ovgenpy.wave import Wave

# Abstract classes

# TODO rename nsamp to trigger_nsamp or tsamp or wsamp
class ITriggerConfig:
    cls: Type['Trigger']

    def __call__(self, wave: 'Wave', nsamp: int, subsampling: int, nsamp_frame: int):
        return self.cls(wave, cfg=self, nsamp=nsamp, subsampling=subsampling,
                        nsamp_frame=nsamp_frame)


def register_trigger(config_t: Type[ITriggerConfig]):
    """ @register_trigger(FooTriggerConfig)
    def FooTrigger(): ...
    """
    def inner(trigger_t: Type[Trigger]):
        config_t.cls = trigger_t
        return trigger_t

    return inner


class Trigger(ABC):
    def __init__(self, wave: 'Wave', cfg: ITriggerConfig, nsamp: int, subsampling: int,
                 nsamp_frame: int):
        self.cfg = cfg
        self._wave = wave

        self._nsamp = nsamp
        self._subsampling = subsampling
        self._nsamp_frame = nsamp_frame

    @abstractmethod
    def get_trigger(self, index: int) -> int:
        """
        :param index: sample index
        :return: new sample index, corresponding to rising edge
        """
        ...


# CorrelationTrigger

@register_config(always_dump='*')
class CorrelationTriggerConfig(ITriggerConfig):
    # get_trigger
    edge_strength: float = 10
    trigger_diameter: float = 0.5
    trigger_falloff: Tuple[float, float] = (4, 1)  # FIXME add default, dump/load as list
    use_edge_trigger: bool = True

    # _update_buffer
    responsiveness: float = 0.1
    buffer_falloff: float = 0.5


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
            nsamp_frame=self._nsamp_frame
        )

        # Precompute tables
        self._windowed_step = self._calc_step()

        # Input data window (narrower for long subsampled data)
        self._data_taper = self._calc_data_taper()  # TODO add a right cosine taper or not?

    def _calc_step(self):
        """ Step function used for approximate edge triggering. """
        edge_strength = self.cfg.edge_strength
        N = self._buffer_nsamp
        halfN = N // 2

        step = np.empty(N, dtype=FLOAT)  # type: np.ndarray[FLOAT]
        step[:halfN] = -edge_strength / 2
        step[halfN:] = edge_strength / 2
        step *= windows.gaussian(N, std=halfN / 3)
        return step

    def _calc_data_taper(self):
        """ Input data window. Zeroes out all data older than 1 frame old.
        See https://github.com/nyanpasu64/ovgenpy/wiki/Correlation-Trigger
        """
        N = self._buffer_nsamp
        halfN = N // 2
        nsamp_frame = self._nsamp_frame

        # Left half of a Hann cosine taper
        taper = windows.hann(nsamp_frame * 2)[:nsamp_frame]

        # Reshape taper to left `halfN` of data_window (right-aligned).
        taper = leftpad(taper, halfN)

        # Generate left half-taper to prevent correlating with 1-frame-old data.
        data_window = np.ones(N)
        data_window[:halfN] = np.minimum(data_window[:halfN], taper)
        return data_window

    def get_trigger(self, index: int) -> int:
        """
        :param index: sample index
        :return: new sample index, corresponding to rising edge
        """
        N = self._buffer_nsamp
        use_edge_trigger = self.cfg.use_edge_trigger

        # Get data
        data = self._wave.get_around(index, N, self._subsampling)

        # Window data
        period = get_period(data)
        print(period)
        diameter, falloff = [round(period * x) for x in self.cfg.trigger_falloff]
        falloff_window = cosine_flat(N, diameter, falloff)

        window = np.minimum(falloff_window, self._data_taper)
        data *= window
        self.lol = window

        # prev_buffer
        prev_buffer = self._windowed_step + self._buffer

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
        radius = round(N * self.cfg.trigger_diameter / 2)

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
        buffer_falloff = self.cfg.buffer_falloff
        responsiveness = self.cfg.responsiveness

        N = len(data)
        if N != self._buffer_nsamp:
            raise ValueError(f'invalid data length {len(data)} does not match '
                             f'CorrelationTrigger {self._buffer_nsamp}')

        # New waveform
        self._normalize_buffer(data)

        wave_period = get_period(data)
        window = windows.gaussian(N, std = wave_period * buffer_falloff)
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


def cosine_flat(n: int, diameter: int, falloff: int):
    cosine = windows.hann(falloff * 2)
    left, right = cosine[:falloff], cosine[falloff:]

    window = np.concatenate([left, np.ones(diameter), right])

    padded = midpad(window, n)
    return padded


def lerp(x: np.ndarray, y: np.ndarray, a: float):
    return x * (1 - a) + y * a


# ZeroCrossingTrigger

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


# NullTrigger

@register_config
class NullTriggerConfig(ITriggerConfig):
    pass


@register_trigger(NullTriggerConfig)
class NullTrigger(Trigger):
    def get_trigger(self, index: int) -> int:
        return index
