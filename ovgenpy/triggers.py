from abc import ABC, abstractmethod
from typing import NamedTuple, List, Dict, Any, TYPE_CHECKING

import numpy as np
from scipy import signal

from ovgenpy.renderer import MatplotlibRenderer, RendererConfig

if TYPE_CHECKING:
    from ovgenpy.wave import Wave


class TriggerConfig(NamedTuple):
    # TODO specialize for CorrelationTrigger, remove args/kwargs
    name: str
    # scan_nsamp: int
    args: List = []
    kwargs: Dict[str, Any] = {}

    def generate_trigger(self, wave: 'Wave', scan_nsamp: int) -> 'Trigger':
        return TRIGGERS[self.name](wave, scan_nsamp, *self.args, **self.kwargs)


TRIGGERS: Dict[str, type] = {}


def register_trigger(trigger_class: type):
    TRIGGERS[trigger_class.__name__] = trigger_class
    return trigger_class


class Trigger(ABC):
    @abstractmethod
    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        ...


SHOW_TRIGGER = True


@register_trigger
class CorrelationTrigger(Trigger):
    MIN_AMPLITUDE = 0.01

    def __init__(self, wave: 'Wave', scan_nsamp: int, falloff_width: float,
                 trigger_strength: float):
        """
        Correlation-based trigger which looks at a window of `scan_nsamp` samples.

        it's complicated

        :param wave: Wave file
        :param scan_nsamp: Number of samples used to align adjacent frames

        :param falloff_width: Amount of previous wave to compare (in periods)
        :param trigger_strength: Amount of centering to apply to each frame, within [0, 1]
        """

        self.wave = wave
        self.buffer_nsamp = scan_nsamp
        # Correlation buffer calculation
        self.falloff_width = falloff_width
        # Wave triggering
        self.trigger_strength = trigger_strength

        # Correlation buffer containing a series of old data
        self._prev_buffer = np.zeros(scan_nsamp)
        self._update_buffer(self._prev_buffer)
        if SHOW_TRIGGER:
            self.trigger_renderer = TriggerRenderer(self)

    def _normalize_buffer(self, data: np.ndarray) -> None:
        """
        Rescales `data` in-place.
        """
        peak = np.amax(abs(data))
        data /= max(peak, self.MIN_AMPLITUDE)


    def _update_buffer(self, data: np.ndarray) -> None:
        """
        Update self._prev_buffer by adding `data` and a step function.
        Data is reshaped to taper away from the center.

        :param data: Wave data. WILL BE MODIFIED.
        """
        N = len(data)
        if N != self.buffer_nsamp:
            raise ValueError(f'invalid data length {len(data)} does not match '
                             f'CorrelationTrigger {self.buffer_nsamp}')

        # New waveform
        self._normalize_buffer(data)

        wave_period = get_period(data)
        window = signal.gaussian(N, std = wave_period * self.falloff_width)
        data *= window

        # Old buffer
        self._normalize_buffer(self._prev_buffer)
        self._prev_buffer += 0.5 * data     # FIXME parameter

        if SHOW_TRIGGER:
            self.trigger_renderer.render_frame()

    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        data = self.wave.get_around(offset, self.buffer_nsamp)
        N = len(data)

        # Add "step function" to correlation buffer
        prev_buffer = self._prev_buffer.copy()
        prev_buffer[N//2:] += self.trigger_strength

        # Find optimal offset (within ±N//4)
        delta = N-1
        radius = N//4

        # Calculate correlation
        corr = signal.correlate(prev_buffer, data)
        assert len(corr) == 2*N - 1
        corr = corr[delta-radius : delta+radius+1]
        delta = radius

        # argmax(corr) == delta + peak_offset == (data >> peak_offset)
        # peak_offset == argmax(corr) - delta
        peak_offset = np.argmax(corr) - delta   # type: int
        trigger = offset + peak_offset

        # Update correlation buffer (distinct from visible area)
        aligned = self.wave.get_around(trigger, self.buffer_nsamp)
        self._update_buffer(aligned)

        return trigger


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


if SHOW_TRIGGER:
    class TriggerRenderer(MatplotlibRenderer):
        # TODO swappable GraphRenderer class shouldn't depend on waves
        # probably don't need to debug multiple triggers
        def __init__(self, trigger: CorrelationTrigger):
            self.trigger = trigger
            cfg = RendererConfig(
                640, 360, trigger.buffer_nsamp, rows_first=False, ncols=1
            )
            super().__init__(cfg, [None])

        def render_frame(self) -> None:
            idx = 0

            # Draw trigger buffer data
            line = self.lines[idx]
            data = self.trigger._prev_buffer
            line.set_ydata(data)

