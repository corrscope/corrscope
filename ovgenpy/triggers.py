from abc import ABC, abstractmethod
from typing import NamedTuple, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ovgenpy.renderer import MatplotlibRenderer, RendererConfig

if TYPE_CHECKING:
    from ovgenpy.wave import Wave


class Trigger(ABC):
    def __init__(self, wave: 'Wave', scan_nsamp: int):
        self.wave = wave
        self.scan_nsamp = scan_nsamp

    @abstractmethod
    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        ...


class TriggerConfig:
    # https://github.com/python/typing/issues/427
    def __call__(self, wave: 'Wave', scan_nsamp: int):
        raise NotImplementedError


SHOW_TRIGGER = True


def lerp(x: np.ndarray, y: np.ndarray, a: float):
    return x * (1 - a) + y * a


class CorrelationTrigger(Trigger):
    MIN_AMPLITUDE = 0.01

    @dataclass
    class Config(TriggerConfig):
        # get_trigger
        trigger_strength: float

        # _update_buffer
        responsiveness: float
        falloff_width: float

        def __call__(self, wave: 'Wave', scan_nsamp: int):
            return CorrelationTrigger(wave, scan_nsamp, cfg=self)

    def __init__(self, wave: 'Wave', scan_nsamp: int, cfg: Config):
        """
        Correlation-based trigger which looks at a window of `scan_nsamp` samples.

        it's complicated

        :param wave: Wave file
        :param scan_nsamp: Number of samples used to align adjacent frames
        :param cfg: Correlation config
        """
        Trigger.__init__(self, wave, scan_nsamp)
        self.buffer_nsamp = self.scan_nsamp

        # Correlation config
        self.cfg = cfg

        # Create correlation buffer (containing a series of old data)
        self._prev_buffer = np.zeros(scan_nsamp)    # FIXME always zero

        if SHOW_TRIGGER:
            self.trigger_renderer = TriggerRenderer(self)

    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        trigger_strength = self.cfg.trigger_strength

        data = self.wave.get_around(offset, self.buffer_nsamp)
        N = len(data)

        # Add "step function" to correlation buffer
        prev_buffer = self._prev_buffer.copy()
        prev_buffer[N//2:] += trigger_strength

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

    def _update_buffer(self, data: np.ndarray) -> None:
        """
        Update self._prev_buffer by adding `data` and a step function.
        Data is reshaped to taper away from the center.

        :param data: Wave data. WILL BE MODIFIED.
        """
        falloff_width = self.cfg.falloff_width
        responsiveness = self.cfg.responsiveness

        N = len(data)
        if N != self.buffer_nsamp:
            raise ValueError(f'invalid data length {len(data)} does not match '
                             f'CorrelationTrigger {self.buffer_nsamp}')

        # New waveform
        self._normalize_buffer(data)

        wave_period = get_period(data)
        window = signal.gaussian(N, std =wave_period * falloff_width)
        data *= window

        # Old buffer
        self._normalize_buffer(self._prev_buffer)
        self._prev_buffer = lerp(self._prev_buffer, data, responsiveness)

        if SHOW_TRIGGER:
            self.trigger_renderer.render_frame()

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

