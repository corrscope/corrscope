import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, Tuple, Optional, ClassVar

import numpy as np
from scipy import signal
from scipy.signal import windows

from ovgenpy.config import register_config, OvgenError, Alias
from ovgenpy.util import find, obj_name
from ovgenpy.utils.keyword_dataclasses import dataclass
from ovgenpy.utils.windows import midpad, leftpad
from ovgenpy.wave import FLOAT


if TYPE_CHECKING:
    from ovgenpy.wave import Wave

# Abstract classes

@dataclass
class ITriggerConfig:
    cls: ClassVar[Type['Trigger']]

    # Optional trigger for postprocessing
    post: 'ITriggerConfig' = None

    def __call__(self, wave: 'Wave', tsamp: int, stride: int, fps: float) \
            -> 'Trigger':
        return self.cls(wave, cfg=self, tsamp=tsamp, stride=stride, fps=fps)


def register_trigger(config_t: Type[ITriggerConfig]):
    """ @register_trigger(FooTriggerConfig)
    def FooTrigger(): ...
    """
    def inner(trigger_t: Type[Trigger]):
        config_t.cls = trigger_t
        return trigger_t

    return inner


class Trigger(ABC):
    POST_PROCESSING_NSAMP = 256

    def __init__(self, wave: 'Wave', cfg: ITriggerConfig, tsamp: int, stride: int,
                 fps: float):
        self.cfg = cfg
        self._wave = wave

        # TODO rename tsamp to buffer_nsamp
        self._tsamp = tsamp
        self._stride = stride
        self._fps = fps

        frame_dur = 1 / fps
        # Subsamples per frame
        self._tsamp_frame = self.time2tsamp(frame_dur)
        # Samples per frame
        self._real_samp_frame = round(frame_dur * self._wave.smp_s)

        # TODO rename to post_trigger
        if cfg.post:
            # Create a post-processing trigger, with narrow nsamp and stride=1.
            # This improves speed and precision.
            self.post = cfg.post(wave, self.POST_PROCESSING_NSAMP, 1, fps)
        else:
            self.post = None

    def time2tsamp(self, time: float):
        return round(time * self._wave.smp_s / self._stride)

    @abstractmethod
    def get_trigger(self, index: int, cache: 'PerFrameCache') -> int:
        """
        :param index: sample index
        :param cache: Information shared across all stacked triggers,
            May be mutated by function.
        :return: new sample index, corresponding to rising edge
        """
        ...


@dataclass
class PerFrameCache:
    """
    The estimated period of a wave region (Wave.get_around())
    is approximately constant, even when multiple triggers are stacked
    and each is called at a slightly different point.

    For each unique (frame, channel), all stacked triggers are passed the same
    TriggerFrameCache object.
    """

    # NOTE: period is a *non-subsampled* period.
    # The period of subsampled data must be multiplied by stride.
    period: Optional[int] = None
    mean: Optional[float] = None


# CorrelationTrigger

@register_config(always_dump='''
    use_edge_trigger
    edge_strength
    responsiveness
    buffer_falloff
''')
class CorrelationTriggerConfig(ITriggerConfig):
    # get_trigger
    edge_strength: float = 10.0
    trigger_diameter: float = 0.5

    trigger_falloff: Tuple[float, float] = (4.0, 1.0)
    recalc_semitones: float = 1.0
    lag_prevention: float = 0.25

    # _update_buffer
    responsiveness: float = 0.1
    buffer_falloff: float = 0.5  # Gaussian std = wave_period * buffer_falloff

    # region Legacy Aliases
    trigger_strength = Alias('edge_strength')
    falloff_width = Alias('buffer_falloff')

    # Problem: InitVar with default values are (wrongly) accessible on object instances.
    # use_edge_trigger is False but self.use_edge_trigger is True, wtf?
    use_edge_trigger: bool = True
    # endregion

    def __post_init__(self):
        self._validate_param('lag_prevention', 0, 1)
        self._validate_param('responsiveness', 0, 1)
        # TODO trigger_falloff >= 0
        self._validate_param('buffer_falloff', 0, np.inf)

        if self.use_edge_trigger:
            if self.post:
                warnings.warn(
                    "Ignoring old `CorrelationTriggerConfig.use_edge_trigger` flag, "
                    "overriden by newer `post` flag."
                )
            else:
                self.post = ZeroCrossingTriggerConfig()

    def _validate_param(self, key: str, begin, end):
        value = getattr(self, key)
        if not begin <= value <= end:
            raise ValueError(
                f'Invalid {key}={value} (should be within [{begin}, {end}])')


@register_trigger(CorrelationTriggerConfig)
class CorrelationTrigger(Trigger):
    cfg: CorrelationTriggerConfig

    def __init__(self, *args, **kwargs):
        """
        Correlation-based trigger which looks at a window of `trigger_tsamp` samples.
        it's complicated
        """
        Trigger.__init__(self, *args, **kwargs)
        self._buffer_nsamp = self._tsamp

        # Create correlation buffer (containing a series of old data)
        self._buffer = np.zeros(self._buffer_nsamp, dtype=FLOAT)    # type: np.ndarray[FLOAT]

        # Precompute edge trigger step
        self._windowed_step = self._calc_step()

        # Input data taper (zeroes out all data older than 1 frame old)
        self._data_taper = self._calc_data_taper()  # Rejected idea: right cosine taper

        # Will be overwritten on the first frame.
        self._prev_period = None
        self._prev_window = None

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
        See https://github.com/jimbo1qaz/ovgenpy/wiki/Correlation-Trigger
        """
        N = self._buffer_nsamp
        halfN = N // 2

        # - Create a cosine taper of `width` <= 1 frame
        # - Right-pad(value=1, len=1 frame)
        # - Place in left half of N-sample buffer.

        # To avoid cutting off data, use a narrow transition zone (invariant to stride).
        # _real_samp_frame (unit=subsample) == stride * frame.
        transition_nsamp = round(self._real_samp_frame * self.cfg.lag_prevention)
        tsamp_frame = self._tsamp_frame

        # Left half of a Hann cosine taper
        # Width (type=subsample) = min(stride*frame * lag_prevention, 1 frame)
        width = min(transition_nsamp, tsamp_frame)
        taper = windows.hann(width * 2)[:width]

        # Right-pad taper to 1 frame long
        if width < tsamp_frame:
            taper = np.pad(taper, (0, tsamp_frame - width), 'constant',
                           constant_values=1)
        assert len(taper) == tsamp_frame

        # Reshape taper to left `halfN` of data_window (right-aligned).
        taper = leftpad(taper, halfN)

        # Generate left half-taper to prevent correlating with 1-frame-old data.
        data_window = np.ones(N)
        data_window[:halfN] = np.minimum(data_window[:halfN], taper)

        return data_window

    def get_trigger(self, index: int, cache: 'PerFrameCache') -> int:
        N = self._buffer_nsamp

        # Get data
        stride = self._stride
        data = self._wave.get_around(index, N, stride)
        cache.mean = np.mean(data)
        data -= cache.mean

        # Window data
        period = get_period(data)
        cache.period = period * stride

        if self._is_window_invalid(period):
            diameter, falloff = [round(period * x) for x in self.cfg.trigger_falloff]
            falloff_window = cosine_flat(N, diameter, falloff)
            window = np.minimum(falloff_window, self._data_taper)

            self._prev_period = period
            self._prev_window = window
        else:
            window = self._prev_window

        data *= window

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

        # Find optimal offset (within trigger_diameter, default=±N/4)
        mid = N-1
        radius = round(N * self.cfg.trigger_diameter / 2)

        left = mid - radius
        right = mid + radius + 1

        corr = corr[left:right]
        mid = mid - left

        # argmax(corr) == mid + peak_offset == (data >> peak_offset)
        # peak_offset == argmax(corr) - mid
        peak_offset = np.argmax(corr) - mid   # type: int
        trigger = index + (stride * peak_offset)

        # Apply post trigger (before updating correlation buffer)
        if self.post:
            trigger = self.post.get_trigger(trigger, cache)

        # Update correlation buffer (distinct from visible area)
        aligned = self._wave.get_around(trigger, self._buffer_nsamp, stride)
        self._update_buffer(aligned, cache)

        return trigger

    def _is_window_invalid(self, period):
        """ Returns True if pitch has changed more than `recalc_semitones`. """

        prev = self._prev_period

        if prev is None:
            return True
        elif prev * period == 0:
            return prev != period
        else:
            semitones = abs(np.log(period/prev) / np.log(2) * 12)

            # If semitones == recalc_semitones == 0, do NOT recalc.
            if semitones <= self.cfg.recalc_semitones:
                return False
            return True

    def _update_buffer(self, data: np.ndarray, cache: PerFrameCache) -> None:
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
        data -= cache.mean
        normalize_buffer(data)
        window = windows.gaussian(N, std =
            (cache.period / self._stride) * buffer_falloff)
        data *= window

        # Old buffer
        normalize_buffer(self._buffer)
        self._buffer = lerp(self._buffer, data, responsiveness)


# get_trigger()

def calc_step(nsamp: int, peak: float, stdev: float):
    """ Step function used for approximate edge triggering.
    TODO deduplicate CorrelationTrigger._calc_step() """
    N = nsamp
    halfN = N // 2

    step = np.empty(N, dtype=FLOAT)  # type: np.ndarray[FLOAT]
    step[:halfN] = -peak / 2
    step[halfN:] = peak / 2
    step *= windows.gaussian(N, std=halfN * stdev)
    return step


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
    return int(peakX)


def cosine_flat(n: int, diameter: int, falloff: int):
    cosine = windows.hann(falloff * 2)
    left, right = cosine[:falloff], cosine[falloff:]

    window = np.concatenate([left, np.ones(diameter), right])

    padded = midpad(window, n)
    return padded


# update_buffer()

MIN_AMPLITUDE = 0.01

def normalize_buffer(data: np.ndarray) -> None:
    """
    Rescales `data` in-place.
    """
    peak = np.amax(abs(data))
    data /= max(peak, MIN_AMPLITUDE)


def lerp(x: np.ndarray, y: np.ndarray, a: float):
    return x * (1 - a) + y * a


#### Post-processing triggers

class PostTrigger(Trigger, ABC):
    """ A post-processing trigger should have stride=1,
     and no more post triggers. This is subject to change. """
    def __init__(self, *args, **kwargs):
        Trigger.__init__(self, *args, **kwargs)

        if self._stride != 1:
            raise OvgenError(
                f'{obj_name(self)} with stride != 1 is not allowed '
                f'(supplied {self._stride})')

        if self.post:
            raise OvgenError(
                f'Passing {obj_name(self)} a post_trigger is not allowed '
                f'({obj_name(self.post)})'
            )


# Local edge-finding trigger

@register_config(always_dump='strength')
class LocalPostTriggerConfig(ITriggerConfig):
    strength: float  # Coefficient

@register_trigger(LocalPostTriggerConfig)
class LocalPostTrigger(PostTrigger):
    cfg: LocalPostTriggerConfig

    def __init__(self, *args, **kwargs):
        PostTrigger.__init__(self, *args, **kwargs)
        self._buffer_nsamp = self._tsamp

        # Precompute data window... TODO Hann, or extract fancy dynamic-width from CorrelationTrigger?
        self._data_window = windows.hann(self._buffer_nsamp).astype(FLOAT)

        # Precompute edge correlation buffer
        self._windowed_step = calc_step(self._tsamp, self.cfg.strength, 1/3)

        # Precompute normalized _cost_norm function
        N = self._buffer_nsamp
        corr_len = 2*N - 1
        self._cost_norm = (np.arange(corr_len, dtype=FLOAT) - N) ** 2

    def get_trigger(self, index: int, cache: 'PerFrameCache') -> int:
        N = self._buffer_nsamp

        # Get data
        data = self._wave.get_around(index, N, self._stride)
        data -= cache.mean
        normalize_buffer(data)
        data *= self._data_window

        # Window data
        if cache.period is None:
            raise ValueError(
                "Missing 'cache.period', try stacking CorrelationTrigger "
                "before LocalPostTrigger")

        # To avoid sign errors, see comment in CorrelationTrigger.get_trigger().
        corr = signal.correlate(data, self._windowed_step)
        assert len(corr) == 2*N - 1
        mid = N-1

        # If we're near a falling edge, don't try to make drastic changes.
        if corr[mid] < 0:
            # Give up early.
            return index

        # Don't punish negative results too much.
        # (probably useless. if corr[mid] >= 0,
        # all other negative entries will never be optimal.)
        # np.abs(corr, out=corr)

        # Subtract cost function
        cost = self._cost_norm / cache.period
        corr -= cost

        # Find optimal offset (within ±N/4)
        mid = N-1
        radius = round(N / 4)

        left = mid - radius
        right = mid + radius + 1

        corr = corr[left:right]
        mid = mid - left

        peak_offset = np.argmax(corr) - mid   # type: int
        trigger = index + (self._stride * peak_offset)

        return trigger

def seq_along(a: np.ndarray):
    return np.arange(len(a))

# ZeroCrossingTrigger

@register_config
class ZeroCrossingTriggerConfig(ITriggerConfig):
    pass


@register_trigger(ZeroCrossingTriggerConfig)
class ZeroCrossingTrigger(PostTrigger):
    # ZeroCrossingTrigger is only used as a postprocessing trigger.
    # stride is only passed 1, for improved precision.

    def get_trigger(self, index: int, cache: 'PerFrameCache'):
        # 'cache' is unused.
        tsamp = self._tsamp

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

        data = self._wave[index : index + (direction * tsamp) : direction]
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
    def get_trigger(self, index: int, cache: 'PerFrameCache') -> int:
        return index
