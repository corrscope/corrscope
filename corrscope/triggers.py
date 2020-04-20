from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, Optional, ClassVar, Union, TypeVar, Generic

import attr
import numpy as np

import corrscope.utils.scipy.signal as signal
import corrscope.utils.scipy.windows as windows
from corrscope.config import KeywordAttrs, CorrError, Alias, with_units
from corrscope.spectrum import SpectrumConfig, DummySpectrum, LogFreqSpectrum
from corrscope.util import find, obj_name, iround
from corrscope.utils.trigger_util import (
    get_period,
    normalize_buffer,
    lerp,
    MIN_AMPLITUDE,
    abs_max,
)
from corrscope.utils.windows import leftpad, midpad, rightpad, gaussian_or_zero
from corrscope.wave import FLOAT

if TYPE_CHECKING:
    from corrscope.wave import Wave
    from corrscope.renderer import RendererFrontend


# Abstract classes


class _TriggerConfig:
    cls: ClassVar[Type["_Trigger"]]

    def __call__(self, wave: "Wave", *args, **kwargs) -> "_Trigger":
        return self.cls(wave, self, *args, **kwargs)


class MainTriggerConfig(
    _TriggerConfig, KeywordAttrs, always_dump="edge_direction post_trigger post_radius"
):
    if TYPE_CHECKING:

        def __call__(self, wave: "Wave", *args, **kwargs) -> "MainTrigger":
            return self.cls(wave, self, *args, **kwargs)

    # Must be 1 or -1.
    # MainTrigger.__init__() multiplies `wave.amplification *= edge_direction`.
    # get_trigger() should ignore `edge_direction` and look for rising edges.
    edge_direction: int = 1

    # Optional trigger for postprocessing
    post_trigger: Optional["PostTriggerConfig"] = None
    post_radius: Optional[int] = with_units("smp", default=3)

    def __attrs_post_init__(self):
        if self.edge_direction not in [-1, 1]:
            raise CorrError(f"{obj_name(self)}.edge_direction must be {{-1, 1}}")

        if self.post_trigger:
            self.post_trigger.parent = self
            if self.post_radius is None:
                name = obj_name(self)
                raise CorrError(
                    f"Cannot supply {name}.post_trigger without supplying {name}.post_radius"
                )


class PostTriggerConfig(_TriggerConfig, KeywordAttrs):
    parent: MainTriggerConfig = attr.ib(init=False)  # TODO Unused

    if TYPE_CHECKING:

        def __call__(self, wave: "Wave", *args, **kwargs) -> "PostTrigger":
            return self.cls(wave, self, *args, **kwargs)


def register_trigger(config_t: Type[_TriggerConfig]):
    """ @register_trigger(FooTriggerConfig)
    def FooTrigger(): ...
    """

    def inner(trigger_t: Type[_Trigger]):
        config_t.cls = trigger_t
        return trigger_t

    return inner


result = TypeVar("result")


class _Trigger(ABC, Generic[result]):
    def __init__(
        self,
        wave: "Wave",
        cfg: _TriggerConfig,
        tsamp: int,
        stride: int,
        fps: float,
        renderer: Optional["RendererFrontend"] = None,
        wave_idx: int = 0,
    ):
        self.cfg = cfg
        self._wave = wave

        # TODO rename tsamp to buffer_nsamp
        self._tsamp = tsamp
        self._stride = stride
        self._fps = fps

        # Only used for debug plots
        self._renderer = renderer
        self._wave_idx = wave_idx

        # Subsamples per second
        self.subsmp_s = self._wave.smp_s / self._stride

        frame_dur = 1 / fps
        # Subsamples per frame
        self._tsamp_frame = self.time2tsamp(frame_dur)

    def time2tsamp(self, time: float) -> int:
        return round(time * self.subsmp_s)

    def custom_line(
        self, name: str, data: np.ndarray, offset: bool, invert: bool = True
    ):
        """
        :param offset:
        - True, for untriggered wave data:
            - line will be shifted and triggered (by offset_viewport()).
        - False, for triggered data and buffers:
            - line is immune to offset_viewport().

        :param invert:
        - True, for wave (data and buffers):
            - If wave data is inverted (edge_direction = -1),
              data will be plotted inverted.
        - False, for buffers and autocorrelated wave data:
            - Data is plotted as-is.
        """
        if self._renderer is None:
            return
        data = data / abs_max(data, 0.01) / 2
        if invert:
            data *= np.copysign(1, self._wave.amplification)
        self._renderer.update_custom_line(
            name, self._wave_idx, self._stride, data, offset=offset
        )

    def custom_vline(self, name: str, x: int, offset: bool):
        """See above for `offset`."""
        if self._renderer is None:
            return
        self._renderer.update_vline(
            name, self._wave_idx, self._stride, x, offset=offset
        )

    def offset_viewport(self, offset: int):
        if self._renderer is None:
            return
        self._renderer.offset_viewport(self._wave_idx, offset)

    @abstractmethod
    def get_trigger(self, index: int, cache: "PerFrameCache") -> result:
        """
        :param index: sample index
        :param cache: Information shared across all stacked triggers,
            May be mutated by function.
        :return: new sample index, corresponding to rising edge
        """
        ...

    @abstractmethod
    def do_not_inherit__Trigger_directly(self):
        pass


@attr.dataclass
class TriggerResult:
    # new sample index, corresponding to rising edge
    result: int

    # Estimated frequency in cycle/sec (Hertz). None if unknown.
    freq_estimate: Optional[float]


class MainTrigger(_Trigger[TriggerResult], ABC):
    cfg: MainTriggerConfig
    post: Optional["PostTrigger"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wave.amplification *= self.cfg.edge_direction

        cfg = self.cfg
        if cfg.post_trigger:
            # Create a post-processing trigger, with narrow nsamp and stride=1.
            # This improves speed and precision.
            self.post = cfg.post_trigger(
                self._wave,
                cfg.post_radius,
                1,
                self._fps,
                self._renderer,
                self._wave_idx,
            )
        else:
            self.post = None

    def set_renderer(self, renderer: "RendererFrontend"):
        self._renderer = renderer
        if self.post:
            self.post._renderer = renderer

    def do_not_inherit__Trigger_directly(self):
        pass


class PostTrigger(_Trigger[int], ABC):
    """ A post-processing trigger should have stride=1,
     and no more post triggers. This is subject to change. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._stride != 1:
            raise CorrError(
                f"{obj_name(self)} with stride != 1 is not allowed "
                f"(supplied {self._stride})"
            )

    def do_not_inherit__Trigger_directly(self):
        pass


@attr.dataclass(slots=True)
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


class LagPrevention(KeywordAttrs):
    max_frames: float = 1
    transition_frames: float = 0.25

    def __attrs_post_init__(self):
        validate_param(self, "max_frames", 0, 1)
        validate_param(self, "transition_frames", 0, self.max_frames)


class CorrelationTriggerConfig(
    MainTriggerConfig,
    always_dump="""
    pitch_tracking
    slope_strength slope_width
    """
    # deprecated
    " buffer_falloff ",
):
    # get_trigger()
    # Edge/area finding
    sign_strength: float = 0
    edge_strength: float

    # Slope detection
    slope_strength: float = 0
    slope_width: float = with_units("period", default=0.07)

    # Correlation detection (meow~ =^_^=)
    buffer_strength: float = 1

    # Both data and buffer uses Gaussian windows. std = wave_period * falloff.
    # get_trigger()
    data_falloff: float = 1.5

    # _update_buffer()
    buffer_falloff: float = 0.5

    # Maximum distance to move
    trigger_diameter: Optional[float] = 0.5

    recalc_semitones: float = 1.0
    lag_prevention: LagPrevention = attr.ib(factory=LagPrevention)

    # _update_buffer
    responsiveness: float

    # Period/frequency estimation (not in GUI)
    max_freq: float = with_units("Hz", default=4000)

    # Pitch tracking = compute spectrum.
    pitch_tracking: Optional["SpectrumConfig"] = None

    # region Legacy Aliases
    trigger_strength = Alias("edge_strength")
    falloff_width = Alias("buffer_falloff")
    # endregion

    def __attrs_post_init__(self) -> None:
        MainTriggerConfig.__attrs_post_init__(self)

        validate_param(self, "slope_width", 0, 0.5)

        validate_param(self, "responsiveness", 0, 1)
        # TODO trigger_falloff >= 0
        validate_param(self, "buffer_falloff", 0, np.inf)


def validate_param(self, key: str, begin: float, end: float) -> None:
    value = getattr(self, key)
    if not begin <= value <= end:
        raise CorrError(f"Invalid {key}={value} (should be within [{begin}, {end}])")


@register_trigger(CorrelationTriggerConfig)
class CorrelationTrigger(MainTrigger):
    """
    Assume that if get_trigger(x) == x, then data[[x-1, x]] == [<0, >0].
    - edge detectors [halfN = N//2] > 0.
    - So wave.get_around(x)[N//2] > 0.
    - So wave.get_around(x) = [x - N//2 : ...]

    test_trigger() checks that get_around() works properly, for even/odd N.
    See Wave.get_around() docstring.
    """

    cfg: CorrelationTriggerConfig

    @property
    def scfg(self) -> SpectrumConfig:
        return self.cfg.pitch_tracking

    def __init__(self, *args, **kwargs):
        """
        Correlation-based trigger which looks at a window of `trigger_tsamp` samples.
        it's complicated
        """
        super().__init__(*args, **kwargs)
        self._buffer_nsamp = self._tsamp

        # (const) Multiplied by each frame of input audio.
        # Zeroes out all data older than 1 frame old.
        self._lag_prevention_window = self._calc_lag_prevention()
        assert self._lag_prevention_window.dtype == FLOAT

        # (mutable) Correlated with data (for triggering).
        # Updated with tightly windowed old data at various pitches.
        self._buffer = np.zeros(
            self._buffer_nsamp, dtype=FLOAT
        )  # type: np.ndarray[FLOAT]

        # (const) Added to self._buffer. Nonzero if edge triggering is nonzero.
        # Left half is -edge_strength, right half is +edge_strength.
        # ASCII art: --._|‾'--
        self._edge_finder = self._calc_step()
        assert self._edge_finder.dtype == FLOAT

        # Will be overwritten on the first frame.
        self._prev_period: Optional[int] = None
        self._prev_window: Optional[np.ndarray] = None
        self._prev_slope_finder: Optional[np.ndarray] = None

        self._prev_trigger: int = 0

        # (mutable) Log-scaled spectrum
        self.frames_since_spectrum = 0

        if self.scfg:
            self._spectrum_calc = LogFreqSpectrum(
                scfg=self.scfg, subsmp_s=self.subsmp_s, dummy_data=self._buffer
            )
        else:
            self._spectrum_calc = DummySpectrum()

    def _calc_lag_prevention(self) -> np.ndarray:
        """ Returns input-data window,
        which zeroes out all data older than 1-ish frame old.
        See https://github.com/corrscope/corrscope/wiki/Correlation-Trigger
        """
        N = self._buffer_nsamp
        halfN = N // 2

        # - Create a cosine taper of `width` <= 1 frame
        # - Right-pad(value=1, len=1 frame)
        # - Place in left half of N-sample buffer.

        # To avoid cutting off data, use a narrow transition zone (invariant to stride).
        lag_prevention = self.cfg.lag_prevention
        tsamp_frame = self._tsamp_frame
        transition_nsamp = round(tsamp_frame * lag_prevention.transition_frames)

        # Left half of a Hann cosine taper
        # Width (type=subsample) = min(frame * lag_prevention, 1 frame)
        assert transition_nsamp <= tsamp_frame
        width = transition_nsamp
        taper = windows.hann(width * 2)[:width]

        # Right-pad=1 taper to lag_prevention.max_frames long [t-#*f, t]
        taper = rightpad(taper, iround(tsamp_frame * lag_prevention.max_frames))

        # Left-pad=0 taper to left `halfN` of data_taper [t-halfN, t]
        taper = leftpad(taper, halfN)

        # Generate left half-taper to prevent correlating with 1-frame-old data.
        # Right-pad=1 taper to [t-halfN, t-halfN+N]
        # TODO switch to rightpad()? Does it return FLOAT or not?
        data_taper = np.ones(N, dtype=FLOAT)
        data_taper[:halfN] = np.minimum(data_taper[:halfN], taper)

        return data_taper

    def _calc_step(self) -> np.ndarray:
        """ Step function used for approximate edge triggering. """

        # Increasing buffer_falloff (width of buffer)
        # causes buffer to affect triggering, more than the step function.
        # So we multiply edge_strength (step function height) by buffer_falloff.

        cfg = self.cfg
        edge_strength = cfg.edge_strength * cfg.buffer_falloff

        N = self._buffer_nsamp
        halfN = N // 2

        step = np.empty(N, dtype=FLOAT)  # type: np.ndarray[FLOAT]
        step[:halfN] = -edge_strength / 2
        step[halfN:] = edge_strength / 2
        step *= windows.gaussian(N, std=halfN / 3)
        return step

    def _calc_slope_finder(self, period: float) -> np.ndarray:
        """ Called whenever period changes substantially.
        Returns a kernel to be correlated with input data,
        to find positive slopes."""

        N = self._buffer_nsamp
        halfN = N // 2
        slope_finder = np.zeros(N)

        cfg = self.cfg
        slope_width = max(iround(cfg.slope_width * period), 1)
        slope_strength = cfg.slope_strength * cfg.buffer_falloff

        slope_finder[halfN - slope_width : halfN] = -slope_strength
        slope_finder[halfN : halfN + slope_width] = slope_strength
        return slope_finder

    # end setup

    # begin per-frame
    def get_trigger(self, index: int, cache: "PerFrameCache") -> TriggerResult:
        N = self._buffer_nsamp
        cfg = self.cfg

        # Get data (1D, downmixed to mono)
        stride = self._stride
        data = self._wave.get_around(index, N, stride)

        if cfg.sign_strength != 0:
            signs = sign_times_peak(data)
            data += cfg.sign_strength * signs

        # Remove mean from data
        data -= np.add.reduce(data) / N

        # Window data
        period = get_period(data, self.subsmp_s, self.cfg.max_freq, self)
        cache.period = period * stride

        semitones = self._is_window_invalid(period)
        # If pitch changed...
        if semitones:
            # Gaussian window
            period_symmetric_window = gaussian_or_zero(N, period * cfg.data_falloff)

            # Left-sided falloff
            lag_prevention_window = self._lag_prevention_window

            # Both combined.
            window = np.minimum(period_symmetric_window, lag_prevention_window)

            # Slope finder
            slope_finder = self._calc_slope_finder(period)

            data *= window

            # If pitch tracking enabled, rescale buffer to match data's pitch.
            if self.scfg and (data != 0).any():
                # Mutates self._buffer.
                self.spectrum_rescale_buffer(data)

            self._prev_period = period
            self._prev_window = window
            self._prev_slope_finder = slope_finder
        else:
            window = self._prev_window
            slope_finder = self._prev_slope_finder

            data *= window

        prev_buffer: np.ndarray = self._buffer * self.cfg.buffer_strength
        prev_buffer += self._edge_finder + slope_finder

        # Calculate correlation
        if self.cfg.trigger_diameter is not None:
            radius = round(N * self.cfg.trigger_diameter / 2)
        else:
            radius = None

        trigger_score = correlate_data(data, prev_buffer, radius)
        peak_offset = trigger_score.peak
        trigger = index + (stride * peak_offset)

        del data

        if self.post:
            new_data = self._wave.get_around(trigger, N, stride)
            cache.mean = np.add.reduce(new_data) / N

            # Apply post trigger (before updating correlation buffer)
            trigger = self.post.get_trigger(trigger, cache)

        # Avoid time traveling backwards.
        self._prev_trigger = trigger = max(trigger, self._prev_trigger)

        # Update correlation buffer (distinct from visible area)
        aligned = self._wave.get_around(trigger, N, stride)
        if cache.mean is None:
            cache.mean = np.add.reduce(aligned) / N
        self._update_buffer(aligned, cache)

        self.frames_since_spectrum += 1

        self.offset_viewport(peak_offset)

        # period: subsmp/cyc
        freq_estimate = self.subsmp_s / period if period else None
        # freq_estimate: cyc/s
        # If period is 0 (unknown), freq_estimate is None.

        return TriggerResult(trigger, freq_estimate)

    def spectrum_rescale_buffer(self, data: np.ndarray) -> None:
        """
        - Cross-correlate the log-frequency spectrum of `data` with `buffer`.
        - Rescale `buffer` until its pitch matches `data`.
        """

        # Setup
        scfg = self.scfg
        N = self._buffer_nsamp
        if self.frames_since_spectrum < self.scfg.min_frames_between_recompute:
            return
        self.frames_since_spectrum = 0

        calc_spectrum = self._spectrum_calc.calc_spectrum

        # Compute log-frequency spectrum of `data`.
        spectrum = calc_spectrum(data)
        normalize_buffer(spectrum)
        assert not np.any(np.isnan(spectrum))

        # Compute log-frequency spectrum of `self._buffer`.
        prev_spectrum = calc_spectrum(self._buffer)
        # Don't normalize self._spectrum. It was already normalized when being assigned.

        # Rescale `self._buffer` until its pitch matches `data`.
        resample_notes = correlate_spectrum(
            spectrum, prev_spectrum, scfg.max_notes_to_resample
        ).peak
        if resample_notes != 0:
            # If we want to double pitch, we must divide data length by 2.
            new_len = iround(N / 2 ** (resample_notes / scfg.notes_per_octave))

            def rescale_mut(in_buf):
                buf = np.interp(
                    np.linspace(0, 1, new_len), np.linspace(0, 1, N), in_buf
                )
                # assert len(buf) == new_len
                buf = midpad(buf, N)
                in_buf[:] = buf

            # Copy+resample self._buffer.
            rescale_mut(self._buffer)

    def _is_window_invalid(self, period: int) -> Union[bool, float]:
        """
        Returns number of semitones,
        if pitch has changed more than `recalc_semitones`.

        Preconditions:
        - self._prev_period is assigned whenever this function returns True.
        - If period cannot be estimated, period == 0.

        Postconditions:
        - On frame 0, MUST return True (to initialize self._prev_window).
            - This is the only way self._prev_period == 0.
        - Elif period is 0 (cannot be estimated), return False.
        """

        prev = self._prev_period
        if prev is None:
            return True
        elif period == 0:
            return False
        elif prev == 0:
            return True
        else:
            # When period doubles, semitones are -12.
            semitones = np.log(period / prev) / np.log(2) * -12
            # If semitones == recalc_semitones == 0, do NOT recalc.
            if abs(semitones) <= self.cfg.recalc_semitones:
                return False
            return semitones

    def _update_buffer(self, data: np.ndarray, cache: PerFrameCache) -> None:
        """
        Update self._buffer by adding `data` and a step function.
        Data is reshaped to taper away from the center.

        :param data: Wave data. WILL BE MODIFIED.
        """
        assert cache.mean is not None
        assert cache.period is not None
        buffer_falloff = self.cfg.buffer_falloff
        responsiveness = self.cfg.responsiveness

        N = len(data)
        if N != self._buffer_nsamp:
            raise ValueError(
                f"invalid data length {len(data)} does not match "
                f"CorrelationTrigger {self._buffer_nsamp}"
            )

        # New waveform
        data -= cache.mean
        normalize_buffer(data)
        window = gaussian_or_zero(N, std=(cache.period / self._stride) * buffer_falloff)
        data *= window

        # Old buffer
        normalize_buffer(self._buffer)
        self._buffer = lerp(self._buffer, data, responsiveness)


@attr.dataclass
class CorrelationResult:
    peak: int
    corr: np.ndarray


@attr.dataclass
class InterpolatedCorrelationResult:
    peak: float
    corr: np.ndarray


def correlate_data(
    data: np.ndarray, prev_buffer: np.ndarray, radius: Optional[int]
) -> CorrelationResult:
    """
    This is confusing.

    If data index < optimal, data will be too far to the right,
    and we need to `index += positive`.
    - The peak will appear near the right of `data`.

    Either we must slide prev_buffer to the right,
    or we must slide data to the left (by sliding index to the right):
    - correlate(data, prev_buffer)
    - trigger = index + peak_offset
    """
    N = len(data)
    corr = signal.correlate(data, prev_buffer)  # returns double, not single/FLOAT
    Ncorr = 2 * N - 1
    assert len(corr) == Ncorr

    # Find optimal offset
    mid = N - 1

    if radius is not None:
        left = max(mid - radius, 0)
        right = min(mid + radius + 1, Ncorr)

        corr = corr[left:right]
        mid = mid - left

    # argmax(corr) == mid + peak_offset == (data >> peak_offset)
    # peak_offset == argmax(corr) - mid
    peak_offset = np.argmax(corr) - mid  # type: int
    return CorrelationResult(peak_offset, corr)


def correlate_spectrum(
    data: np.ndarray, prev_buffer: np.ndarray, radius: Optional[int]
) -> CorrelationResult:
    """
    I used to use parabolic() on the return value,
    but unfortunately it was unreliable and caused Plok Beach bass to jitter,
    so I turned it off (resulting in the same code as correlate_data).
    """
    return correlate_data(data, prev_buffer, radius)


def parabolic(xint: int, ys: np.ndarray) -> float:
    """
    Quadratic interpolation for estimating the true position of an inter-sample maximum
    when nearby samples are known.
    """

    if xint - 1 < 0 or xint + 1 >= len(ys):
        return float(xint)

    left = ys[xint - 1]
    mid = ys[xint]
    right = ys[xint + 1]

    # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    dx = 0.5 * (+left - right) / (+left - 2 * mid + right)
    assert -1 < dx < 1
    return xint + dx


SIGN_AMPLIFICATION = 1000


def sign_times_peak(data: np.ndarray) -> np.ndarray:
    """
    Computes peak = max(abs(data)).
    Returns `peak` for positive parts of data, and `-peak` for negative parts,
    and heavily amplifies parts of the wave near zero.
    """
    data = data.copy()

    peak = abs_max(data)
    data *= SIGN_AMPLIFICATION / (peak + MIN_AMPLITUDE)

    sign_data = np.tanh(data)
    sign_data *= peak

    return sign_data


#### Post-processing triggers
# ZeroCrossingTrigger


class ZeroCrossingTriggerConfig(PostTriggerConfig):
    pass


# Edge finding trigger
@register_trigger(ZeroCrossingTriggerConfig)
class ZeroCrossingTrigger(PostTrigger):
    # ZeroCrossingTrigger is only used as a postprocessing trigger.
    # stride is only passed 1, for improved precision.
    cfg: ZeroCrossingTriggerConfig

    def get_trigger(self, index: int, cache: "PerFrameCache") -> int:
        radius = self._tsamp

        wave = self._wave.with_offset(-cache.mean)

        if not 0 <= index < wave.nsamp:
            return index

        if wave[index] < 0:
            direction = 1
            test = lambda a: a >= 0

        elif wave[index] > 0:
            direction = -1
            test = lambda a: a <= 0

        else:  # self._wave[sample] == 0
            return index + 1

        data = wave[index : index + direction * (radius + 1) : direction]
        # TODO remove unnecessary complexity, since diameter is probably under 10.
        intercepts = find(data, test)
        try:
            (delta,), value = next(intercepts)
            return index + (delta * direction) + int(value <= 0)

        except StopIteration:  # No zero-intercepts
            return index + (direction * radius)

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


class NullTriggerConfig(MainTriggerConfig):
    pass


@register_trigger(NullTriggerConfig)
class NullTrigger(MainTrigger):
    def get_trigger(self, index: int, cache: "PerFrameCache") -> TriggerResult:
        return TriggerResult(index, None)
