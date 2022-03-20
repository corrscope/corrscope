from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Type,
    Optional,
    ClassVar,
    Union,
    TypeVar,
    Generic,
    cast,
)

import attr
import numpy as np

import corrscope.utils.scipy.signal as signal
import corrscope.utils.scipy.windows as windows
from corrscope.config import KeywordAttrs, CorrError, Alias, with_units
from corrscope.spectrum import SpectrumConfig, DummySpectrum, LogFreqSpectrum
from corrscope.util import find, obj_name, iround
from corrscope.utils.trigger_util import (
    get_period,
    UNKNOWN_PERIOD,
    normalize_buffer,
    lerp,
    MIN_AMPLITUDE,
    abs_max,
)
from corrscope.utils.windows import midpad, gaussian_or_zero
from corrscope.wave_common import f32

if TYPE_CHECKING:
    import numpy.typing as npt
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
    """@register_trigger(FooTriggerConfig)
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

        # TODO rename _tsamp to _trigger_subsmp, tsamp to subsamp
        self._tsamp = tsamp
        self._stride = stride
        self._fps = fps

        # Only used for debug plots
        self._renderer = renderer
        self._wave_idx = wave_idx

        # Subsamples per second
        self.subsmp_per_s = self._wave.smp_s / self._stride

        seconds_per_frame = 1 / fps
        # Full samples per frame
        self._smp_per_frame = self.seconds_to_samp(seconds_per_frame)

    def seconds_to_samp(self, time: float) -> int:
        return round(time * self._wave.smp_s)

    def custom_line(
        self,
        name: str,
        data: np.ndarray,
        xs: np.ndarray,
        absolute: bool,
        invert: bool = True,
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
            name, self._wave_idx, self._stride, data, xs, absolute
        )

    def custom_vline(self, name: str, x: int, absolute: bool):
        """See above for `offset`."""
        if self._renderer is None:
            return
        self._renderer.update_vline(
            name, self._wave_idx, self._stride, x, absolute=absolute
        )

    @abstractmethod
    def get_trigger(self, index: int, cache: "PerFrameCache") -> result:
        """
        :param index: sample index
        :param cache: Information shared across all stacked triggers. result_mean is
            None. May be mutated by function.
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
    """A post-processing trigger should have stride=1,
    and no more post triggers. This is subject to change."""

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

    # The mean of the trigger input, smoothed by mean_responsiveness, used for regular
    # and post triggering.
    smoothed_mean: Optional[float] = None

    # The mean of the wave around the trigger result. The same segment of the wave is
    # also used to update the buffer.
    result_mean: Optional[float] = None


# CorrelationTrigger


class CorrelationTriggerConfig(
    MainTriggerConfig,
    always_dump="""
    mean_responsiveness
    slope_width
    pitch_tracking
    """
    # deprecated
    " buffer_falloff ",
):
    ## General parameters ##

    # Whether to subtract the mean from each frame
    mean_responsiveness: float = 1.0

    # Sign amplification
    sign_strength: float = 0

    # Maximum distance to move, in terms of trigger_ms/trigger_samp (not in GUI)
    trigger_diameter: float = 0.5

    # Maximum distance to move, in terms of estimated wave period (not in GUI)
    trigger_radius_periods: Optional[float] = 1.5

    ## Period/frequency estimation (not in GUI) ##

    # Minimum pitch change to recalculate _prev_slope_finder (not in GUI)
    recalc_semitones: float = 1.0

    # (not in GUI)
    max_freq: float = with_units("Hz", default=4000)

    ## Edge triggering ##

    # Competes against buffer_strength.
    edge_strength: float

    slope_width: float = with_units("period", default=0.25)

    ## Correlation ##

    # Competes against edge_strength.
    buffer_strength: float = 1

    # How much to update the buffer *after* each frame.
    responsiveness: float

    # Standard deviation of buffer window, in terms of estimated wave period. Used by
    # _update_buffer() *after* each frame. (not in GUI)
    buffer_falloff: float = 0.5

    # Below a specific correlation quality, discard the buffer entirely.
    reset_below: float = 0

    # Whether to compute a spectrum to rescale correlation buffer in response to pitch
    # changes. (GUI only has a checkbox)
    pitch_tracking: Optional[SpectrumConfig] = None

    # region Legacy Aliases
    trigger_strength = Alias("edge_strength")
    falloff_width = Alias("buffer_falloff")
    data_falloff = Alias("trigger_radius_periods")
    # endregion

    def __attrs_post_init__(self) -> None:
        MainTriggerConfig.__attrs_post_init__(self)

        # Don't validate slope_width.

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
    Correlation-based trigger. it's complicated.

    The data read from the input _wave (length A+B+_trigger_diameter) is longer than
    the correlation buffer (length A+B), to ensure that when sliding the buffer
    across the wave, no edge effects are encountered. This is inspired by
    overlap-save convolution, but adapted to cross-correlation.
    """

    cfg: CorrelationTriggerConfig

    A: int
    """(const) Number of subsamples on buffer/step's left (negative) side."""
    B: int
    """(const) Number of subsamples on buffer/step's right (positive) side. Equals A.
    buffer_nsamp = A+B
    """

    _trigger_diameter: int
    """(const) Distance (in subsamples) between the smallest and largest positions we can
    output on a given frame. Approximately equal to 0.5 * (A+B). """

    _corr_buffer: "npt.NDArray[f32]"
    """(mutable) [A+B] Amplitude"""

    _prev_mean: float
    _prev_period: Optional[int]
    _prev_slope_finder: "Optional[npt.NDArray[f32]]"
    """(mutable) [A+B] Amplitude"""
    _prev_window: "npt.NDArray[f32]"
    """(mutable) [A+B] Amplitude"""

    _prev_trigger: int
    _frames_since_spectrum: int
    _spectrum_calc: DummySpectrum

    @property
    def scfg(self) -> Optional[SpectrumConfig]:
        return self.cfg.pitch_tracking

    def calc_buffer_std(self, period: float) -> float:
        return period * self.cfg.buffer_falloff

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = self.B = self._tsamp // 2
        kernel_size = self.A + self.B
        self._trigger_diameter = int(kernel_size * self.cfg.trigger_diameter)

        # (mutable) Correlated with data (for triggering).
        # Updated with tightly windowed old data at various pitches.
        self._corr_buffer = np.zeros(kernel_size, dtype=f32)

        self._prev_mean = 0.0
        # Will be overwritten on the first frame.
        self._prev_period = None
        self._prev_slope_finder = None
        self._prev_window = np.zeros(kernel_size, f32)

        self._prev_trigger = 0

        # (mutable) Log-scaled spectrum
        self._frames_since_spectrum = 0

        if self.scfg:
            self._spectrum_calc = LogFreqSpectrum(
                scfg=self.scfg,
                subsmp_per_s=self.subsmp_per_s,
                dummy_data=self._corr_buffer,
            )
        else:
            self._spectrum_calc = DummySpectrum()

    def _calc_slope_finder(self, period: float) -> np.ndarray:
        """Called whenever period changes substantially.
        Returns a kernel to be correlated with input data to find positive slopes,
        with length A+B."""

        cfg = self.cfg
        kernel_size = self.A + self.B

        # noinspection PyTypeChecker
        slope_width: float = np.clip(cfg.slope_width * period, 1.0, self.A / 3)

        # This is a fudge factor. Adjust it until it feels right.
        slope_strength = cfg.edge_strength * 5
        # slope_width is 1.0 or greater, so this doesn't divide by 0.

        slope_finder = np.empty(kernel_size, dtype=f32)  # type: np.ndarray[f32]
        slope_finder[: self.A] = -slope_strength / 2
        slope_finder[self.A :] = slope_strength / 2

        slope_finder *= windows.gaussian(kernel_size, std=slope_width)
        return slope_finder

    # end setup

    # begin per-frame
    def get_trigger(self, pos: int, cache: "PerFrameCache") -> TriggerResult:
        cfg = self.cfg

        stride = self._stride

        # Convert sizes to full samples (not trigger subsamples) when indexing into
        # _wave.

        # _trigger_diameter is defined as inclusive. The length of find_peak()'s
        # corr variable is (A + _trigger_diameter + B) - (A + B) + 1, or
        # _trigger_diameter + 1. This gives us a possible triggering range of
        # _trigger_diameter inclusive, which is what we want.
        data_nsubsmp = self.A + self._trigger_diameter + self.B

        trigger_begin = max(
            pos - self._smp_per_frame, pos - self._trigger_diameter // 2
        )
        data_begin = trigger_begin - stride * self.A

        # Get subsampled data (1D, downmixed to mono)
        # [data_nsubsmp = A + _trigger_diameter + B] Amplitude
        data = self._wave.get_padded(
            data_begin, data_begin + stride * data_nsubsmp, stride
        )
        assert data.size == data_nsubsmp, (data.size, data_nsubsmp)

        if cfg.sign_strength != 0:
            signs = sign_times_peak(data)
            data += cfg.sign_strength * signs

        # Remove mean from data, if enabled.
        mean = np.add.reduce(data) / data.size
        period_data = data - mean

        if cfg.mean_responsiveness:
            self._prev_mean += cfg.mean_responsiveness * (mean - self._prev_mean)
            if cfg.mean_responsiveness != 1:
                data -= self._prev_mean
            else:
                data = period_data
        cache.smoothed_mean = self._prev_mean

        # Use period to recompute slope finder (if enabled) and restrict trigger
        # diameter.
        period = get_period(period_data, self.subsmp_per_s, cfg.max_freq, self)
        cache.period = period * stride

        semitones = self._is_window_invalid(period)
        # If pitch changed...
        if semitones:
            slope_finder = self._calc_slope_finder(period)

            # If pitch tracking enabled, rescale buffer to match data's pitch.
            if self.scfg and (data != 0).any():
                # Mutates self._buffer.
                self.spectrum_rescale_buffer(data)

            self._prev_period = period
            self._prev_slope_finder = slope_finder
        else:
            slope_finder = cast(np.ndarray, self._prev_slope_finder)

        corr_enabled = bool(cfg.buffer_strength) and bool(cfg.responsiveness)

        # Buffer sizes:
        # data_nsubsmp = A + _trigger_diameter + B
        kernel_size = self.A + self.B
        corr_nsamp = self._trigger_diameter + 1
        assert corr_nsamp == data_nsubsmp - kernel_size + 1

        # Check if buffer still lines up well with data.
        if corr_enabled:
            # array[corr_nsamp] Amplitude
            corr_quality = signal.correlate_valid(data, self._corr_buffer)
            assert len(corr_quality) == corr_nsamp

            if cfg.reset_below > 0:
                peak_idx = np.argmax(corr_quality)
                peak_quality = corr_quality[peak_idx]

                data_slice = data[peak_idx : peak_idx + kernel_size]

                # Keep in sync with _update_buffer()!
                windowed_slice = data_slice - mean
                normalize_buffer(windowed_slice)
                windowed_slice *= self._prev_window
                self_quality = np.add.reduce(data_slice * windowed_slice)

                relative_quality = peak_quality / (self_quality + 0.001)
                should_reset = relative_quality < cfg.reset_below
                if should_reset:
                    corr_quality[:] = 0
                    self._corr_buffer[:] = 0
                    corr_enabled = False
        else:
            corr_quality = np.zeros(corr_nsamp, f32)

        # array[A+B] Amplitude
        corr_kernel = slope_finder
        del slope_finder
        if corr_enabled:
            corr_kernel += self._corr_buffer * cfg.buffer_strength

        # `corr[x]` = correlation of kernel placed at position `x` in data.
        # `corr_kernel` is not allowed to move past the boundaries of `data`.
        corr = signal.correlate_valid(data, corr_kernel)
        assert len(corr) == corr_nsamp

        peaks = corr_quality
        del corr_quality
        peaks *= cfg.buffer_strength

        if cfg.edge_strength:
            # I want a half-open cumsum, where edge_score[0] = 0, [1] = data[A], [2] =
            # data[A] + data[A+1], etc. But cumsum is inclusive, which causes tests to
            # fail. So subtract 1 from the input range.
            edge_score = np.cumsum(data[self.A - 1 : len(data) - self.B])

            # The optimal edge alignment is the *minimum* cumulative sum, so invert
            # the cumsum so the minimum amplitude maps to the highest score.
            edge_score *= -cfg.edge_strength
            peaks += edge_score

        # Don't pick peaks more than `period * trigger_radius_periods` away from the
        # center.
        if cfg.trigger_radius_periods and period != UNKNOWN_PERIOD:
            trigger_radius = round(period * cfg.trigger_radius_periods)
        else:
            trigger_radius = None

        def find_peak(
            corr: np.ndarray, peaks: np.ndarray, radius: Optional[int]
        ) -> int:
            """If radius is set, the returned offset is limited to ±radius from the
            center of correlation.
            """
            assert len(corr) == len(peaks) == corr_nsamp
            # returns double, not single/f32
            begin_offset = 0

            if radius is not None:
                Ncorr = len(corr)
                mid = Ncorr // 2

                left = max(mid - radius, 0)
                right = min(mid + radius + 1, Ncorr)

                corr = corr[left:right]
                peaks = peaks[left:right]
                begin_offset = left

            min_corr = np.min(corr)

            # Only permit local maxima. This fixes triggering errors where the edge
            # of the allowed range has higher correlation than edges in-bounds,
            # but isn't a rising edge itself (a local maximum of alignment).
            corr[:-1][peaks[:-1] < peaks[1:]] = min_corr
            corr[1:][peaks[1:] < peaks[:-1]] = min_corr
            corr[0] = corr[-1] = min_corr

            # Find optimal offset
            peak_offset = np.argmax(corr) + begin_offset  # type: int
            return peak_offset

        # Find correlation peak.
        peak_offset = find_peak(corr, peaks, trigger_radius)
        trigger = trigger_begin + stride * (peak_offset)

        del data

        if self.post:
            # Apply post trigger (before updating correlation buffer)
            trigger = self.post.get_trigger(trigger, cache)

        # Avoid time traveling backwards.
        self._prev_trigger = trigger = max(trigger, self._prev_trigger)

        # Update correlation buffer (distinct from visible area)
        aligned = self._wave.get_around(trigger, kernel_size, stride)
        cache.result_mean = np.add.reduce(aligned) / kernel_size
        self._update_buffer(aligned, cache)

        self._frames_since_spectrum += 1

        # period: subsmp/cyc
        freq_estimate = self.subsmp_per_s / period if period else None
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
        Ntrigger = self._corr_buffer.size
        if self._frames_since_spectrum < self.scfg.min_frames_between_recompute:
            return
        self._frames_since_spectrum = 0

        calc_spectrum = self._spectrum_calc.calc_spectrum

        # Compute log-frequency spectrum of `data`.
        spectrum = calc_spectrum(data)
        normalize_buffer(spectrum)
        assert not np.any(np.isnan(spectrum))

        # Compute log-frequency spectrum of `self._buffer`.
        prev_spectrum = calc_spectrum(self._corr_buffer)
        # Don't normalize self._spectrum. It was already normalized when being assigned.

        # Rescale `self._buffer` until its pitch matches `data`.
        resample_notes = correlate_spectrum(
            spectrum, prev_spectrum, scfg.max_notes_to_resample
        ).peak
        if resample_notes != 0:
            # If we want to double pitch, we must divide data length by 2.
            new_len = iround(Ntrigger / 2 ** (resample_notes / scfg.notes_per_octave))

            def rescale_mut(corr_kernel_mut):
                buf = np.interp(
                    np.linspace(0, 1, new_len),
                    np.linspace(0, 1, Ntrigger),
                    corr_kernel_mut,
                )
                # assert len(buf) == new_len
                buf = midpad(buf, Ntrigger)
                corr_kernel_mut[:] = buf

            # Copy+resample self._buffer.
            rescale_mut(self._corr_buffer)

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
        assert cache.result_mean is not None
        assert cache.period is not None
        responsiveness = self.cfg.responsiveness

        if self.cfg.buffer_strength and responsiveness:
            # N should equal self.A + self.B.
            N = len(data)
            if N != self._corr_buffer.size:
                raise ValueError(
                    f"invalid data length {len(data)} does not match "
                    f"CorrelationTrigger {self._corr_buffer.size}"
                )

            # New waveform
            data -= cache.result_mean
            normalize_buffer(data)
            window = gaussian_or_zero(
                N, std=self.calc_buffer_std(cache.period / self._stride)
            )
            data *= window
            self._prev_window = window

            # Old buffer
            normalize_buffer(self._corr_buffer)
            self._corr_buffer = lerp(self._corr_buffer, data, responsiveness)


@attr.dataclass
class SpectrumResult:
    peak: int
    corr: np.ndarray


def correlate_spectrum(
    data: np.ndarray, prev_buffer: np.ndarray, radius: Optional[int]
) -> SpectrumResult:
    """
    This is confusing.

    If data index < optimal, data will be too far to the right,
    and we need to `index += positive`.
    - The peak will appear near the right of `data`.

    Either we must slide prev_buffer to the right,
    or we must slide data to the left (by sliding index to the right):
    - correlate(data, prev_buffer)
    - trigger = index + peak_offset

    In correlate_spectrum(), I used to use parabolic() on the return value,
    but unfortunately it was unreliable and caused Plok Beach bass to jitter,
    so I turned it off (resulting in the same code as correlate_data).
    """
    N = len(data)
    corr = signal.correlate(data, prev_buffer)  # returns double, not single/f32
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
    return SpectrumResult(peak_offset, corr)


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

        wave = self._wave.with_offset(-cache.smoothed_mean)

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
