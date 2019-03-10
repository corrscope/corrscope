from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Type,
    Tuple,
    Optional,
    ClassVar,
    Callable,
    Union,
    NewType,
    Sequence,
    List,
    Any,
)

import attr
import numpy as np

import corrscope.utils.scipy.signal as signal
import corrscope.utils.scipy.windows as windows
from corrscope.config import KeywordAttrs, CorrError, Alias
from corrscope.util import find, obj_name
from corrscope.utils.windows import midpad, leftpad
from corrscope.wave import FLOAT

if TYPE_CHECKING:
    from corrscope.wave import Wave


# Abstract classes


class _TriggerConfig:
    cls: ClassVar[Type["_Trigger"]]

    def __call__(self, wave: "Wave", tsamp: int, stride: int, fps: float) -> "_Trigger":
        return self.cls(wave, cfg=self, tsamp=tsamp, stride=stride, fps=fps)


class MainTriggerConfig(_TriggerConfig, KeywordAttrs, always_dump="edge_direction"):
    # Must be 1 or -1.
    # MainTrigger.__init__() multiplies `wave.amplification *= edge_direction`.
    # get_trigger() should ignore `edge_direction` and look for rising edges.
    edge_direction: int = 1

    # Optional trigger for postprocessing
    post_trigger: Optional["PostTriggerConfig"] = None
    post_radius: Optional[int] = 3

    @property
    def post_nsamp(self) -> Optional[int]:
        if self.post_radius is not None:
            return 2 * self.post_radius + 1
        else:
            return None

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
    pass


def register_trigger(
    config_t: Type[_TriggerConfig]
) -> "Callable[[Type[_Trigger]], Type[_Trigger]]":  # my god mypy-strict sucks
    """ @register_trigger(FooTriggerConfig)
    def FooTrigger(): ...
    """

    def inner(trigger_t: Type[_Trigger]):
        config_t.cls = trigger_t
        return trigger_t

    return inner


class _Trigger(ABC):
    def __init__(
        self, wave: "Wave", cfg: _TriggerConfig, tsamp: int, stride: int, fps: float
    ):
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

    def time2tsamp(self, time: float) -> int:
        return round(time * self._wave.smp_s / self._stride)

    @abstractmethod
    def get_trigger(self, index: int, cache: "PerFrameCache") -> int:
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


class MainTrigger(_Trigger, ABC):
    cfg: MainTriggerConfig
    post: Optional["PostTrigger"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wave.amplification *= self.cfg.edge_direction

        cfg = self.cfg
        if cfg.post_trigger:
            # Create a post-processing trigger, with narrow nsamp and stride=1.
            # This improves speed and precision.
            self.post = cfg.post_trigger(self._wave, cfg.post_nsamp, 1, self._fps)
        else:
            self.post = None

    def do_not_inherit__Trigger_directly(self):
        pass


class PostTrigger(_Trigger, ABC):
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


@attr.dataclass
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


class SpectrumConfig(KeywordAttrs):
    """
    # Rationale:
    If no basal frequency note-bands are to be truncated,
    the spectrum must have freq resolution
        `min_hz * (2 ** 1/notes_per_octave - 1)`.

    At 20hz, 10 octaves, 12 notes/octave, this is 1.19Hz fft freqs.
    Our highest band must be
        `min_hz * 2**octaves`,
    leading to nearly 20K freqs, which produces an somewhat slow FFT.

    So increase min_hz and decrease octaves and notes_per_octave.
    --------
    Using a Constant-Q transform may eliminate performance concerns?
    """

    # Spectrum X density
    min_hz: float = 20
    octaves: int = 8
    notes_per_octave: int = 6

    # Spectrum Y power
    exponent: float = 1
    divide_by_freq: bool = True

    # Spectral alignment and resampling
    pitch_estimate_boost: float = 1.2
    max_octaves_to_resample: float = 1.0

    @property
    def max_notes_to_resample(self) -> int:
        return round(self.notes_per_octave * self.max_octaves_to_resample)

    # Time-domain history parameters
    min_frames_between_recompute: int = 6
    frames_to_lookbehind: int = 2


class DummySpectrum:
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def calc_spectrum(self, data: np.ndarray) -> np.ndarray:
        return np.array([])


# Indices are linearly spaced in FFT. Notes are exponentially spaced.
# FFT is grouped into notes.
FFTIndex = NewType("FFTIndex", int)
# Very hacky and weird. Maybe it's not worth getting mypy to pass.
if TYPE_CHECKING:
    FFTIndexArray = Any  # mypy
else:
    FFTIndexArray = "np.ndarray[FFTIndex]"  # pycharm


class LogFreqSpectrum(DummySpectrum):
    """
    Invariants:
    - len(note_fenceposts) == n_fencepost

    - rfft()[ : note_fenceposts[0]] is NOT used.
    - rfft()[note_fenceposts[-1] : ] is NOT used.
    - rfft()[note_fenceposts[0] : note_fenceposts[1]] becomes a note.
    """

    n_fftindex: FFTIndex  # Determines frequency resolution, not range.
    note_fenceposts: FFTIndexArray
    n_fencepost: int

    def __init__(self, scfg: SpectrumConfig, subsmp_s: float, dummy_data: np.ndarray):
        self.scfg = scfg

        n_fftindex: FFTIndex = signal.next_fast_len(len(dummy_data))

        # Increase n_fftindex until every note has nonzero width.
        while True:
            # Compute parameters
            self.min_hz = scfg.min_hz
            self.max_hz = self.min_hz * 2 ** scfg.octaves
            n_fencepost = scfg.notes_per_octave * scfg.octaves + 1

            note_fenceposts_hz = np.geomspace(
                self.min_hz, self.max_hz, n_fencepost, dtype=FLOAT
            )

            # Convert fenceposts to FFTIndex
            fft_from_hertz = n_fftindex / subsmp_s
            note_fenceposts: FFTIndexArray = (
                fft_from_hertz * note_fenceposts_hz
            ).astype(np.int32)
            note_widths = np.diff(note_fenceposts)

            if np.any(note_widths == 0):
                n_fftindex = signal.next_fast_len(n_fftindex + n_fftindex // 5 + 1)
                continue
            else:
                break

        self.n_fftindex = n_fftindex  # Passed to rfft() to automatically zero-pad data.
        self.note_fenceposts = note_fenceposts
        self.n_fencepost = len(note_fenceposts)

    def calc_spectrum(self, data: np.ndarray) -> np.ndarray:
        """ Unfortunately converting to FLOAT (single) adds too much overhead.

        Input: Time-domain signal to be analyzed.
        Output: Frequency-domain spectrum with exponentially-spaced notes.
        - ret[note] = nonnegative float.
        """
        scfg = self.scfg

        # Compute FFT spectrum[freq]
        spectrum = np.fft.rfft(data, self.n_fftindex)
        spectrum = abs(spectrum)
        if scfg.exponent != 1:
            spectrum **= scfg.exponent

        # Compute energy of each note
        # spectrum_per_note[note] = np.ndarray[float]
        spectrum_per_note: List[np.ndarray] = split(spectrum, self.note_fenceposts)

        # energy_per_note[note] = float
        energy_per_note: np.ndarray

        # np.add.reduce is much faster than np.sum/mean.
        if scfg.divide_by_freq:
            energy_per_note = np.array(
                [np.add.reduce(region) / len(region) for region in spectrum_per_note]
            )
        else:
            energy_per_note = np.array(
                [np.add.reduce(region) for region in spectrum_per_note]
            )

        assert len(energy_per_note) == self.n_fencepost - 1
        return energy_per_note


def split(data: np.ndarray, fenceposts: Sequence[FFTIndex]) -> List[np.ndarray]:
    """ Based off np.split(), but faster.
    Unlike np.split, does not include data before fenceposts[0] or after fenceposts[-1].
    """
    sub_arys = []
    ndata = len(data)
    for i in range(len(fenceposts) - 1):
        st = fenceposts[i]
        end = fenceposts[i + 1]
        if not st < ndata:
            break
        region = data[st:end]
        sub_arys.append(region)

    return sub_arys


class CircularArray:
    def __init__(self, size: int, *dims: int):
        self.size = size
        self.buf = np.zeros((size, *dims))
        self.index = 0

    def push(self, arr: np.ndarray) -> None:
        if self.size == 0:
            return
        self.buf[self.index] = arr
        self.index = (self.index + 1) % self.size

    def peek(self) -> np.ndarray:
        """Return is borrowed from self.buf.
        Do NOT push to self while borrow is alive."""
        return self.buf[self.index]


class CorrelationTriggerConfig(MainTriggerConfig, always_dump="pitch_tracking"):
    # get_trigger
    edge_strength: float
    trigger_diameter: Optional[float] = 0.5

    trigger_falloff: Tuple[float, float] = (4.0, 1.0)
    recalc_semitones: float = 1.0
    lag_prevention: float = 0.25

    # _update_buffer
    responsiveness: float
    buffer_falloff: float  # Gaussian std = wave_period * buffer_falloff

    # Pitch tracking = compute spectrum.
    pitch_tracking: Optional["SpectrumConfig"] = None

    # region Legacy Aliases
    trigger_strength = Alias("edge_strength")
    falloff_width = Alias("buffer_falloff")
    # endregion

    def __attrs_post_init__(self) -> None:
        MainTriggerConfig.__attrs_post_init__(self)

        self._validate_param("lag_prevention", 0, 1)
        self._validate_param("responsiveness", 0, 1)
        # TODO trigger_falloff >= 0
        self._validate_param("buffer_falloff", 0, np.inf)

    def _validate_param(self, key: str, begin: float, end: float) -> None:
        value = getattr(self, key)
        if not begin <= value <= end:
            raise CorrError(
                f"Invalid {key}={value} (should be within [{begin}, {end}])"
            )


@register_trigger(CorrelationTriggerConfig)
class CorrelationTrigger(MainTrigger):
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
        self._data_taper = self._calc_data_taper()
        assert self._data_taper.dtype == FLOAT

        # (mutable) Correlated with data (for triggering).
        # Updated with tightly windowed old data at various pitches.
        self._buffer = np.zeros(
            self._buffer_nsamp, dtype=FLOAT
        )  # type: np.ndarray[FLOAT]

        # (const) Added to self._buffer. Nonzero if edge triggering is nonzero.
        # Left half is -edge_strength, right half is +edge_strength.
        # ASCII art: --._|‾'--
        self._windowed_step = self._calc_step()
        assert self._windowed_step.dtype == FLOAT

        # Will be overwritten on the first frame.
        self._prev_period: Optional[int] = None
        self._prev_window: Optional[np.ndarray] = None

        # (mutable) Log-scaled spectrum
        self.frames_since_spectrum = 0

        if self.scfg:
            self._spectrum_calc = LogFreqSpectrum(
                scfg=self.scfg,
                subsmp_s=self._wave.smp_s / self._stride,
                dummy_data=self._buffer,
            )
            self._spectrum = self._spectrum_calc.calc_spectrum(self._buffer)
            self.history = CircularArray(
                self.scfg.frames_to_lookbehind, self._buffer_nsamp
            )
        else:
            self._spectrum_calc = DummySpectrum()
            self._spectrum = np.array([0])
            self.history = CircularArray(0, self._buffer_nsamp)

    def _calc_data_taper(self) -> np.ndarray:
        """ Input data window. Zeroes out all data older than 1 frame old.
        See https://github.com/nyanpasu64/corrscope/wiki/Correlation-Trigger
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

        # Right-pad=1 taper to 1 frame long [t-1f, t]
        if width < tsamp_frame:
            taper = np.pad(
                taper, (0, tsamp_frame - width), "constant", constant_values=1
            )
        assert len(taper) == tsamp_frame

        # Left-pad=0 taper to left `halfN` of data_taper [t-halfN, t]
        taper = leftpad(taper, halfN)

        # Generate left half-taper to prevent correlating with 1-frame-old data.
        # Right-pad=1 taper to [t-halfN, t-halfN+N]
        # TODO why not extract a right-pad function?
        data_taper = np.ones(N, dtype=FLOAT)
        data_taper[:halfN] = np.minimum(data_taper[:halfN], taper)

        return data_taper

    def _calc_step(self) -> np.ndarray:
        """ Step function used for approximate edge triggering. """

        # Increasing buffer_falloff (width of history buffer)
        # causes buffer to affect triggering, more than the step function.
        # So we multiply edge_strength (step function height) by buffer_falloff.

        edge_strength = self.cfg.edge_strength * self.cfg.buffer_falloff
        N = self._buffer_nsamp
        halfN = N // 2

        step = np.empty(N, dtype=FLOAT)  # type: np.ndarray[FLOAT]
        step[:halfN] = -edge_strength / 2
        step[halfN:] = edge_strength / 2
        step *= windows.gaussian(N, std=halfN / 3)
        return step

    # end setup

    # begin per-frame
    def get_trigger(self, index: int, cache: "PerFrameCache") -> int:
        N = self._buffer_nsamp
        cfg = self.cfg

        # Get data
        stride = self._stride
        data = self._wave.get_around(index, N, stride)
        cache.mean = np.mean(data)
        data -= cache.mean

        # Window data
        period = get_period(data)
        cache.period = period * stride

        semitones = self._is_window_invalid(period)
        # If pitch changed...
        if semitones:
            diameter, falloff = [round(period * x) for x in cfg.trigger_falloff]
            falloff_window = cosine_flat(N, diameter, falloff)
            window = np.minimum(falloff_window, self._data_taper)

            # If pitch tracking enabled, rescale buffer to match data's pitch.
            if self.scfg and (data != 0).any():
                if isinstance(semitones, float):
                    peak_semitones = semitones
                else:
                    peak_semitones = None
                self.spectrum_rescale_buffer(data, peak_semitones)

            self._prev_period = period
            self._prev_window = window
        else:
            window = self._prev_window

        self.history.push(data)
        data *= window

        prev_buffer: np.ndarray = self._buffer.copy()
        prev_buffer += self._windowed_step

        # Calculate correlation
        if self.cfg.trigger_diameter is not None:
            radius = round(N * self.cfg.trigger_diameter / 2)
        else:
            radius = None

        peak_offset = self.correlate_offset(data, prev_buffer, radius)
        trigger = index + (stride * peak_offset)

        # Apply post trigger (before updating correlation buffer)
        if self.post:
            trigger = self.post.get_trigger(trigger, cache)

        # Update correlation buffer (distinct from visible area)
        aligned = self._wave.get_around(trigger, self._buffer_nsamp, stride)
        self._update_buffer(aligned, cache)
        self.frames_since_spectrum += 1

        return trigger

    def spectrum_rescale_buffer(
        self, data: np.ndarray, peak_semitones: Optional[float]
    ) -> None:
        """Rewrites self._spectrum, and possibly rescales self._buffer."""

        scfg = self.scfg
        N = self._buffer_nsamp

        if self.frames_since_spectrum < self.scfg.min_frames_between_recompute:
            return
        self.frames_since_spectrum = 0

        spectrum = self._spectrum_calc.calc_spectrum(data)
        normalize_buffer(spectrum)

        # Don't normalize self._spectrum. It was already normalized when being assigned.
        prev_spectrum = self._spectrum_calc.calc_spectrum(self.history.peek())

        # rewrite spectrum
        self._spectrum = spectrum

        assert not np.any(np.isnan(spectrum))

        # Find spectral correlation peak,
        # but prioritize "changing pitch by ???".
        if peak_semitones is not None:
            boost_x = int(round(peak_semitones / 12 * scfg.notes_per_octave))
            boost_y: float = scfg.pitch_estimate_boost
        else:
            boost_x = 0
            boost_y = 1.0

        # If we want to double pitch...
        resample_notes = self.correlate_offset(
            spectrum,
            prev_spectrum,
            scfg.max_notes_to_resample,
            boost_x=boost_x,
            boost_y=boost_y,
        )
        if resample_notes != 0:
            # we must divide sampling rate by 2.
            new_len = int(round(N / 2 ** (resample_notes / scfg.notes_per_octave)))

            # Copy+resample self._buffer.
            self._buffer = np.interp(
                np.linspace(0, 1, new_len), np.linspace(0, 1, N), self._buffer
            )
            # assert len(self._buffer) == new_len
            self._buffer = midpad(self._buffer, N)

    @staticmethod
    def correlate_offset(
        data: np.ndarray,
        prev_buffer: np.ndarray,
        radius: Optional[int],
        boost_x: int = 0,
        boost_y: float = 1.0,
    ) -> int:
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

        # Prioritize part of it.
        corr[mid + boost_x : mid + boost_x + 1] *= boost_y

        # argmax(corr) == mid + peak_offset == (data >> peak_offset)
        # peak_offset == argmax(corr) - mid
        peak_offset = np.argmax(corr) - mid  # type: int
        return peak_offset

    def _is_window_invalid(self, period: int) -> Union[bool, float]:
        """ Returns number of semitones,
        if pitch has changed more than `recalc_semitones`. """

        prev = self._prev_period

        if prev is None:
            return True
        elif prev * period == 0:
            return prev != period
        else:
            # If period doubles, semitones are -12.
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
        window = windows.gaussian(N, std=(cache.period / self._stride) * buffer_falloff)
        data *= window

        # Old buffer
        normalize_buffer(self._buffer)
        self._buffer = lerp(self._buffer, data, responsiveness)


# get_trigger()


def calc_step(nsamp: int, peak: float, stdev: float) -> np.ndarray:
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
    corr = signal.correlate(data, data)
    corr = corr[len(corr) // 2 :]

    # Remove the zero-correlation peak
    zero_crossings = np.where(corr < 0)[0]

    if len(zero_crossings) == 0:
        # This can happen given an array of all zeros. Anything else?
        return len(data)

    crossX = zero_crossings[0]
    peakX = crossX + np.argmax(corr[crossX:])
    return int(peakX)


def cosine_flat(n: int, diameter: int, falloff: int) -> np.ndarray:
    cosine = windows.hann(falloff * 2)
    # assert cosine.dtype == FLOAT
    left, right = cosine[:falloff], cosine[falloff:]

    window = np.concatenate([left, np.ones(diameter, dtype=FLOAT), right])

    padded = midpad(window, n)
    # assert padded.dtype == FLOAT
    return padded


# update_buffer()

MIN_AMPLITUDE = 0.01


def normalize_buffer(data: np.ndarray) -> None:
    """
    Rescales `data` in-place.
    """
    peak = np.amax(abs(data))
    data /= max(peak, MIN_AMPLITUDE)


def lerp(x: np.ndarray, y: np.ndarray, a: float) -> Union[np.ndarray, float]:
    return x * (1 - a) + y * a


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

        else:  # self._wave[sample] == 0
            return index + 1

        data = self._wave[index : index + (direction * tsamp) : direction]
        # TODO remove unnecessary complexity, since diameter is probably under 10.
        intercepts = find(data, test)
        try:
            (delta,), value = next(intercepts)
            return index + (delta * direction) + int(value <= 0)

        except StopIteration:  # No zero-intercepts
            return index + (direction * tsamp)

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
    def get_trigger(self, index: int, cache: "PerFrameCache") -> int:
        return index
