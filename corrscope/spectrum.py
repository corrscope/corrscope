from typing import TYPE_CHECKING, NewType, Sequence, List, Any

import numpy as np

import corrscope.utils.scipy.signal as signal
from corrscope.config import KeywordAttrs
from corrscope.wave_common import f32


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
    max_octaves_to_resample: float = 1.0

    @property
    def max_notes_to_resample(self) -> int:
        return round(self.notes_per_octave * self.max_octaves_to_resample)

    # Time-domain history parameters
    min_frames_between_recompute: int = 1


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

    def __init__(
        self, scfg: SpectrumConfig, subsmp_per_s: float, dummy_data: np.ndarray
    ):
        self.scfg = scfg

        n_fftindex = FFTIndex(signal.next_fast_len(len(dummy_data)))

        # Increase n_fftindex until every note has nonzero width.
        while True:
            # Compute parameters
            self.min_hz = scfg.min_hz
            self.max_hz = self.min_hz * 2**scfg.octaves
            n_fencepost = scfg.notes_per_octave * scfg.octaves + 1

            note_fenceposts_hz = np.geomspace(
                self.min_hz, self.max_hz, n_fencepost, dtype=f32
            )

            # Convert fenceposts to FFTIndex
            fft_from_hertz = n_fftindex / subsmp_per_s
            note_fenceposts: FFTIndexArray = (
                fft_from_hertz * note_fenceposts_hz
            ).astype(np.int32)
            note_widths = np.diff(note_fenceposts)

            if np.any(note_widths == 0):
                n_fftindex = FFTIndex(
                    signal.next_fast_len(n_fftindex + n_fftindex // 5 + 1)
                )
                continue
            else:
                break

        self.n_fftindex = n_fftindex  # Passed to rfft() to automatically zero-pad data.
        self.note_fenceposts = note_fenceposts
        self.n_fencepost = len(note_fenceposts)

    def calc_spectrum(self, data: np.ndarray) -> np.ndarray:
        """Unfortunately converting to f32 (single) adds too much overhead.

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
    """Based off np.split(), but faster.
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
