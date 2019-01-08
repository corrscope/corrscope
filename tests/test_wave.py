import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest
from delayed_assert import expect, assert_expectations

from corrscope.utils.scipy_wavfile import WavFileWarning
from corrscope.wave import Wave

prefix = "tests/wav-formats/"
wave_paths = [
    # 2000 samples, with a full-scale peak at data[1000].
    "u8-impulse1000.wav",
    "s16-impulse1000.wav",
    "s32-impulse1000.wav",
    "f32-impulse1000.wav",
    "f64-impulse1000.wav",
]


@pytest.mark.parametrize("wave_path", wave_paths)
def test_wave(wave_path):
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        wave = Wave(None, prefix + wave_path)
        data = wave[:]

        # Audacity dithers <=16-bit WAV files upon export, creating a few bits of noise.
        # As a result, amin(data) <= 0.
        assert -0.01 < np.amin(data) <= 0
        assert 0.99 < np.amax(data) <= 1

        # check for FutureWarning (raised when determining wavfile type)
        warns = [o for o in w if issubclass(o.category, FutureWarning)]
        assert not [str(w) for w in warns]


def test_stereo_merge():
    """ Ensure stereo channels are combined properly, when indexing by slices
    *or* ints. """

    # Contains a full-scale sine wave in left channel, and silence in right.
    # λ=100, nsamp=2000
    wave = Wave(None, prefix + "stereo-sine-left-2000.wav")
    period = 100
    nsamp = 2000

    # [-1, 1) from [-32768..32768)
    int16_step = (1 - -1) / (2 ** 16)
    assert int16_step == 2 ** -15

    # Check wave indexing dimensions.
    assert wave[0].shape == ()
    assert wave[:].shape == (nsamp,)

    # Check stereo merging.
    assert_allclose(wave[0], 0)
    assert_allclose(wave[period], 0)
    assert_allclose(wave[period // 4], 0.5, atol=int16_step)

    def check_bound(obj):
        amax = np.amax(obj)
        assert amax.shape == ()

        assert_allclose(amax, 0.5, atol=int16_step)
        assert_allclose(np.amin(obj), -0.5, atol=int16_step)

    check_bound(wave[:])


def test_stereo_mmap():
    wave = Wave(None, prefix + "stereo-sine-left-2000.wav")
    assert isinstance(wave.data, np.memmap)


def test_wave_subsampling():
    wave = Wave(None, "tests/sine440.wav")
    # period = 48000 / 440 = 109.(09)*

    wave.get_around(1000, return_nsamp=501, stride=4)
    # len([:region_len:subsampling]) == ceil(region_len / subsampling)
    # If region_len % subsampling != 0, len() != region_len // subsampling.

    stride = 4
    region = 100  # diameter = region * stride
    for i in [-1000, 50000]:
        data = wave.get_around(i, region, stride)
        assert (data == 0).all()


def test_stereo_doesnt_overflow():
    """ Ensure loud stereo tracks do not overflow. """
    wave = Wave(None, "tests/stereo in-phase.wav")

    samp = 100
    stride = 1
    data = wave.get_around(wave.nsamp // 2, samp, stride)
    expect(np.amax(data) > 0.99)
    expect(np.amin(data) < -0.99)

    # In the absence of overflow, sine waves have no large jumps.
    # In the presence of overflow, stereo sum will jump between INT_MAX and INT_MIN.
    # np.mean and rescaling converts to 0.499... and -0.5, which is nearly 1.
    expect(np.amax(np.abs(np.diff(data))) < 0.5)

    assert_expectations()


def test_header_larger_than_filesize():
    """According to Zeinok, VortexTracker 2.5 produces slightly corrupted WAV files
    whose RIFF header metadata indicates a filesize larger than the actual filesize.

    Most programs read the audio chunk fine.
    Scipy normally rejects such files, raises ValueError("Unexpected end of file.")
    My version instead accepts such files (but warns WavFileWarning).
    """
    with pytest.warns(WavFileWarning):
        wave = Wave(None, "tests/header larger than filesize.wav")
        assert wave
