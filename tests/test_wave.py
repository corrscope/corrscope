import warnings
from typing import Sequence

import numpy as np
import pytest
from delayed_assert import expect, assert_expectations
from numpy.testing import assert_allclose

from corrscope.config import CorrError
from corrscope.utils.scipy.wavfile import WavFileWarning
from corrscope.wave import Wave, Flatten, calc_flatten_matrix

parametrize = pytest.mark.parametrize

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

        wave = Wave(prefix + wave_path)
        data = wave[:]

        # Audacity dithers <=16-bit WAV files upon export, creating a few bits of noise.
        # As a result, amin(data) <= 0.
        assert -0.01 < np.amin(data) <= 0
        assert 0.99 < np.amax(data) <= 1

        # check for FutureWarning (raised when determining wavfile type)
        warns = [o for o in w if issubclass(o.category, FutureWarning)]
        assert not [str(w) for w in warns]


def test_incomplete_wav_chunk():
    """Tests that WAV files exported from foobar2000 can be loaded properly,
    without a `ValueError: Incomplete wav chunk.` exception being raised."""

    wave = Wave(prefix + "incomplete-wav-chunk.wav")
    data = wave[:]


# Stereo tests


def arr(*args):
    return np.array(args)


def test_calc_flatten_matrix():
    nchan = 3

    # Test Stereo
    np.testing.assert_equal(calc_flatten_matrix(Flatten.Stereo, nchan), np.eye(nchan))

    # Test SumAvg on various channel counts
    np.testing.assert_equal(calc_flatten_matrix(Flatten.SumAvg, 1), [1])
    np.testing.assert_equal(calc_flatten_matrix(Flatten.SumAvg, 2), [0.5, 0.5])
    np.testing.assert_equal(calc_flatten_matrix(Flatten.SumAvg, 4), [0.25] * 4)

    # Test DiffAvg on various channel counts
    # (Wave will use Mono instead of DiffAvg, on mono audio signals.
    # But ensure it doesn't crash anyway.)
    np.testing.assert_equal(calc_flatten_matrix(Flatten.DiffAvg, 1), [0.5])
    np.testing.assert_equal(calc_flatten_matrix(Flatten.DiffAvg, 2), [0.5, -0.5])
    np.testing.assert_equal(calc_flatten_matrix(Flatten.DiffAvg, 4), [0.5, -0.5, 0, 0])

    # Test Mono
    np.testing.assert_equal(calc_flatten_matrix(Flatten.Mono, 1), [1])

    # Test custom strings and delimiters
    out = arr(1, 2, 1)
    nchan = 3
    np.testing.assert_equal(calc_flatten_matrix(",1,2,1,", nchan), out / sum(out))
    np.testing.assert_equal(calc_flatten_matrix(" 1, 2, 1 ", nchan), out / sum(out))
    np.testing.assert_equal(calc_flatten_matrix("1 2 1", nchan), out / sum(out))

    # Test negative values
    nchan = 2
    np.testing.assert_equal(calc_flatten_matrix("1, -1", nchan), arr(1, -1) / 2)
    np.testing.assert_equal(calc_flatten_matrix("-1, 1", nchan), arr(-1, 1) / 2)
    np.testing.assert_equal(calc_flatten_matrix("-1, -1", nchan), arr(-1, -1) / 2)

    # Test invalid inputs
    with pytest.raises(CorrError):
        calc_flatten_matrix("", 0)

    with pytest.raises(CorrError):
        calc_flatten_matrix("1 -1 uwu", 3)

    with pytest.raises(CorrError):
        calc_flatten_matrix("0 0", 2)


def test_stereo_merge():
    """Test indexing Wave by slices *or* ints. Flatten using default SumAvg mode."""

    # Contains a full-scale sine wave in left channel, and silence in right.
    # λ=100, nsamp=2000
    wave = Wave(prefix + "stereo-sine-left-2000.wav")
    period = 100
    nsamp = 2000

    # [-1, 1) from [-32768..32768)
    int16_step = (1 - -1) / (2**16)
    assert int16_step == 2**-15

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


AllFlattens = [*Flatten.__members__.values(), "1 1", "1 0", "1 -1"]


@pytest.mark.parametrize("flatten", AllFlattens)
@pytest.mark.parametrize("return_channels", [False, True])
@pytest.mark.parametrize(
    "path,nchan,peaks",
    [
        ("tests/sine440.wav", 1, [0.5]),
        ("tests/stereo in-phase.wav", 2, [1, 1]),
        ("tests/wav-formats/stereo-sine-left-2000.wav", 2, [1, 0]),
    ],
)
def test_stereo_flatten_modes(
    flatten: Flatten,
    return_channels: bool,
    path: str,
    nchan: int,
    peaks: Sequence[float],
):
    """Ensures all Flatten modes are handled properly
    for stereo and mono signals."""

    # return_channels=False <-> triggering.
    # flatten=stereo -> rendering.
    # These conditions do not currently coexist.
    # if not return_channels and flatten == Flatten.Stereo:
    #     return

    assert nchan == len(peaks)
    wave = Wave(path)

    if flatten is Flatten.Mono:
        with pytest.raises(CorrError):
            wave.with_flatten(flatten, return_channels)
        return
    else:
        wave = wave.with_flatten(flatten, return_channels)

    nsamp = wave.nsamp
    data = wave[:]

    # wave.data == 2-D array of shape (nsamp, nchan)
    if flatten == Flatten.Stereo:
        assert data.shape == (nsamp, nchan)
        for chan_data, peak in zip(data.T, peaks):
            assert_full_scale(chan_data, peak)
    else:
        if return_channels:
            assert data.shape == (nsamp, 1)
        else:
            assert data.shape == (nsamp,)

        # If DiffAvg and in-phase, L-R=0.
        if flatten == Flatten.DiffAvg:
            if len(peaks) >= 2 and peaks[0] == peaks[1]:
                np.testing.assert_equal(data, 0)
            else:
                pass
        # If SumAvg, check average.
        elif flatten == Flatten.SumAvg:
            assert_full_scale(data, np.mean(peaks))
        # Don't test custom string modes for now.


def assert_full_scale(data, peak):
    peak = abs(peak)
    assert np.amax(data) == pytest.approx(peak, rel=0.01)
    assert np.amin(data) == pytest.approx(-peak, rel=0.01)


def test_stereo_mmap():
    wave = Wave(prefix + "stereo-sine-left-2000.wav")
    assert isinstance(wave.data, np.memmap)


# Miscellaneous tests


def test_wave_subsampling():
    wave = Wave("tests/sine440.wav")
    # period = 48000 / 440 = 109.(09)*

    wave.get_around(1000, return_nsamp=501, stride=4)
    # len([:region_len:subsampling]) == ceil(region_len / subsampling)
    # If region_len % subsampling != 0, len() != region_len // subsampling.

    stride = 4
    region = 100  # diameter = region * stride
    for i in [-1000, 50000]:
        data = wave.get_around(i, region, stride)
        assert (data == 0).all()


is_odd = parametrize("is_odd", [False, True])


@is_odd
def test_wave_subsampling_off_by_1(is_odd: bool):
    """When calling wave.get_around(x, N), ensure that result[N//2] == wave[x]."""
    wave = Wave(prefix + "s16-impulse1000.wav")
    for stride in range(1, 5 + 1):
        N = 500 + is_odd
        halfN = N // 2
        result = wave.get_around(1000, N, stride)

        assert result[halfN] > 0.5, stride

        result[halfN] = 0
        np.testing.assert_almost_equal(result, 0, decimal=3)
        # looks like s16-impulse1000.wav isn't 0-filled after all


def test_stereo_doesnt_overflow():
    """Ensure loud stereo tracks do not overflow."""
    wave = Wave("tests/stereo in-phase.wav")

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
        wave = Wave("tests/header larger than filesize.wav")
        assert wave
