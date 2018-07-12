from typing import List, Union

import numpy as np
from wavetable.gauss import nyquist_exclusive, nyquist_inclusive


# I don't know why, but "inclusive" makes my test cases work.
NYQUIST = nyquist_inclusive


def _zoh_transfer(nsamp):
    nyquist = NYQUIST(nsamp)
    return np.sinc(np.arange(nyquist) / nsamp)


def _realify(a: np.ndarray):
    return np.copysign(np.abs(a), a.real)


InputArray = Union[np.ndarray, List[complex]]


def rfft_norm(signal: InputArray, *args, **kwargs) -> np.ndarray:
    """ Computes "normalized" FFT of signal. """
    return np.fft.rfft(signal, *args, **kwargs) / len(signal)


def irfft_norm(spectrum: InputArray, nsamp=None, *args, **kwargs) -> np.ndarray:
    """ Computes "normalized" signal of spectrum. """
    signal = np.fft.irfft(spectrum, nsamp, *args, **kwargs)
    return signal * len(signal)


def rfft_zoh(signal: InputArray):
    """ Computes "normalized" FFT of signal, with simulated ZOH frequency response. """
    nsamp = len(signal)
    spectrum = rfft_norm(signal)

    # Muffle everything ~~but Nyquist~~, like real hardware.
    # Nyquist is already real.
    nyquist = NYQUIST(nsamp)
    spectrum[:nyquist] *= _zoh_transfer(nsamp)

    return spectrum


def _zero_pad(spectrum: InputArray, harmonic):
    """ Do not use, concatenating a waveform multiple times works just as well. """

    # https://stackoverflow.com/a/5347492/2683842
    nyquist = len(spectrum) - 1
    padded = np.zeros(nyquist * harmonic + 1, dtype=complex)
    padded[::harmonic] = spectrum
    return padded


def irfft_zoh(spectrum: InputArray, nsamp=None):
    """ Computes "normalized" signal of spectrum, counteracting ZOH frequency response. """
    compute_nsamp = (nsamp is None)

    if nsamp is None:
        nsamp = 2 * (len(spectrum) - 1)
    nin = nsamp // 2 + 1

    if compute_nsamp:
        assert nin == len(spectrum)

    spectrum = np.copy(spectrum[:nin])

    # Treble-boost everything ~~but Nyquist~~.
    # Make Nyquist purely real.
    nyquist = NYQUIST(nsamp)
    real = nyquist_exclusive(nsamp)

    spectrum[:nyquist] /= _zoh_transfer(nsamp)
    spectrum[real:] = _realify(spectrum[real:])

    return irfft_norm(spectrum, nsamp)


def irfft_old(spectrum: InputArray, nsamp=None):
    """
    Calculate the inverse Fourier transform of $spectrum, optionally
    truncating to $nsamp entries.

    if nsamp is None, calculate nsamp = 2*(len-1).
        nin = (nsamp//2) + 1.

    if nsamp is even, realify the last coeff to preserve Nyquist energy.
    """
    compute_nsamp = (nsamp is None)

    if nsamp is None:
        nsamp = 2 * (len(spectrum) - 1)
    nin = nsamp // 2 + 1

    if compute_nsamp:
        assert nin == len(spectrum)

    spectrum = spectrum[:nin].copy()

    if nsamp % 2 == 0:
        last = spectrum[-1]
        spectrum[-1] = np.copysign(abs(last), last.real)

    return irfft_norm(spectrum, nsamp)
