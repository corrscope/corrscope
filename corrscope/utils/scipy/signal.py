from bisect import bisect_left

import numpy as np

def correlate_valid(buffer: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Based on scipy.correlate. buffer must be longer (or equal) to kernel. Returns an
    array of length (buffer - kernel + 1) without edge effects, much like mode="valid".
    """
    buffer = np.asarray(buffer)
    kernel = np.asarray(kernel)

    assert buffer.ndim == kernel.ndim == 1
    kernel = _reverse_and_conj(kernel)

    # Taken from scipy fftconvolve()

    kernel_support = len(kernel) - 1
    out_nsamp = len(buffer) - kernel_support

    # fft_nsamp = 1 << (out_nsamp - 1).bit_length()
    fft_nsamp = next_fast_len(len(buffer))
    assert fft_nsamp >= out_nsamp

    # return convolve(in1, _reverse_and_conj(in2), mode, method)
    sp1 = np.fft.rfft(buffer, fft_nsamp)
    # Already reversed above.
    sp2 = np.fft.rfft(kernel, fft_nsamp)

    corr = np.fft.irfft(sp1 * sp2, fft_nsamp)
    # Slice the returned data to the valid region, for complex math reasons.
    ret = corr[kernel_support : kernel_support + out_nsamp].copy()
    return ret



def correlate(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """
    Based on scipy.correlate.
    Assumed: mode='full', method='fft'
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    assert in1.ndim == in2.ndim == 1
    in2 = _reverse_and_conj(in2)

    # Taken from scipy fftconvolve()

    out_nsamp = len(in1) + len(in2) - 1

    # fft_nsamp = 1 << (out_nsamp - 1).bit_length()
    fft_nsamp = next_fast_len(out_nsamp)
    assert fft_nsamp >= out_nsamp

    # return convolve(in1, _reverse_and_conj(in2), mode, method)
    sp1 = np.fft.rfft(in1, fft_nsamp)
    sp2 = np.fft.rfft(in2, fft_nsamp)
    ret = np.fft.irfft(sp1 * sp2, fft_nsamp)[:out_nsamp].copy()

    return ret


def _reverse_and_conj(x: np.ndarray) -> np.ndarray:
    return x[::-1].conj()


def next_fast_len(target: int) -> int:
    """
    Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    On a particular machine, an FFT of prime length takes 133 ms:

    >>> from scipy import fftpack
    >>> min_len = 10007  # prime length is worst case for speed
    >>> a = np.random.randn(min_len)
    >>> b = fftpack.fft(a)

    Zero-padding to the next 5-smooth length reduces computation time to
    211 us, a speedup of 630 times:

    >>> fftpack.helper.next_fast_len(min_len)
    10125
    >>> b = fftpack.fft(a, 10125)

    Rounding up to the next power of 2 is not optimal, taking 367 us to
    compute, 1.7 times as long as the 5-smooth size:

    >>> b = fftpack.fft(a, 16384)

    """
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
            135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
            256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
            480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
            750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
            1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
            1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
            2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
            3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
            3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
            5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
            6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
            8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)

    target = int(target)

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
