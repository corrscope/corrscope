import copy
import enum
import warnings
from enum import auto
from typing import Union, List

import numpy as np

import corrscope.utils.scipy_wavfile as wavfile
from corrscope.config import CorrError, CorrWarning, TypedEnumDump

FLOAT = np.single


@enum.unique
class Flatten(TypedEnumDump, enum.Flag):
    """ How to flatten a stereo signal. (Channels beyond first 2 are ignored.)

    Flatten(0) == Flatten.Stereo == Flatten['Stereo']
    """

    # Keep both channels.
    Stereo = 0

    # Mono
    Mono = auto()  # NOT publicly exposed

    # Take sum or difference.
    SumAvg = auto()
    DiffAvg = auto()

    modes: List["Flatten"]


_rejected_modes = {Flatten.Mono}
Flatten.modes = [f for f in Flatten.__members__.values() if f not in _rejected_modes]


class Wave:
    __slots__ = """
    wave_path
    amplification
    smp_s data _flatten is_mono
    nsamp dtype
    center max_val
    """.split()

    smp_s: int
    data: "np.ndarray"
    """2-D array of shape (nsamp, nchan)"""

    @property
    def flatten(self) -> Flatten:
        """
        If data is stereo:
        - flatten can be Stereo (2D) or Sum/Diff(Avg) (1D).

        If data is mono:
        - flatten can be Stereo (2D) or Mono (1D).
        - If flatten != Stereo, set flatten = Mono.
        """
        return self._flatten

    @flatten.setter
    def flatten(self, flatten: Flatten) -> None:
        """ If self.is_mono, converts all non-Stereo modes to Mono. """
        if flatten not in Flatten.modes:
            raise CorrError(
                f"Wave {self.wave_path} has invalid flatten mode {flatten} "
                f"not in {Flatten.modes}"
            )
        self._flatten = flatten
        if self.is_mono:
            if flatten != Flatten.Stereo:
                self._flatten = Flatten.Mono
        else:
            if self.flatten == Flatten.Mono:
                raise CorrError(
                    f"Cannot initialize stereo file {self.wave_path} with flatten=Mono"
                )

    def __init__(
        self,
        wave_path: str,
        amplification: float = 1.0,
        flatten: Flatten = Flatten.SumAvg,
    ):
        self.wave_path = wave_path
        self.amplification = amplification
        self.smp_s, self.data = wavfile.read(wave_path, mmap=True)

        assert self.data.ndim in [1, 2]
        self.is_mono = self.data.ndim == 1
        self.flatten = flatten

        # Cast self.data to stereo (nsamp, nchan)
        if self.is_mono:
            self.data.shape = (-1, 1)

        self.nsamp, stereo_nchan = self.data.shape
        if stereo_nchan > 2:
            warnings.warn(
                f"File {wave_path} has {stereo_nchan} channels, "
                f"only first 2 will be used",
                CorrWarning,
            )

        dtype = self.data.dtype

        # Calculate scaling factor.
        def is_type(parent: type) -> bool:
            return np.issubdtype(dtype, parent)

        # Numpy types: https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
        if is_type(np.integer):
            max_int = np.iinfo(dtype).max + 1
            assert max_int & (max_int - 1) == 0  # power of 2

            if is_type(np.unsignedinteger):
                self.center = max_int // 2
                self.max_val = max_int // 2

            elif is_type(np.signedinteger):
                self.center = 0
                self.max_val = max_int

        elif is_type(np.floating):
            self.center = 0
            self.max_val = 1

        else:
            raise CorrError(f"unexpected wavfile dtype {dtype}")

    def with_flatten(self, flatten: Flatten) -> "Wave":
        new = copy.copy(self)
        new.flatten = flatten
        return new

    def __getitem__(self, index: Union[int, slice]) -> np.ndarray:
        """ Copies self.data[item], converted to a FLOAT within range [-1, 1). """
        # subok=False converts data from memmap (slow) to ndarray (faster).
        data: np.ndarray = self.data[index].astype(FLOAT, subok=False, copy=True)

        # Flatten stereo to mono.
        flatten = self._flatten  # Potentially faster than property getter.
        if flatten == Flatten.Mono:
            data = data.reshape(-1)  # ndarray.flatten() creates copy, is slow.
        elif flatten != Flatten.Stereo:
            # data.strides = (4,), so data == contiguous float32
            if flatten & Flatten.SumAvg:
                data = data[..., 0] + data[..., 1]
            else:
                data = data[..., 0] - data[..., 1]
            data /= 2

        data -= self.center
        data *= self.amplification / self.max_val
        return data

    def _get(self, begin: int, end: int, subsampling: int) -> np.ndarray:
        """ Copies self.data[begin:end] with zero-padding. """
        if 0 <= begin and end <= self.nsamp:
            return self[begin:end:subsampling]

        region_len = end - begin

        def constrain(idx):
            delta = 0
            if idx < 0:
                delta = 0 - idx  # delta > 0
                assert idx + delta == 0

            if idx > self.nsamp:
                delta = self.nsamp - idx  # delta < 0
                assert idx + delta == self.nsamp

            return delta

        begin_index = constrain(begin)
        end_index = region_len + constrain(end)
        del end
        data = self[begin + begin_index : begin + end_index : subsampling]

        # Compute subsampled output ranges
        out_len = region_len // subsampling
        out_begin = begin_index // subsampling
        out_end = out_begin + len(data)
        # len(data) == ceil((end_index - begin_index) / subsampling)

        out = np.zeros((out_len, *data.shape[1:]), dtype=FLOAT)

        out[out_begin:out_end] = data

        return out

    def get_around(self, sample: int, return_nsamp: int, stride: int) -> np.ndarray:
        """ Returns `return_nsamp` samples, centered around `sample`,
        sampled with spacing `stride`.

        Copies self.data[...] """
        distance = return_nsamp * stride
        end = sample + distance // 2
        begin = end - distance
        return self._get(begin, end, stride)

    def get_s(self) -> float:
        """
        :return: time (seconds)
        """
        return self.nsamp / self.smp_s
