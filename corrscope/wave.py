import copy
import enum
from typing import Union, List

import numpy as np

import corrscope.utils.scipy.wavfile as wavfile
from corrscope.config import CorrError, TypedEnumDump
from corrscope.utils.windows import rightpad
from corrscope.wave_common import f32


@enum.unique
class Flatten(str, TypedEnumDump):
    """How to flatten a stereo signal. (Channels beyond first 2 are ignored.)

    Flatten(0) == Flatten.Stereo == Flatten['Stereo']
    """

    # Keep both channels.
    Stereo = "stereo"

    # Mono
    Mono = "1"  # NOT publicly exposed

    # Take sum or difference.
    SumAvg = "1 1"
    DiffAvg = "1, -1"

    def __str__(self):
        return self.value

    # Both our app and GUI treat:
    # - Flatten.SumAvg -> "sum of all channels"
    # - "1 1" -> "assert nchan == 2, left + right".
    # - "1 0" -> "assert nchan == 2, left".
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(self.value)

    modes: List["Flatten"]


assert "1" == str(Flatten.Mono)
assert not "1" == Flatten.Mono
assert not Flatten.Mono == "1"

FlattenOrStr = Union[Flatten, str]


def calc_flatten_matrix(flatten: FlattenOrStr, stereo_nchan: int) -> np.ndarray:
    """Raises CorrError on invalid input.

    If flatten is Flatten.Stereo, returns shape=(nchan,nchan) identity matrix.
    - (N,nchan) @ (nchan,nchan) = (N,nchan).

    Otherwise, returns shape=(nchan) flattening matrix.
    - (N,nchan) @ (nchan) = (N)

    https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
    '''
    If the second argument is 1-D,
    it is promoted to a matrix by appending a 1 to its dimensions.
    After matrix multiplication the appended 1 is removed."
    '''
    """

    if flatten is Flatten.Stereo:
        # 2D identity (results in 2-dim data)
        flatten_matrix = np.eye(stereo_nchan, dtype=f32)

    # 1D (results in 1-dim data)
    elif flatten is Flatten.SumAvg:
        flatten_matrix = np.ones(stereo_nchan, dtype=f32) / stereo_nchan

    elif flatten is Flatten.DiffAvg:
        flatten_matrix = calc_flatten_matrix(str(flatten), stereo_nchan)
        flatten_matrix = rightpad(flatten_matrix, stereo_nchan, 0)

    else:
        words = flatten.replace(",", " ").split()
        try:
            flatten_matrix = np.array([f32(word) for word in words])
        except ValueError as e:
            raise CorrError("Invalid stereo flattening matrix") from e

        flatten_abs_sum = np.sum(np.abs(flatten_matrix))
        if flatten_abs_sum == 0:
            raise CorrError("Stereo flattening matrix must have nonzero elements")

        flatten_matrix /= flatten_abs_sum

    assert flatten_matrix.dtype == f32, flatten_matrix.dtype
    return flatten_matrix


_rejected_modes = {Flatten.Mono}
Flatten.modes = [f for f in Flatten.__members__.values() if f not in _rejected_modes]


class Wave:
    smp_s: int
    data: np.ndarray

    _flatten: FlattenOrStr
    flatten_matrix: np.ndarray

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
    def flatten(self, flatten: FlattenOrStr) -> None:
        # Reject invalid modes (including Mono).
        if flatten in _rejected_modes:
            # Flatten.Mono not in Flatten.modes.
            raise CorrError(
                f"Wave {self.wave_path} has invalid flatten mode {flatten} "
                f"not a numeric string, nor in {Flatten.modes}"
            )

        # If self.is_mono, converts all non-Stereo modes to Mono.
        self._flatten = flatten
        if self.is_mono and flatten != Flatten.Stereo:
            self._flatten = Flatten.Mono

        self.flatten_matrix = calc_flatten_matrix(self._flatten, self.stereo_nchan)

    def __init__(
        self,
        wave_path: str,
        amplification: float = 1.0,
        flatten: FlattenOrStr = Flatten.SumAvg,
    ):
        self.wave_path = wave_path
        self.amplification = amplification
        self.offset = 0

        # self.data: 2-D array of shape (nsamp, nchan)
        self.smp_s, self.data = wavfile.read(wave_path, mmap=True)

        assert self.data.ndim in [1, 2]
        self.is_mono = self.data.ndim == 1
        self.return_channels = False

        # Cast self.data to stereo (nsamp, nchan)
        if self.is_mono:
            self.data.shape = (-1, 1)

        self.nsamp, self.stereo_nchan = self.data.shape

        # Depends on self.stereo_nchan
        self.flatten = flatten

        # Calculate scaling factor.
        dtype = self.data.dtype

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

    def with_flatten(self, flatten: FlattenOrStr, return_channels: bool) -> "Wave":
        new = copy.copy(self)
        new.flatten = flatten
        new.return_channels = return_channels
        return new

    def with_offset(self, offset: float):
        """offset is applied *after* amplification,
        and corresponds directly to the output signal."""

        new = copy.copy(self)
        new.offset = offset
        return new

    def __getitem__(self, index: Union[int, slice]) -> np.ndarray:
        """Copies self.data[item], converted to a f32 within range [-1, 1)."""
        # subok=False converts data from memmap (slow) to ndarray (faster).
        data: np.ndarray = self.data[index].astype(f32, subok=False, copy=True)

        # Flatten stereo to mono.
        data = data @ self.flatten_matrix

        data -= self.center
        data *= self.amplification / self.max_val
        data += self.offset

        if self.return_channels and len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return data

    def get_padded(self, begin: int, end: int, subsampling: int) -> np.ndarray:
        """Copies self.data[begin:end] with zero-padding."""
        if 0 <= begin and end <= self.nsamp:
            return self[begin:end:subsampling]

        region_len = end - begin

        def constrain(idx: int) -> int:
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

        out = np.zeros((out_len, *data.shape[1:]), dtype=f32)

        out[out_begin:out_end] = data

        return out

    def get_around(self, sample: int, return_nsamp: int, stride: int) -> np.ndarray:
        """Returns `return_nsamp` samples, centered around `sample`,
        sampled with spacing `stride`.
        result[N//2] == self[sample].
        See designNotes.md and CorrelationTrigger docstring.

        Copies self.data[...]."""

        begin = sample - (return_nsamp // 2) * stride
        end = begin + return_nsamp * stride
        return self.get_padded(begin, end, stride)

    def get_s(self) -> float:
        """
        :return: time (seconds)
        """
        return self.nsamp / self.smp_s
