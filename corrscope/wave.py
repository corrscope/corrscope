import enum
import warnings
from enum import auto
from typing import Optional, Union

import attr
import numpy as np

import corrscope.utils.scipy_wavfile as wavfile
from corrscope.config import CorrError, CorrWarning


FLOAT = np.single


@enum.unique
class Flatten(enum.Flag):
    """ How to flatten a stereo signal. (Channels beyond first 2 are ignored.)

    Flatten(0) == Flatten.Stereo == Flatten['Stereo']
    """

    # Keep both channels.
    Stereo = 0

    # Take sum or difference.
    Sum = auto()
    Diff = auto()

    # Divide by nchan=2.
    IsAvg = auto()
    SumAvg = Sum | IsAvg
    DiffAvg = Diff | IsAvg

    def __str__(self):
        try:
            return flat2str[self]
        except KeyError:
            return repr(self)


str2flat = {
    "=": Flatten.Stereo,
    "+": Flatten.Sum,
    "+/": Flatten.SumAvg,
    "-": Flatten.Diff,
    "-/": Flatten.DiffAvg,
}
flat2str = {flat: str_ for str_, flat in str2flat.items()}


def make_flatten(obj: Union[str, Flatten]):
    if isinstance(obj, Flatten):
        return obj
    try:
        return str2flat[obj]
    except KeyError:
        raise CorrError(f"invalid Flatten mode {obj} not in {list(str2flat.keys())}")


@attr.dataclass(kw_only=True)
class _WaveConfig:
    """Internal class, not exposed via YAML"""

    amplification: float = 1
    flatten: Flatten = Flatten.SumAvg


class Wave:
    __slots__ = (
        "cfg smp_s data nsamp dtype is_stereo stereo_nchan center max_val".split()
    )

    smp_s: int
    data: "np.ndarray"
    """2-D array of shape (nsamp, nchan)"""

    def __init__(self, cfg: Optional[_WaveConfig], wave_path: str):
        self.cfg = cfg or _WaveConfig()
        self.smp_s, self.data = wavfile.read(
            wave_path, mmap=True
        )  # type: int, np.ndarray

        # self.is_stereo = self.data.ndim == 2

        # Cast self.data to stereo (nsamp, nchan)
        assert self.data.ndim in [1, 2]
        if self.data.ndim == 1:
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

    def __getitem__(self, index: Union[int, slice]) -> "np.ndarray[FLOAT]":
        """ Copies self.data[item], converted to a FLOAT within range [-1, 1). """
        # subok=False converts data from memmap (slow) to ndarray (faster).
        data = self.data[index].astype(FLOAT, subok=False, copy=True)

        # Flatten stereo to mono.
        flatten = self.cfg.flatten
        if flatten:
            # data.strides = (4,), so data == contiguous float32
            if flatten & Flatten.Sum:
                data = data[..., 0] + data[..., 1]
            else:
                data = data[..., 0] - data[..., 1]

            if flatten & Flatten.IsAvg:
                data /= 2

        data -= self.center
        data *= self.cfg.amplification / self.max_val
        return data

    def _get(self, begin: int, end: int, subsampling: int) -> "np.ndarray[FLOAT]":
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

    def get_around(self, sample: int, return_nsamp: int, stride: int):
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
