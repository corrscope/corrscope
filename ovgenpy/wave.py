from typing import Optional

import numpy as np
from ovgenpy.config import dataclass
from scipy.io import wavfile


# Internal class, not exposed via YAML
@dataclass
class _WaveConfig:
    amplification: float = 1


FLOAT = np.single


class Wave:
    def __init__(self, cfg: Optional[_WaveConfig], wave_path: str):
        self.cfg = cfg or _WaveConfig()
        self.smp_s, self.data = wavfile.read(wave_path, mmap=True)  # type: int, np.ndarray
        dtype = self.data.dtype

        # Flatten stereo to mono
        assert self.data.ndim in [1, 2]
        if self.data.ndim == 2:
            self.data = np.mean(self.data, axis=1, dtype=dtype)

        self.nsamp = len(self.data)

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
            raise ValueError(f'unexpected wavfile dtype {dtype}')

    def __getitem__(self, index: int) -> 'np.ndarray[FLOAT]':
        """ Copies self.data[item], converted to a FLOAT within range [-1, 1). """
        data = self.data[index].astype(FLOAT)
        data -= self.center
        data *= self.cfg.amplification / self.max_val
        return data

    def _get(self, begin: int, end: int, subsampling: int) -> 'np.ndarray[FLOAT]':
        """ Copies self.data[begin:end] with zero-padding. """
        if 0 <= begin and end <= self.nsamp:
            return self[begin:end:subsampling]

        region_len = end - begin

        def constrain(idx):
            delta = 0
            if idx < 0:
                delta = 0 - idx             # delta > 0
                assert idx + delta == 0

            if idx > self.nsamp:
                delta = self.nsamp - idx    # delta < 0
                assert idx + delta == self.nsamp

            return delta

        begin_index = constrain(begin)
        end_index = region_len + constrain(end)
        del end
        data = self[begin+begin_index : begin+end_index : subsampling]

        # Compute subsampled output ranges
        out_len = region_len // subsampling
        out_begin = begin_index // subsampling
        out_end = out_begin + len(data)
        # len(data) == ceil((end_index - begin_index) / subsampling)

        out = np.zeros(out_len, dtype=FLOAT)

        out[out_begin : out_end] = data

        return out

    def get_around(self, sample: int, region_nsamp: int, subsampling: int):
        """" Copies self.data[...] """
        region_nsamp *= subsampling
        end = sample + region_nsamp // 2
        begin = end - region_nsamp
        return self._get(begin, end, subsampling)

    def get_s(self) -> float:
        """
        :return: time (seconds)
        """
        return self.nsamp / self.smp_s


