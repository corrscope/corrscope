from typing import NamedTuple, TYPE_CHECKING, Optional

import numpy as np
from scipy.io import wavfile

if TYPE_CHECKING:
    from ovgenpy.triggers import Trigger


class WaveConfig(NamedTuple):
    wave_path: str
    # TODO color

    # TODO wave-specific trigger options?


FLOAT = np.single


class Wave:
    def __init__(self, wcfg: Optional[WaveConfig], wave_path: str):
        self.cfg = wcfg
        self.smp_s, self.data = wavfile.read(wave_path, mmap=True)  # type: int, np.ndarray
        self.nsamp = len(self.data)
        self.trigger: Trigger = None

        # Calculate scaling factor.
        dtype = self.data.dtype

        def is_type(parent: type) -> bool:
            return np.issubdtype(dtype, parent)

        if is_type(np.integer):
            max_int = np.iinfo(dtype).max + 1
            assert max_int & (max_int - 1) == 0  # power of 2

            if is_type(np.unsignedinteger):
                self.center = max_int // 2
                self.max_val = max_int // 2

            elif is_type(np.signedinteger):
                self.center = 0
                self.max_val = max_int

        elif is_type(np.float):
            self.center = 0
            self.max_val = 1

        else:
            raise ValueError(f'unexpected wavfile dtype {dtype}')

    def __getitem__(self, index: int) -> 'np.ndarray[FLOAT]':
        """ Copies self.data[item], converted to a FLOAT within range [-1, 1). """
        data = self.data[index].astype(FLOAT)
        data -= self.center
        data /= self.max_val
        return data

    def get(self, begin: int, end: int) -> 'np.ndarray[FLOAT]':
        """ Copies self.data[begin:end] with zero-padding. """
        if 0 <= begin and end <= self.nsamp:
            return self[begin:end]

        region_len = end - begin

        def constrain(idx):
            delta = 0
            if idx < 0:
                delta = 0 - idx             # delta > 0
                assert idx + delta == 0

            if idx > self.nsamp:
                delta = self.nsamp - idx    # delta < 0
                assert idx + delta == self.nsamp

            return delta, idx

        delta_begin, begin = constrain(begin)
        delta_end, end = constrain(end)

        out = np.zeros(region_len, dtype=FLOAT)

        # out[0 : region_len]. == self[begin: end]
        # out[Δbegin : region_len+Δend] == self[begin + Δbegin: end + Δend]
        out[delta_begin : region_len+delta_end] = self[begin+delta_begin : end+delta_end]
        return out

    def get_around(self, sample: int, region_len: int):
        """" Copies self.data[...] """
        end = sample + region_len // 2
        begin = end - region_len
        return self.get(begin, end)

    def set_trigger(self, trigger: 'Trigger'):
        self.trigger = trigger

    def get_s(self) -> float:
        """
        :return: time (seconds)
        """
        return self.nsamp / self.smp_s


