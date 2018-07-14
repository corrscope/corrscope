from abc import ABC, abstractmethod
from typing import NamedTuple, List, Dict, Any

import numpy as np

from ovgenpy.wave import Wave


class TriggerConfig(NamedTuple):
    name: str
    # scan_nsamp: int
    args: List = []
    kwargs: Dict[str, Any] = {}

    def generate_trigger(self, wave: Wave, scan_nsamp: int) -> 'Trigger':
        return TRIGGERS[self.name](wave, scan_nsamp, *self.args, **self.kwargs)


TRIGGERS: Dict[str, type] = {}


def register_trigger(trigger_class: type):
    TRIGGERS[trigger_class.__name__] = trigger_class
    return trigger_class


class Trigger(ABC):
    @abstractmethod
    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        ...




@register_trigger
class CorrelationTrigger(Trigger):
    def __init__(self, wave: Wave, scan_nsamp: int, align_amount: float):
        """
        Correlation-based trigger which looks at a window of `scan_nsamp` samples.

        it's complicated

        :param wave: Wave file
        :param scan_nsamp: Number of samples used to align adjacent frames
        :param align_amount: Amount of centering to apply to each frame, within [0, 1]
        """

        self.wave = wave
        self.scan_nsamp = scan_nsamp
        self.align_amount = align_amount

        # Correlation buffer containing a series of old data
        self._prev_buffer = np.zeros(scan_nsamp)
        self._update_buffer(self._prev_buffer)

    def _update_buffer(self, data: np.ndarray) -> None:
        """
        Update self._prev_buffer by adding `data` and a step function.
        Data is reshaped to taper away from the center.
        :param data: Wave data, containing scan_nsamp samples
        """
        self._prev_buffer = data    # TODO

    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        data = self.wave.get_around(offset, self.scan_nsamp)
        self._update_buffer(data)
        return offset  # todo
