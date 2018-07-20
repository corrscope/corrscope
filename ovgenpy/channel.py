from typing import NamedTuple, TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from ovgenpy.triggers import Trigger
    from ovgenpy.wave import Wave
    from ovgenpy.ovgenpy import Config


class ChannelConfig(NamedTuple):
    _main_cfg: 'Config'
    visible_ms: Optional[int]
    scan_ratio: Optional[float]
    line_color: Any
    background_color: Any


class Channel:
    def __init__(self, cfg: ChannelConfig, wave: 'Wave', trigger: 'Trigger'):
        self.cfg = cfg
        self.wave = wave
        self.trigger = trigger
