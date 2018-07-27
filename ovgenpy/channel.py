from typing import NamedTuple, TYPE_CHECKING, Any, Optional

from ovgenpy.config import register_config


if TYPE_CHECKING:
    from ovgenpy.triggers import Trigger
    from ovgenpy.wave import Wave
    from ovgenpy.ovgenpy import Config


@register_config
class ChannelConfig:
    _main_cfg: 'Config'

    trigger_width_ratio: int = 1
    render_width_ratio: int = 1

    amplification: float = 1.0
    line_color: Any = None
    background_color: Any = None


class Channel:
    def __init__(self, cfg: ChannelConfig, wave: 'Wave', trigger: 'Trigger'):
        self.cfg = cfg
        self.wave = wave
        self.trigger = trigger
