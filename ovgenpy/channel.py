from typing import TYPE_CHECKING, Any

from ovgenpy.config import register_config
from ovgenpy.wave import _WaveConfig, Wave


if TYPE_CHECKING:
    from ovgenpy.triggers import ITriggerConfig
    from ovgenpy.ovgenpy import Config


@register_config
class ChannelConfig:
    wav_path: str

    trigger: 'ITriggerConfig' = None    # TODO test channel-specific triggers
    trigger_width_ratio: int = 1
    render_width_ratio: int = 1

    ampl_ratio: float = 1.0     # TODO use amplification = None instead?
    line_color: Any = None


class Channel:
    def __init__(self, cfg: ChannelConfig, ovgen_cfg: 'Config'):
        self.cfg = cfg

        # Compute subsampling factors.
        self.trigger_subsampling = ovgen_cfg.subsampling * cfg.trigger_width_ratio
        self.render_subsampling = ovgen_cfg.subsampling * cfg.render_width_ratio

        # Create a Wave object. (TODO maybe create in Ovgenpy()?)
        wcfg = _WaveConfig(amplification=ovgen_cfg.amplification * cfg.ampl_ratio)
        self.wave = Wave(wcfg, cfg.wav_path)

        # Create a Trigger object.
        tcfg = cfg.trigger or ovgen_cfg.trigger
        trigger_nsamp = round(
            ovgen_cfg.render_width_s * cfg.trigger_width_ratio * self.wave.smp_s
        )
        self.trigger = tcfg(
            wave=self.wave,
            nsamp=trigger_nsamp,
            subsampling=ovgen_cfg.subsampling * self.trigger_subsampling
        )

