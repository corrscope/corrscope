from typing import TYPE_CHECKING, Any

from ovgenpy.config import register_config, Alias
from ovgenpy.util import coalesce
from ovgenpy.wave import _WaveConfig, Wave


if TYPE_CHECKING:
    from ovgenpy.triggers import ITriggerConfig
    from ovgenpy.ovgenpy import Config


@register_config
class ChannelConfig:
    wav_path: str

    trigger: 'ITriggerConfig' = None    # TODO test channel-specific triggers
    # Multiplies how wide the window is, in milliseconds.
    trigger_width: int = None
    render_width: int = None

    ampl_ratio: float = 1.0     # TODO use amplification = None instead?
    line_color: Any = None

    # region Legacy Fields
    trigger_width_ratio = Alias('trigger_width')
    render_width_ratio = Alias('render_width')
    # endregion


class Channel:
    # Shared between trigger and renderer.
    window_samp: int

    # Product of ovgen_cfg.subsampling and trigger/render_width.
    trigger_subsampling: int
    render_subsampling: int

    def __init__(self, cfg: ChannelConfig, ovgen_cfg: 'Config'):
        self.cfg = cfg
        subsampling = ovgen_cfg.subsampling

        # Create a Wave object.
        wcfg = _WaveConfig(amplification=ovgen_cfg.amplification * cfg.ampl_ratio)
        self.wave = Wave(wcfg, cfg.wav_path)

        # Compute subsampling (array stride).
        tw = coalesce(cfg.trigger_width, ovgen_cfg.trigger_width)
        self.trigger_subsampling = subsampling * tw

        rw = coalesce(cfg.render_width, ovgen_cfg.render_width)
        self.render_subsampling = subsampling * rw

        # Compute window_samp and tsamp_frame.
        nsamp = ovgen_cfg.render_width_s * self.wave.smp_s / subsampling
        self.window_samp = round(nsamp)

        del subsampling
        del nsamp

        # Create a Trigger object.
        tcfg = cfg.trigger or ovgen_cfg.trigger
        self.trigger = tcfg(
            wave=self.wave,
            tsamp=self.window_samp,
            subsampling=self.trigger_subsampling,
            fps=ovgen_cfg.fps
        )

