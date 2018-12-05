from os.path import abspath
from typing import TYPE_CHECKING, Optional

from ovgenpy.config import register_config, Alias
from ovgenpy.util import coalesce
from ovgenpy.wave import _WaveConfig, Wave


if TYPE_CHECKING:
    from ovgenpy.triggers import ITriggerConfig
    from ovgenpy.ovgenpy import Config


@register_config
class ChannelConfig:
    wav_path: str

    trigger: Optional['ITriggerConfig'] = None    # TODO test channel-specific triggers
    # Multiplies how wide the window is, in milliseconds.
    trigger_width: Optional[int] = None
    render_width: Optional[int] = None

    ampl_ratio: float = 1.0     # TODO use amplification = None instead?
    line_color: Optional[str] = None

    # region Legacy Fields
    trigger_width_ratio = Alias('trigger_width')
    render_width_ratio = Alias('render_width')
    # endregion


class Channel:
    # trigger_samp is unneeded, since __init__ (not Ovgenpy) constructs triggers.
    render_samp: int
    # TODO add a "get_around" method for rendering (also helps test_channel_subsampling)
    # Currently Ovgenpy peeks at Chanel.render_samp and render_stride (bad).

    # Product of ovgen_cfg.trigger/render_subsampling and trigger/render_width.
    trigger_stride: int
    render_stride: int

    def __init__(self, cfg: ChannelConfig, ovgen_cfg: 'Config'):
        self.cfg = cfg

        # Create a Wave object.
        wcfg = _WaveConfig(amplification=ovgen_cfg.amplification * cfg.ampl_ratio)
        self.wave = Wave(wcfg, abspath(cfg.wav_path))

        # `subsampling` increases `stride` and decreases `nsamp`.
        # `width` increases `stride` without changing `nsamp`.
        tsub = ovgen_cfg.trigger_subsampling
        # FIXME remove ovgen_cfg._width, replace with 1
        tw = coalesce(cfg.trigger_width, ovgen_cfg.trigger_width)

        rsub = ovgen_cfg.render_subsampling
        rw = coalesce(cfg.render_width, ovgen_cfg.render_width)

        # nsamp = orig / subsampling
        # stride = subsampling * width
        def calculate_nsamp(width_ms, sub):
            width_s = width_ms / 1000
            return round(width_s * self.wave.smp_s / sub)

        trigger_samp = calculate_nsamp(ovgen_cfg.trigger_ms, tsub)
        self.render_samp = calculate_nsamp(ovgen_cfg.render_ms, rsub)

        self.trigger_stride = tsub * tw
        self.render_stride = rsub * rw

        # Create a Trigger object.
        tcfg = cfg.trigger or ovgen_cfg.trigger
        self.trigger = tcfg(
            wave=self.wave,
            tsamp=trigger_samp,
            stride=self.trigger_stride,
            fps=ovgen_cfg.fps
        )

