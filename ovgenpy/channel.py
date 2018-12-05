from os.path import abspath
from typing import TYPE_CHECKING, Optional, Union

import attr

from ovgenpy.config import register_config, Alias, OvgenError
from ovgenpy.triggers import ITriggerConfig
from ovgenpy.util import coalesce
from ovgenpy.wave import _WaveConfig, Wave

if TYPE_CHECKING:
    from ovgenpy.ovgenpy import Config


@register_config
class ChannelConfig:
    wav_path: str

    trigger: Union[ITriggerConfig, dict, None] = None    # TODO test channel-specific triggers
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
        tw = coalesce(cfg.trigger_width, ovgen_cfg.trigger_width)

        rsub = ovgen_cfg.render_subsampling
        rw = coalesce(cfg.render_width, ovgen_cfg.render_width)

        # nsamp = orig / subsampling
        # stride = subsampling * width
        def calculate_nsamp(sub):
            return round(ovgen_cfg.width_s * self.wave.smp_s / sub)
        trigger_samp = calculate_nsamp(tsub)
        self.render_samp = calculate_nsamp(rsub)

        self.trigger_stride = tsub * tw
        self.render_stride = rsub * rw

        # Create a Trigger object.
        if isinstance(cfg.trigger, ITriggerConfig):
            tcfg = cfg.trigger
        elif isinstance(cfg.trigger, dict):
            tcfg = attr.evolve(ovgen_cfg.trigger, **cfg.trigger)
        elif cfg.trigger is None:
            tcfg = ovgen_cfg.trigger
        else:
            raise OvgenError(
                f'invalid per-channel trigger {cfg.trigger}, type={type(cfg.trigger)}, '
                f'must be (*)TriggerConfig, dict, or None')

        self.trigger = tcfg(
            wave=self.wave,
            tsamp=trigger_samp,
            stride=self.trigger_stride,
            fps=ovgen_cfg.fps
        )

