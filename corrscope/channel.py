from os.path import abspath
from typing import TYPE_CHECKING, Optional, Union, Dict, Any

import attr
from ruamel.yaml.comments import CommentedMap

from corrscope.config import DumpableAttrs, Alias, CorrError
from corrscope.triggers import ITriggerConfig
from corrscope.util import coalesce
from corrscope.wave import Wave, Flatten

if TYPE_CHECKING:
    from corrscope.corrscope import Config


class ChannelConfig(DumpableAttrs):
    wav_path: str

    # Supplying a dict inherits attributes from global trigger.
    # TODO test channel-specific triggers
    trigger: Union[ITriggerConfig, Dict[str, Any], None] = attr.Factory(dict)

    # Multiplies how wide the window is, in milliseconds.
    trigger_width: int = 1
    render_width: int = 1

    # Overrides global amplification.
    amplification: Optional[float] = None

    # Stereo config
    trigger_stereo: Optional[Flatten] = None
    render_stereo: Optional[Flatten] = None

    line_color: Optional[str] = None

    # region Legacy Fields
    trigger_width_ratio = Alias("trigger_width")
    render_width_ratio = Alias("render_width")
    # endregion


class Channel:
    # trigger_samp is unneeded, since __init__ (not CorrScope) constructs triggers.
    render_samp: int
    # TODO add a "get_around" method for rendering (also helps test_channel_subsampling)
    # Currently CorrScope peeks at Channel.render_samp and render_stride (bad).

    # Product of corr_cfg.trigger/render_subsampling and trigger/render_width.
    trigger_stride: int
    render_stride: int

    def __init__(self, cfg: ChannelConfig, corr_cfg: "Config"):
        self.cfg = cfg

        # Create a Wave object.
        wave = Wave(
            abspath(cfg.wav_path),
            amplification=coalesce(cfg.amplification, corr_cfg.amplification),
        )

        # Flatten wave stereo for trigger and render.
        tflat = coalesce(cfg.trigger_stereo, corr_cfg.trigger_stereo)
        rflat = coalesce(cfg.render_stereo, corr_cfg.render_stereo)

        self.trigger_wave = wave.with_flatten(tflat, return_channels=False)
        self.render_wave = wave.with_flatten(rflat, return_channels=True)

        # `subsampling` increases `stride` and decreases `nsamp`.
        # `width` increases `stride` without changing `nsamp`.
        tsub = corr_cfg.trigger_subsampling
        tw = cfg.trigger_width

        rsub = corr_cfg.render_subsampling
        rw = cfg.render_width

        # nsamp = orig / subsampling
        # stride = subsampling * width
        def calculate_nsamp(width_ms, sub):
            width_s = width_ms / 1000
            return round(width_s * wave.smp_s / sub)

        trigger_samp = calculate_nsamp(corr_cfg.trigger_ms, tsub)
        self.render_samp = calculate_nsamp(corr_cfg.render_ms, rsub)

        self.trigger_stride = tsub * tw
        self.render_stride = rsub * rw

        # Create a Trigger object.
        if isinstance(cfg.trigger, ITriggerConfig):
            tcfg = cfg.trigger
        elif isinstance(
            cfg.trigger, (CommentedMap, dict)
        ):  # CommentedMap may/not be subclass of dict.
            tcfg = attr.evolve(corr_cfg.trigger, **cfg.trigger)
        elif cfg.trigger is None:
            tcfg = corr_cfg.trigger
        else:
            raise CorrError(
                f"invalid per-channel trigger {cfg.trigger}, type={type(cfg.trigger)}, "
                f"must be (*)TriggerConfig, dict, or None"
            )

        self.trigger = tcfg(
            wave=self.trigger_wave,
            tsamp=trigger_samp,
            stride=self.trigger_stride,
            fps=corr_cfg.fps,
        )
