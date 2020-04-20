from typing import Optional, List

import hypothesis.strategies as hs
import numpy as np
import pytest
from hypothesis import given
from pytest_mock import MockFixture

import corrscope.channel
import corrscope.corrscope
from corrscope.channel import ChannelConfig, Channel, DefaultLabel
from corrscope.corrscope import template_config, CorrScope, BenchmarkMode, Arguments
from corrscope.renderer import RenderInput
from corrscope.triggers import NullTriggerConfig
from corrscope.util import coalesce
from corrscope.wave import Flatten


positive = hs.integers(min_value=1, max_value=100)
real = hs.floats(min_value=0, max_value=100)
maybe_real = hs.one_of(hs.none(), real)
bools = hs.booleans()
default_labels = hs.sampled_from(DefaultLabel)


@given(
    # Channel
    c_amplification=maybe_real,
    c_trigger_width=positive,
    c_render_width=positive,
    # Global
    amplification=real,
    trigger_ms=positive,
    render_ms=positive,
    tsub=positive,
    rsub=positive,
    default_label=hs.sampled_from(DefaultLabel),
    override_label=bools,
)
def test_config_channel_integration(
    # Channel
    c_amplification: Optional[float],
    c_trigger_width: int,
    c_render_width: int,
    # Global
    amplification: float,
    trigger_ms: int,
    render_ms: int,
    tsub: int,
    rsub: int,
    default_label: DefaultLabel,
    override_label: bool,
    mocker: MockFixture,
):
    """ (Tautologically) verify:
    - channel.  r_samp (given cfg)
    - channel.t/r_stride (given cfg.*_subsampling/*_width)
    - trigger._tsamp, _stride
    - renderer's method calls(samp, stride)
    - rendered label (channel.label, given cfg, corr_cfg.default_label)
    """

    # region setup test variables
    corrscope.corrscope.PRINT_TIMESTAMP = False  # Cleanup Hypothesis testing logs

    Wave = mocker.patch.object(corrscope.channel, "Wave")
    wave = Wave.return_value

    def get_around(sample: int, return_nsamp: int, stride: int):
        return np.zeros(return_nsamp)

    wave.get_around.side_effect = get_around
    wave.with_flatten.return_value = wave
    wave.nsamp = 10000
    wave.smp_s = 48000

    ccfg = ChannelConfig(
        "tests/sine440.wav",
        trigger_width=c_trigger_width,
        render_width=c_render_width,
        amplification=c_amplification,
        label="label" if override_label else "",
    )

    def get_cfg():
        return template_config(
            trigger_ms=trigger_ms,
            render_ms=render_ms,
            trigger_subsampling=tsub,
            render_subsampling=rsub,
            amplification=amplification,
            channels=[ccfg],
            default_label=default_label,
            trigger=NullTriggerConfig(),
            benchmark_mode=BenchmarkMode.OUTPUT,
        )

    # endregion

    cfg = get_cfg()
    channel = Channel(ccfg, cfg)

    # Ensure cfg.width_ms etc. are correct
    assert cfg.trigger_ms == trigger_ms
    assert cfg.render_ms == render_ms

    # Ensure channel.window_samp, trigger_subsampling, render_subsampling are correct.
    def ideal_samp(width_ms, sub):
        width_s = width_ms / 1000
        return pytest.approx(
            round(width_s * channel.trigger_wave.smp_s / sub), rel=1e-6
        )

    ideal_tsamp = ideal_samp(cfg.trigger_ms, tsub)
    ideal_rsamp = ideal_samp(cfg.render_ms, rsub)
    assert channel._render_samp == ideal_rsamp

    assert channel._trigger_stride == tsub * c_trigger_width
    assert channel.render_stride == rsub * c_render_width

    # Ensure amplification override works
    args, kwargs = Wave.call_args
    assert kwargs["amplification"] == coalesce(c_amplification, amplification)

    ## Ensure trigger uses channel.window_samp and _trigger_stride.
    trigger = channel.trigger
    assert trigger._tsamp == ideal_tsamp
    assert trigger._stride == channel._trigger_stride

    ## Ensure corrscope calls render using channel._render_samp and _render_stride.
    corr = CorrScope(cfg, Arguments(cfg_dir=".", outputs=[]))
    renderer = mocker.patch.object(CorrScope, "_load_renderer").return_value
    corr.play()

    # Only Channel.get_render_around() (not NullTrigger) calls wave.get_around().
    (_sample, _return_nsamp, _subsampling), kwargs = wave.get_around.call_args
    assert _return_nsamp == channel._render_samp
    assert _subsampling == channel.render_stride

    # Inspect arguments to renderer.update_main_lines()
    datas: List[RenderInput]
    (datas,), kwargs = renderer.update_main_lines.call_args
    render_inputs = datas[0]
    assert len(render_inputs.data) == channel._render_samp

    # Inspect arguments to renderer.add_labels().
    (labels,), kwargs = renderer.add_labels.call_args
    label = labels[0]
    if override_label:
        assert label == "label"
    else:
        if default_label is DefaultLabel.FileName:
            assert label == "sine440"
        elif default_label is DefaultLabel.Number:
            assert label == "1"
        else:
            assert label == ""


# line_color is tested in test_renderer.py


@pytest.mark.parametrize("filename", ["tests/sine440.wav", "tests/stereo in-phase.wav"])
@pytest.mark.parametrize(
    ("global_stereo", "chan_stereo"),
    [
        [Flatten.SumAvg, None],
        [Flatten.Stereo, None],
        [Flatten.SumAvg, Flatten.Stereo],
        [Flatten.Stereo, Flatten.SumAvg],
        [Flatten.Stereo, "1 0"],
    ],
)
def test_per_channel_stereo(
    filename: str, global_stereo: Flatten, chan_stereo: Optional[Flatten]
):
    """Ensure you can enable/disable stereo on a per-channel basis."""
    stereo = coalesce(chan_stereo, global_stereo)

    # Test render wave.
    cfg = template_config(render_stereo=global_stereo)
    ccfg = ChannelConfig("tests/stereo in-phase.wav", render_stereo=chan_stereo)
    channel = Channel(ccfg, cfg)

    # Render wave *must* return stereo.
    assert channel.render_wave[0:1].ndim == 2
    data = channel.render_wave.get_around(0, return_nsamp=4, stride=1)
    assert data.ndim == 2

    if "stereo" in filename:
        assert channel.render_wave._flatten == stereo
        assert data.shape[1] == (2 if stereo is Flatten.Stereo else 1)
