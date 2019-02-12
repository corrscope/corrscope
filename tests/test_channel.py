from typing import Optional

import hypothesis.strategies as hs
import numpy as np
import pytest
from hypothesis import given
from pytest_mock import MockFixture

import corrscope.channel
import corrscope.corrscope
from corrscope import parallelism
from corrscope.channel import ChannelConfig, Channel
from corrscope.corrscope import (
    default_config,
    CorrScope,
    BenchmarkMode,
    Arguments,
    RenderJob,
)
from corrscope.triggers import NullTriggerConfig
from corrscope.util import coalesce


positive = hs.integers(min_value=1, max_value=100)
real = hs.floats(min_value=0, max_value=100)
maybe_real = hs.one_of(hs.none(), real)


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
)
def test_config_channel_width_stride(
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
    mocker: MockFixture,
):
    """ (Tautologically) verify:
    - channel.  r_samp (given cfg)
    - channel.t/r_stride (given cfg.*_subsampling/*_width)
    - trigger._tsamp, _stride
    - renderer's method calls(samp, stride)
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
    )

    def get_cfg():
        return default_config(
            trigger_ms=trigger_ms,
            render_ms=render_ms,
            trigger_subsampling=tsub,
            render_subsampling=rsub,
            amplification=amplification,
            channels=[ccfg],
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
    assert channel.render_samp == ideal_rsamp

    assert channel.trigger_stride == tsub * c_trigger_width
    assert channel.render_stride == rsub * c_render_width

    # Ensure amplification override works
    args, kwargs = Wave.call_args
    assert kwargs["amplification"] == coalesce(c_amplification, amplification)

    ## Ensure trigger uses channel.window_samp and trigger_stride.
    trigger = channel.trigger
    assert trigger._tsamp == ideal_tsamp
    assert trigger._stride == channel.trigger_stride

    ## Ensure corrscope calls render using channel.render_samp and render_stride.
    corr = CorrScope(
        cfg, Arguments(cfg_dir=".", outputs=[], worker=parallelism.SerialWorker)
    )
    renderer = mocker.patch.object(RenderJob, "_load_renderer").return_value
    corr.play()

    # Only render (not NullTrigger) calls wave.get_around().
    (_sample, _return_nsamp, _subsampling), kwargs = wave.get_around.call_args
    assert _return_nsamp == channel.render_samp
    assert _subsampling == channel.render_stride

    # Inspect arguments to renderer.render_frame()
    # datas: List[np.ndarray]
    (datas,), kwargs = renderer.render_frame.call_args
    render_data = datas[0]
    assert len(render_data) == channel.render_samp


# line_color is tested in test_renderer.py
