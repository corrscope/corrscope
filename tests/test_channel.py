from typing import Optional

import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as hs
from pytest_mock import MockFixture

import ovgenpy.channel
import ovgenpy.ovgenpy
from ovgenpy.channel import ChannelConfig, Channel
from ovgenpy.ovgenpy import default_config, Ovgen, BenchmarkMode
from ovgenpy.triggers import NullTriggerConfig
from ovgenpy.util import coalesce

positive = hs.integers(min_value=1, max_value=100)
maybe = hs.one_of(positive, hs.none())

@given(subsampling=positive, tsub=maybe, rsub=maybe,
       trigger_width=positive, render_width=positive)
def test_channel_subsampling(
        subsampling: int,
        tsub: Optional[int],
        rsub: Optional[int],
        trigger_width: int,
        render_width: int,
        mocker: MockFixture
):
    """ Ensure trigger/render_samp and trigger/render subsampling
    are computed correctly. """

    # region setup test variables
    ovgenpy.ovgenpy.PRINT_TIMESTAMP = False    # Cleanup Hypothesis testing logs

    Wave = mocker.patch.object(ovgenpy.channel, 'Wave')
    wave = Wave.return_value

    def get_around(sample: int, region_nsamp: int, stride: int):
        return np.zeros(region_nsamp)

    wave.get_around.side_effect = get_around
    wave.nsamp = 10000
    wave.smp_s = 48000

    ccfg = ChannelConfig(
        'tests/sine440.wav',
        trigger_width=trigger_width,
        render_width=render_width,
    )
    cfg = default_config(
        channels=[ccfg],
        subsampling=subsampling,
        trigger_subsampling=tsub,
        render_subsampling=rsub,
        trigger=NullTriggerConfig(),
        benchmark_mode=BenchmarkMode.OUTPUT
    )
    channel = Channel(ccfg, cfg)
    # endregion

    # Ensure channel.window_samp, trigger_subsampling, render_subsampling are correct.
    tsub = coalesce(tsub, subsampling)
    rsub = coalesce(rsub, subsampling)

    def ideal_samp(width_ms, sub):
        width_s = width_ms / 1000
        return pytest.approx(
            round(width_s * channel.wave.smp_s / sub), abs=1)

    ideal_tsamp = ideal_samp(cfg.trigger_ms, tsub)
    ideal_rsamp = ideal_samp(cfg.render_ms, rsub)
    assert channel.render_samp == ideal_rsamp
    del subsampling

    assert channel.trigger_stride == tsub * trigger_width
    assert channel.render_stride == rsub * render_width

    ## Ensure trigger uses channel.window_samp and trigger_stride.
    trigger = channel.trigger
    assert trigger._tsamp == ideal_tsamp
    assert trigger._stride == channel.trigger_stride

    ## Ensure ovgenpy calls render using channel.render_samp and render_stride.
    ovgen = Ovgen(cfg, '.', outputs=[])
    renderer = mocker.patch.object(Ovgen, '_load_renderer').return_value
    ovgen.play()

    # Only render (not NullTrigger) calls wave.get_around().
    (_sample, _region_nsamp, _subsampling), kwargs = wave.get_around.call_args
    assert _region_nsamp == channel.render_samp
    assert _subsampling == channel.render_stride

    # Inspect arguments to renderer.render_frame()
    # datas: List[np.ndarray]
    (datas,), kwargs = renderer.render_frame.call_args
    render_data = datas[0]
    assert len(render_data) == channel.render_samp


# line_color is tested in test_renderer.py
# todo test ChannelConfig.ampl_ratio
