import numpy as np
import pytest
from hypothesis import given, reproduce_failure
from hypothesis.strategies import integers
from pytest_mock import MockFixture

import ovgenpy.channel
import ovgenpy.ovgenpy
from ovgenpy.channel import ChannelConfig, Channel
from ovgenpy.ovgenpy import default_config, Ovgen, BenchmarkMode
from ovgenpy.triggers import NullTriggerConfig

assert reproduce_failure


positive = integers(min_value=1, max_value=100)

@given(subsampling=positive, trigger_width=positive, render_width=positive)
def test_channel_subsampling(
    subsampling: int,
    trigger_width: int,
    render_width: int,
    mocker: MockFixture
):
    """ Ensure window_samp and trigger/render subsampling are computed correctly. """

    ovgenpy.ovgenpy.PRINT_TIMESTAMP = False    # Cleanup Hypothesis testing logs

    Wave = mocker.patch.object(ovgenpy.channel, 'Wave')
    wave = Wave.return_value

    def get_around(sample: int, region_nsamp: int, subsampling: int):
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
        trigger=NullTriggerConfig(),
        outputs=[],
        benchmark_mode=BenchmarkMode.OUTPUT
    )
    channel = Channel(ccfg, cfg)

    # Ensure channel.window_samp, trigger_subsampling, render_subsampling are correct.
    ideal_nsamp = pytest.approx(
        round(cfg.render_width_s * channel.wave.smp_s / subsampling), 1)

    assert channel.window_samp == ideal_nsamp
    assert channel.trigger_subsampling == subsampling * trigger_width
    assert channel.render_subsampling == subsampling * render_width

    # Ensure trigger uses channel.window_samp and trigger_subsampling.
    trigger = channel.trigger
    assert trigger._tsamp == channel.window_samp
    assert trigger._subsampling == channel.trigger_subsampling

    # Ensure ovgenpy calls render using channel.window_samp and render_subsampling.
    ovgen = Ovgen(cfg, '.')
    renderer = mocker.patch.object(Ovgen, '_load_renderer').return_value
    ovgen.play()

    # Inspect arguments to wave.get_around()
    (_sample, _region_nsamp, _subsampling), kwargs = wave.get_around.call_args
    assert _region_nsamp == channel.window_samp
    assert _subsampling == channel.render_subsampling

    # Inspect arguments to renderer.render_frame()
    # datas: List[np.ndarray]
    (datas,), kwargs = renderer.render_frame.call_args
    render_data = datas[0]
    assert len(render_data) == channel.window_samp


# line_color is tested in test_renderer.py
# todo test ChannelConfig.ampl_ratio
