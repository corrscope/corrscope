from typing import Optional

import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as hs
from pytest_mock import MockFixture

import ovgenpy.channel
import ovgenpy.ovgenpy
from ovgenpy.channel import ChannelConfig, Channel
from ovgenpy.config import OvgenError
from ovgenpy.ovgenpy import default_config, Ovgen, BenchmarkMode
from ovgenpy.triggers import NullTriggerConfig
from ovgenpy.util import coalesce


positive = hs.integers(min_value=1, max_value=100)
Positive = int

# In order to get good shrinking behaviour, try to put simpler strategies first.
maybe = hs.one_of(hs.none(), positive)
Maybe = Optional[int]


@pytest.mark.filterwarnings("ignore::ovgenpy.config.OvgenWarning")
@given(
    # Channel
    c_trigger_width=maybe, c_render_width=maybe,

    # Global
    width_ms=maybe, trigger_ms=maybe, render_ms=maybe,
    subsampling=positive, tsub=maybe, rsub=maybe,
    g_trigger_width=positive, g_render_width=positive,

)
def test_config_channel_width_stride(
    # Channel
    c_trigger_width: Maybe, c_render_width: Maybe,

    # Global
    width_ms: Maybe, trigger_ms: Maybe, render_ms: Maybe,
    subsampling: Positive, tsub: Maybe, rsub: Maybe,
    g_trigger_width: Positive, g_render_width: Positive,

    mocker: MockFixture,
):
    """ (Tautologically) verify:
    -     cfg.t/r_ms (given width_ms)
    - channel.  r_samp (given cfg)
    - channel.t/r_stride (given cfg.sub/width and cfg.width)
    - trigger._tsamp, _stride
    - renderer's method calls(samp, stride)
    """

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
        trigger_width=c_trigger_width,
        render_width=c_render_width,
    )
    def get_cfg():
        return default_config(
            width_ms=width_ms,
            trigger_ms=trigger_ms,
            render_ms=render_ms,

            subsampling=subsampling,
            trigger_subsampling=tsub,
            render_subsampling=rsub,

            trigger_width=g_trigger_width,
            render_width=g_render_width,

            channels=[ccfg],
            trigger=NullTriggerConfig(),
            benchmark_mode=BenchmarkMode.OUTPUT
        )
    # endregion

    if not (width_ms or (trigger_ms and render_ms)):
        with pytest.raises(OvgenError):
            _cfg = get_cfg()
        return

    cfg = get_cfg()
    channel = Channel(ccfg, cfg)

    # Ensure cfg.width_ms etc. are correct
    assert cfg.trigger_ms == coalesce(trigger_ms, width_ms)
    assert cfg.render_ms == coalesce(render_ms, width_ms)

    # Ensure channel.window_samp, trigger_subsampling, render_subsampling are correct.
    tsub = coalesce(tsub, subsampling)
    rsub = coalesce(rsub, subsampling)

    def ideal_samp(width_ms, sub):
        width_s = width_ms / 1000
        return pytest.approx(
            round(width_s * channel.wave.smp_s / sub), rel=1e-6)

    ideal_tsamp = ideal_samp(cfg.trigger_ms, tsub)
    ideal_rsamp = ideal_samp(cfg.render_ms, rsub)
    assert channel.render_samp == ideal_rsamp
    del subsampling

    assert channel.trigger_stride == tsub * coalesce(c_trigger_width, g_trigger_width)
    assert channel.render_stride == rsub * coalesce(c_render_width, g_render_width)

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
