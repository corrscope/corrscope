import pytest
from hypothesis import given, example
from hypothesis.strategies import integers


from ovgenpy.channel import ChannelConfig, Channel
from ovgenpy.ovgenpy import default_config


positive = integers(min_value=1)

@given(subsampling=positive, trigger_width_ratio=positive, render_width_ratio=positive)
def test_channel_subsampling(
    subsampling: int,
    trigger_width_ratio: int,
    render_width_ratio: int,
):
    """ Ensure nsamp and trigger/render subsampling are computed correctly. """

    ccfg = ChannelConfig(
        'tests/sine440.wav',
        trigger_width_ratio=trigger_width_ratio,
        render_width_ratio=render_width_ratio,
    )
    cfg = default_config(subsampling=subsampling)
    channel = Channel(ccfg, cfg)
    assert channel.wave.smp_s == 48000

    ideal_nsamp = round(cfg.render_width_s * channel.wave.smp_s / subsampling)
    assert channel.nsamp == pytest.approx(ideal_nsamp, 1)
    assert channel.trigger_subsampling == subsampling * trigger_width_ratio
    assert channel.render_subsampling == subsampling * render_width_ratio



# line_color is tested in test_renderer.py
# todo test ChannelConfig.ampl_ratio
