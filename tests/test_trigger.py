import attr
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pytest_cases import pytest_fixture_plus

from corrscope import triggers
from corrscope.triggers import (
    CorrelationTriggerConfig,
    CorrelationTrigger,
    PerFrameCache,
    ZeroCrossingTriggerConfig,
    SpectrumConfig,
)
from corrscope.wave import Wave

parametrize = pytest.mark.parametrize


triggers.SHOW_TRIGGER = False


def cfg_template(**kwargs) -> CorrelationTriggerConfig:
    """ Not identical to default_config() template. """
    cfg = CorrelationTriggerConfig(
        edge_strength=2, responsiveness=1, buffer_falloff=0.5
    )
    return attr.evolve(cfg, **kwargs)


@pytest_fixture_plus
@parametrize("trigger_diameter", [None, 0.5])
@parametrize("pitch_tracking", [None, SpectrumConfig()])
def cfg(trigger_diameter, pitch_tracking):
    return cfg_template(
        trigger_diameter=trigger_diameter, pitch_tracking=pitch_tracking
    )


FPS = 60

is_odd = parametrize("is_odd", [False, True])


# CorrelationTrigger overall tests


@is_odd
def test_trigger(cfg: CorrelationTriggerConfig, is_odd: bool):
    """Ensures that trigger can locate
    the first positive sample of a -+ step exactly,
    without off-by-1 errors."""
    wave = Wave("tests/step2400.wav")

    iters = 5
    plot = False
    x0 = 2400
    x = x0 - 50
    trigger: CorrelationTrigger = cfg(wave, 400 + int(is_odd), stride=1, fps=FPS)

    if plot:
        BIG = 0.95
        SMALL = 0.05
        fig, axes = plt.subplots(
            iters, gridspec_kw=dict(top=BIG, right=BIG, bottom=SMALL, left=SMALL)
        )  # type: Figure, Axes
        fig.tight_layout()
    else:
        axes = range(iters)

    for i, ax in enumerate(axes):
        if i:
            offset = trigger.get_trigger(x, PerFrameCache())
            print(offset)
            assert offset == x0
        if plot:
            ax.plot(trigger._buffer, label=str(i))
            ax.grid()

    if plot:
        plt.show()


@parametrize("post_trigger", [None, ZeroCrossingTriggerConfig()])
def test_post_stride(post_trigger):
    """
    Test that stride is respected when post_trigger is disabled,
    and ignored when post_trigger is enabled.
    """
    cfg = cfg_template(post_trigger=post_trigger)

    wave = Wave("tests/sine440.wav")
    iters = 5
    x0 = 24000
    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)

    cache = PerFrameCache()
    for i in range(1, iters):
        offset = trigger.get_trigger(x0, cache)

        if not cfg.post_trigger:
            assert (offset - x0) % stride == 0, f"iteration {i}"
            assert abs(offset - x0) < 10, f"iteration {i}"

        else:
            # If assertion fails, remove it.
            assert (offset - x0) % stride != 0, f"iteration {i}"
            assert abs(offset - x0) <= 2, f"iteration {i}"


@parametrize("post_trigger", [None, ZeroCrossingTriggerConfig()])
@parametrize("double_negate", [False, True])
def test_trigger_direction(post_trigger, double_negate):
    """
    Right now, MainTrigger is responsible for negating wave.amplification
    if edge_direction == -1.
    And triggers should not actually access edge_direction.
    """

    index = 2400
    wave = Wave("tests/step2400.wav")

    if double_negate:
        wave.amplification = -1
        cfg = cfg_template(post_trigger=post_trigger, edge_direction=-1)
    else:
        cfg = cfg_template(post_trigger=post_trigger)

    trigger = cfg(wave, 100, 1, FPS)
    cfg.edge_direction = None
    assert trigger._wave.amplification == 1

    cache = PerFrameCache()
    for dx in [-10, 10, 0]:
        assert trigger.get_trigger(index + dx, cache) == index


def test_trigger_out_of_bounds(cfg: CorrelationTriggerConfig):
    """Ensure out-of-bounds triggering with stride does not crash.
    (why does stride matter? IDK.)"""
    wave = Wave("tests/sine440.wav")
    # period = 48000 / 440 = 109.(09)*

    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)
    # real window_samp = window_samp*stride
    # period = 109

    trigger.get_trigger(0, PerFrameCache())
    trigger.get_trigger(-1000, PerFrameCache())
    trigger.get_trigger(50000, PerFrameCache())


def test_when_does_trigger_recalc_window():
    cfg = cfg_template(recalc_semitones=1.0)
    wave = Wave("tests/sine440.wav")
    trigger: CorrelationTrigger = cfg(wave, tsamp=1000, stride=1, fps=FPS)

    for x in [0, 1, 1000]:
        assert trigger._is_window_invalid(x), x

    trigger._prev_period = 100

    for x in [99, 101]:
        assert not trigger._is_window_invalid(x), x
    for x in [0, 80, 120]:
        assert trigger._is_window_invalid(x), x

    trigger._prev_period = 0

    x = 0
    assert not trigger._is_window_invalid(x), x
    for x in [1, 100]:
        assert trigger._is_window_invalid(x), x


# Test pitch-tracking (spectrum)


def test_correlate_offset():
    """
    Catches bug where writing N instead of Ncorr
    prevented function from returning positive numbers.
    """

    np.random.seed(31337)
    correlate_offset = CorrelationTrigger.correlate_offset

    # Ensure autocorrelation on random data returns peak at 0.
    N = 100
    spectrum = np.random.random(N)
    assert correlate_offset(spectrum, spectrum, 12) == 0

    # Ensure cross-correlation of time-shifted impulses works.
    # Assume wave where y=[i==99].
    wave = np.eye(N)[::-1]
    # Taking a slice beginning at index i will produce an impulse at 99-i.
    left = wave[30]
    right = wave[40]

    # We need to slide `left` to the right by 10 samples, and vice versa.
    for radius in [None, 12]:
        assert correlate_offset(data=left, prev_buffer=right, radius=radius) == 10
        assert correlate_offset(data=right, prev_buffer=left, radius=radius) == -10

    # The correlation peak at zero-offset is small enough for boost_x to be returned.
    boost_y = 1.5
    ones = np.ones(N)
    for boost_x in [6, -6]:
        assert (
            correlate_offset(ones, ones, radius=9, boost_x=boost_x, boost_y=boost_y)
            == boost_x
        )


# Test the ability to load legacy TriggerConfig


def test_load_trigger_config():
    from corrscope.config import yaml

    # Ensure no exceptions
    yaml.load(
        """\
!CorrelationTriggerConfig
  trigger_strength: 3
  responsiveness: 0.2
  falloff_width: 2
"""
    )


# TODO test_period get_period()
