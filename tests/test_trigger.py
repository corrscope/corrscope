import attr
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pytest_mock
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Pycharm assumes anything called "fixture" is pytest.fixture.
from pytest_cases import pytest_fixture_plus as fixture

from corrscope import triggers
from corrscope.triggers import (
    CorrelationTriggerConfig,
    CorrelationTrigger,
    PerFrameCache,
    ZeroCrossingTriggerConfig,
    SpectrumConfig,
    correlate_data,
    correlate_spectrum,
)
from corrscope.wave import Wave

parametrize = pytest.mark.parametrize


triggers.SHOW_TRIGGER = False


def trigger_template(**kwargs) -> CorrelationTriggerConfig:
    cfg = CorrelationTriggerConfig(
        edge_strength=2, responsiveness=1, buffer_falloff=0.5
    )
    return attr.evolve(cfg, **kwargs)


@fixture
@parametrize("trigger_diameter", [None, 0.5])
@parametrize("pitch_tracking", [None, SpectrumConfig()])
@parametrize("slope_strength", [0, 100])
@parametrize("sign_strength", [0, 1])
def trigger_cfg(
    trigger_diameter, pitch_tracking, slope_strength, sign_strength
) -> CorrelationTriggerConfig:
    return trigger_template(
        trigger_diameter=trigger_diameter,
        pitch_tracking=pitch_tracking,
        slope_strength=slope_strength,
        sign_strength=sign_strength,
        slope_width=0.14,
    )


FPS = 60

is_odd = parametrize("is_odd", [False, True])


# CorrelationTrigger overall tests


@is_odd
@parametrize("post_trigger", [None, ZeroCrossingTriggerConfig()])
def test_trigger(trigger_cfg, is_odd: bool, post_trigger):
    """Ensures that trigger can locate
    the first positive sample of a -+ step exactly,
    without off-by-1 errors.

    See CorrelationTrigger and Wave.get_around() docstrings.
    """
    wave = Wave("tests/step2400.wav")
    trigger_cfg = attr.evolve(trigger_cfg, post_trigger=post_trigger)

    iters = 5
    plot = False
    x0 = 2400
    x = x0 - 50
    trigger: CorrelationTrigger = trigger_cfg(
        wave, 400 + int(is_odd), stride=1, fps=FPS
    )

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
            offset = trigger.get_trigger(x, PerFrameCache()).result
            assert offset == x0, offset
        if plot:
            ax.plot(trigger._buffer, label=str(i))
            ax.grid()

    if plot:
        plt.show()


def test_mean_subtraction(trigger_cfg, mocker: "pytest_mock.MockFixture"):
    """
    Ensure that trigger subtracts mean properly in all configurations.
    -   Due to a regression, mean was not subtracted when sign_strength = 0.
        This caused get_period() to malfunction.
    """
    wave = Wave("tests/step2400.wav")

    get_period = mocker.spy(triggers, "get_period")
    trigger = trigger_cfg(wave, tsamp=100, stride=1, fps=FPS)
    cache = PerFrameCache()
    trigger.get_trigger(2600, cache)  # step2400.wav

    (data, *args), kwargs = get_period.call_args
    assert isinstance(data, np.ndarray)
    assert abs(np.mean(data)) < 0.01


@parametrize("post_trigger", [None, ZeroCrossingTriggerConfig()])
def test_post_stride(post_trigger):
    """
    Test that stride is respected when post_trigger is disabled,
    and ignored when post_trigger is enabled.
    """
    cfg = trigger_template(post_trigger=post_trigger)

    wave = Wave("tests/sine440.wav")
    iters = 5
    x0 = 24000
    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)

    cache = PerFrameCache()
    for i in range(1, iters):
        offset = trigger.get_trigger(x0, cache).result

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
        cfg = trigger_template(post_trigger=post_trigger, edge_direction=-1)
    else:
        cfg = trigger_template(post_trigger=post_trigger)

    trigger = cfg(wave, 100, 1, FPS)
    cfg.edge_direction = None
    assert trigger._wave.amplification == 1

    cache = PerFrameCache()
    for dx in [-10, 10, 0]:
        assert trigger.get_trigger(index + dx, cache).result == index


def test_trigger_out_of_bounds(trigger_cfg):
    """Ensure out-of-bounds triggering with stride does not crash.
    (why does stride matter? IDK.)"""
    wave = Wave("tests/sine440.wav")
    # period = 48000 / 440 = 109.(09)*

    stride = 4
    trigger = trigger_cfg(wave, tsamp=100, stride=stride, fps=FPS)
    # real window_samp = window_samp*stride
    # period = 109

    trigger.get_trigger(0, PerFrameCache())
    trigger.get_trigger(-1000, PerFrameCache())
    trigger.get_trigger(50000, PerFrameCache())


def test_when_does_trigger_recalc_window():
    cfg = trigger_template(recalc_semitones=1.0)
    wave = Wave("tests/sine440.wav")
    trigger: CorrelationTrigger = cfg(wave, tsamp=1000, stride=1, fps=FPS)

    for x in [0, 1, 1000]:
        assert trigger._is_window_invalid(x), x

    trigger._prev_period = 100

    for x in [0, 99, 101]:
        assert not trigger._is_window_invalid(x), x
    for x in [80, 120]:
        assert trigger._is_window_invalid(x), x

    trigger._prev_period = 0

    x = 0
    assert not trigger._is_window_invalid(x), x
    for x in [1, 100]:
        assert trigger._is_window_invalid(x), x


# Test post triggering by itself


def test_post_trigger_radius():
    """
    Ensure ZeroCrossingTrigger has no off-by-1 errors when locating edges,
    and slides at a fixed rate if no edge is found.
    """
    wave = Wave("tests/step2400.wav")
    center = 2400
    radius = 5

    cfg = ZeroCrossingTriggerConfig()
    post = cfg(wave, radius, 1, FPS)

    cache = PerFrameCache(mean=0)

    for offset in range(-radius, radius + 1):
        assert post.get_trigger(center + offset, cache) == center, offset

    for offset in [radius + 1, radius + 2, 100]:
        assert post.get_trigger(center - offset, cache) == center - offset + radius
        assert post.get_trigger(center + offset, cache) == center + offset - radius


# Test pitch-tracking (spectrum)


@parametrize("correlate", [correlate_data, correlate_spectrum])
def test_correlate_offset(correlate):
    """
    Catches bug where writing N instead of Ncorr
    prevented function from returning positive numbers.

    Right now, correlate_spectrum() is identical to correlate_data().
    """
    approx = lambda x: x

    np.random.seed(31337)

    # Ensure autocorrelation on random data returns peak at 0.
    N = 100
    spectrum = np.random.random(N)
    assert correlate(spectrum, spectrum, 12).peak == approx(0)

    # Ensure cross-correlation of time-shifted impulses works.
    # Assume wave where y=[i==99].
    wave = np.eye(N)[::-1]
    # Taking a slice beginning at index i will produce an impulse at 99-i.
    left = wave[30]
    right = wave[40]

    # We need to slide `left` to the right by 10 samples, and vice versa.
    for radius in [None, 12]:
        assert correlate(data=left, prev_buffer=right, radius=radius).peak == approx(10)
        assert correlate(data=right, prev_buffer=left, radius=radius).peak == approx(
            -10
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
