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
    LocalPostTriggerConfig,
    SpectrumConfig,
)
from corrscope.wave import Wave

triggers.SHOW_TRIGGER = False


def cfg_template(**kwargs) -> CorrelationTriggerConfig:
    """ Not identical to default_config() template. """
    cfg = CorrelationTriggerConfig(
        edge_strength=2, responsiveness=1, buffer_falloff=0.5, use_edge_trigger=False
    )
    return attr.evolve(cfg, **kwargs)


@pytest_fixture_plus
@pytest.mark.parametrize("use_edge_trigger", [False, True])
@pytest.mark.parametrize("trigger_diameter", [None, 0.5])
@pytest.mark.parametrize("pitch_tracking", [None, SpectrumConfig()])
def cfg(use_edge_trigger, trigger_diameter, pitch_tracking):
    return cfg_template(
        use_edge_trigger=use_edge_trigger,
        trigger_diameter=trigger_diameter,
        pitch_tracking=pitch_tracking,
    )


@pytest.fixture(
    scope="session",
    params=[None, ZeroCrossingTriggerConfig(), LocalPostTriggerConfig(strength=1)],
)
def post_cfg(request):
    post = request.param
    return cfg_template(post=post)


# I regret adding the nsamp_frame parameter. It makes unit tests hard.

FPS = 60


def test_trigger(cfg: CorrelationTriggerConfig):
    wave = Wave("tests/impulse24000.wav")

    iters = 5
    plot = False
    x0 = 24000
    x = x0 - 500
    trigger: CorrelationTrigger = cfg(wave, 4000, stride=1, fps=FPS)

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


def test_trigger_stride(cfg: CorrelationTriggerConfig):
    wave = Wave("tests/sine440.wav")
    # period = 48000 / 440 = 109.(09)*

    iters = 5
    x0 = 24000
    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)
    # real window_samp = window_samp*stride
    # period = 109

    cache = PerFrameCache()

    for i in range(1, iters):
        offset = trigger.get_trigger(x0, cache)

        # Debugging CorrelationTrigger.get_trigger:
        # from matplotlib import pyplot as plt
        # plt.plot(data)
        # plt.plot(prev_buffer)
        # plt.plot(corr)

        # When i=0, the data has 3 peaks, the rightmost taller than the center. The
        # *tips* of the outer peaks are truncated between `left` and `right`.
        # After truncation, corr[mid+1] is almost identical to corr[mid], for
        # reasons I don't understand (mid+1 > mid because dithering?).
        if not cfg.use_edge_trigger:
            assert (offset - x0) % stride == 0, f"iteration {i}"
            assert abs(offset - x0) < 10, f"iteration {i}"

        # The edge trigger activates at x0+1=24001. Likely related: it triggers
        # when moving from <=0 to >0. This is a necessary evil, in order to
        # recognize 0-to-positive edges while testing tests/impulse24000.wav .

        else:
            # If assertion fails, remove it.
            assert (offset - x0) % stride != 0, f"iteration {i}"
            assert abs(offset - x0) <= 2, f"iteration {i}"


def test_post_trigger_stride(post_cfg: CorrelationTriggerConfig):
    cfg = post_cfg

    wave = Wave("tests/sine440.wav")
    iters = 5
    x0 = 24000
    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)

    cache = PerFrameCache()
    for i in range(1, iters):
        offset = trigger.get_trigger(x0, cache)

        if not cfg.post:
            assert (offset - x0) % stride == 0, f"iteration {i}"
            assert abs(offset - x0) < 10, f"iteration {i}"

        else:
            # If assertion fails, remove it.
            assert (offset - x0) % stride != 0, f"iteration {i}"
            assert abs(offset - x0) <= 2, f"iteration {i}"


def test_trigger_stride_edges(cfg: CorrelationTriggerConfig):
    wave = Wave("tests/sine440.wav")
    # period = 48000 / 440 = 109.(09)*

    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)
    # real window_samp = window_samp*stride
    # period = 109

    trigger.get_trigger(0, PerFrameCache())
    trigger.get_trigger(-1000, PerFrameCache())
    trigger.get_trigger(50000, PerFrameCache())


def test_trigger_should_recalc_window():
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


# Test pitch-invariant triggering using spectrum
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
  use_edge_trigger: false
  responsiveness: 0.2
  falloff_width: 2
"""
    )


# TODO test_period get_period()
