import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ovgenpy import triggers
from ovgenpy.triggers import CorrelationTriggerConfig, CorrelationTrigger, \
    PerFrameCache, ZeroCrossingTriggerConfig, LocalPostTriggerConfig
from ovgenpy.wave import Wave

triggers.SHOW_TRIGGER = False


@pytest.fixture(scope='session', params=[False, True])
def cfg(request):
    use_edge_trigger = request.param
    return CorrelationTriggerConfig(
        use_edge_trigger=use_edge_trigger,
        responsiveness=1,
    )


@pytest.fixture(scope='session', params=[
    None,
    ZeroCrossingTriggerConfig(),
    LocalPostTriggerConfig(strength=1)
])
def post_cfg(request):
    post = request.param
    return CorrelationTriggerConfig(
        use_edge_trigger=False,
        responsiveness=1,
        post=post
    )


# I regret adding the nsamp_frame parameter. It makes unit tests hard.

FPS = 60

def test_trigger(cfg: CorrelationTriggerConfig):
    wave = Wave(None, 'tests/impulse24000.wav')

    iters = 5
    plot = False
    x0 = 24000
    x = x0 - 500
    trigger: CorrelationTrigger = cfg(wave, 4000, stride=1, fps=FPS)

    if plot:
        BIG = 0.95
        SMALL = 0.05
        fig, axes = plt.subplots(iters, gridspec_kw=dict(
            top=BIG, right=BIG,
            bottom=SMALL, left=SMALL,
        ))    # type: Figure, Axes
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
    wave = Wave(None, 'tests/sine440.wav')
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
            assert (offset - x0) % stride == 0, f'iteration {i}'
            assert abs(offset - x0) < 10, f'iteration {i}'

        # The edge trigger activates at x0+1=24001. Likely related: it triggers
        # when moving from <=0 to >0. This is a necessary evil, in order to
        # recognize 0-to-positive edges while testing tests/impulse24000.wav .

        else:
            # If assertion fails, remove it.
            assert (offset - x0) % stride != 0, f'iteration {i}'
            assert abs(offset - x0) <= 2, f'iteration {i}'


def test_post_trigger_stride(post_cfg: CorrelationTriggerConfig):
    cfg = post_cfg

    wave = Wave(None, 'tests/sine440.wav')
    iters = 5
    x0 = 24000
    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)

    cache = PerFrameCache()
    for i in range(1, iters):
        offset = trigger.get_trigger(x0, cache)

        if not cfg.post:
            assert (offset - x0) % stride == 0, f'iteration {i}'
            assert abs(offset - x0) < 10, f'iteration {i}'

        else:
            # If assertion fails, remove it.
            assert (offset - x0) % stride != 0, f'iteration {i}'
            assert abs(offset - x0) <= 2, f'iteration {i}'


def test_trigger_stride_edges(cfg: CorrelationTriggerConfig):
    wave = Wave(None, 'tests/sine440.wav')
    # period = 48000 / 440 = 109.(09)*

    stride = 4
    trigger = cfg(wave, tsamp=100, stride=stride, fps=FPS)
    # real window_samp = window_samp*stride
    # period = 109

    trigger.get_trigger(0, PerFrameCache())
    trigger.get_trigger(-1000, PerFrameCache())
    trigger.get_trigger(50000, PerFrameCache())


def test_trigger_should_recalc_window():
    cfg = CorrelationTriggerConfig(recalc_semitones=1.0)
    wave = Wave(None, 'tests/sine440.wav')
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


# Test the ability to load legacy TriggerConfig

def test_load_trigger_config():
    from ovgenpy.config import yaml

    # Ensure no exceptions
    yaml.load('''\
!CorrelationTriggerConfig
  trigger_strength: 3
  use_edge_trigger: false
  responsiveness: 0.2
  falloff_width: 2
''')

# TODO test_period get_period()
