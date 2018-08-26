import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ovgenpy import triggers
from ovgenpy.triggers import CorrelationTriggerConfig
from ovgenpy.wave import Wave

triggers.SHOW_TRIGGER = False


@pytest.fixture(scope='session', params=[False, True])
def cfg(request):
    use_edge_trigger = request.param
    return CorrelationTriggerConfig(
        use_edge_trigger=use_edge_trigger,
        responsiveness=1,
    )


# I regret adding the nsamp_frame parameter. It makes unit tests hard.

FPS = 60

def test_trigger(cfg: CorrelationTriggerConfig):
    wave = Wave(None, 'tests/impulse24000.wav')

    iters = 5
    plot = False
    x0 = 24000
    x = x0 - 500
    trigger = cfg(wave, 4000, subsampling=1, fps=FPS)

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
            offset = trigger.get_trigger(x)
            print(offset)
            assert offset == x0
        if plot:
            ax.plot(trigger._buffer, label=str(i))
            ax.grid()

    if plot:
        plt.show()


def test_trigger_subsampling(cfg: CorrelationTriggerConfig):
    wave = Wave(None, 'tests/sine440.wav')
    # period = 48000 / 440 = 109.(09)*

    iters = 5
    x0 = 24000
    subsampling = 4
    trigger = cfg(wave, tsamp=100, subsampling=subsampling, fps=FPS)
    # real window_samp = window_samp*subsampling
    # period = 109

    for i in range(1, iters):
        offset = trigger.get_trigger(x0)
        print(offset)

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
            assert (offset - x0) % subsampling == 0
            assert abs(offset - x0) < 10

        # The edge trigger activates at x0+1=24001. Likely related: it triggers
        # when moving from <=0 to >0. This is a necessary evil, in order to
        # recognize 0-to-positive edges while testing tests/impulse24000.wav .

        else:
            assert abs(offset - x0) <= 2


def test_trigger_subsampling_edges(cfg: CorrelationTriggerConfig):
    wave = Wave(None, 'tests/sine440.wav')
    # period = 48000 / 440 = 109.(09)*

    iters = 5
    subsampling = 4
    trigger = cfg(wave, tsamp=100, subsampling=subsampling, fps=FPS)
    # real window_samp = window_samp*subsampling
    # period = 109

    trigger.get_trigger(0)
    trigger.get_trigger(-1000)
    trigger.get_trigger(50000)


# TODO test_period get_period()
