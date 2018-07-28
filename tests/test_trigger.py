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
        trigger_strength=1,
        use_edge_trigger=use_edge_trigger,

        responsiveness=1,
        falloff_width=2,
    )


def test_trigger(cfg: CorrelationTriggerConfig):
    # wave = Wave(None, 'tests/sine440.wav')
    wave = Wave(None, 'tests/impulse24000.wav')

    iters = 5
    plot = False
    x = 24000 - 500
    trigger = cfg(wave, 4000, subsampling=1)

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
            offset2 = trigger.get_trigger(x)
            print(offset2)
            assert offset2 == 24000
        if plot:
            ax.plot(trigger._buffer, label=str(i))
            ax.grid()

    if plot:
        plt.show()
