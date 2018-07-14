import matplotlib.pyplot as plt

from ovgenpy import triggers
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ovgenpy.triggers import CorrelationTrigger
from ovgenpy.wave import Wave


triggers.SHOW_TRIGGER = False

cfg = CorrelationTrigger.Config(
    trigger_strength=1,
    responsiveness=1,
    falloff_width=2,
)


def test_trigger():
    wave = Wave(None, 'tests/sine440.wav')
    trigger = cfg(wave, 4000)

    BIG = 0.95
    SMALL = 0.05
    fig, axes = plt.subplots(5, gridspec_kw=dict(
        top=BIG, right=BIG,
        bottom=SMALL, left=SMALL,
    ))    # type: Figure, Axes
    fig.tight_layout()

    for i, ax in enumerate(axes):
        if i:
            print(trigger.get_trigger(4000))
        ax.plot(trigger._buffer, label=str(i))
        ax.grid()

    plt.show()
