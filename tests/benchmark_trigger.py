import numpy as np
import timeit

from ovgenpy import triggers, wave as wave_m
from ovgenpy.triggers import CorrelationTrigger
from ovgenpy.wave import Wave


triggers.SHOW_TRIGGER = False


cfg = CorrelationTrigger.Config(
    trigger_strength=1,
    use_edge_trigger=True,

    responsiveness=1,
    falloff_width=2,
)


def main():
    for float_t in [np.single, np.double] * 3:
        wave_m.FLOAT = float_t
        trigger()


def trigger():
    print(wave_m.FLOAT)
    wave = Wave(None, 'sine440.wav')

    x = 24000 - 500
    trigger = cfg(wave, scan_nsamp=4000)

    def test() -> None:
        trigger.get_trigger(x)

    N = 10**4
    t = timeit.timeit(test, number=N) / N
    print('%E' % t)
    print()


if __name__ == '__main__':
    main()
