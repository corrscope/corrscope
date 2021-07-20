from typing import List

import attr
import matplotlib as mpl
import matplotlib.colors
import numpy as np

TWOPI = 2 * np.pi


@attr.dataclass
class Palette:
    unclipped: np.ndarray
    clipped: List[str]


def gen_circular_palette() -> Palette:
    """Return a palette of 12 *distinct* colors in #rrggbb format.
    The last color is *not* the same as the first.
    """
    from colorspacious import cspace_convert

    # Based on https://youtu.be/xAoljeRJ3lU?t=887
    N = 12

    VALUE = 85
    SATURATION = 30

    # Phase of color, chosen so output[0] is red
    HUE = 0.2
    DIRECTION = -1

    # constant lightness
    Jp = VALUE * np.ones(N)

    # constant saturation, varying hue
    t = DIRECTION * np.linspace(0, 1, N, endpoint=False) + HUE

    ap = SATURATION * np.sin(TWOPI * t)
    bp = SATURATION * np.cos(TWOPI * t)

    # [N](Jp, ap, bp) real
    Jp_ap_bp = np.column_stack((Jp, ap, bp))

    rgb_raw = cspace_convert(Jp_ap_bp, "CAM02-UCS", "sRGB1")

    rgb = np.clip(rgb_raw, 0, None)
    rgb_max = np.max(rgb, axis=1).reshape(-1, 1)
    rgb /= rgb_max
    assert ((0 <= rgb) * (rgb <= 1)).all()

    print(f"Peak overflow = {np.max(rgb_max - 1)}")
    print(f"Peak underflow = {np.min(rgb_raw - rgb)}")

    rgb = [mpl.colors.to_hex(c) for c in rgb]
    print(repr(rgb))
    return Palette(rgb_raw, rgb)


if False:
    spectral_colors = gen_circular_palette().clipped
else:
    spectral_colors = [
        "#ff8189",
        "#ff9155",
        "#ffba37",
        "#f7ff52",
        "#95ff85",
        "#16ffc1",
        "#00ffff",
        "#4dccff",
        "#86acff",
        "#b599ff",
        "#ed96ff",
        "#ff87ca",
    ]
