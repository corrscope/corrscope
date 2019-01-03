import numpy as np
import pytest
import matplotlib.colors

from corrscope.channel import ChannelConfig
from corrscope.outputs import RGB_DEPTH
from corrscope.renderer import RendererConfig, MatplotlibRenderer
from corrscope.layout import LayoutConfig

WIDTH = 640
HEIGHT = 360

ALL_ZEROS = np.array([0, 0])

all_colors = pytest.mark.parametrize(
    "bg_str,fg_str",
    [
        ("#000000", "#ffffff"),
        ("#ffffff", "#000000"),
        ("#0000aa", "#aaaa00"),
        ("#aaaa00", "#0000aa"),
    ],
)


@all_colors
def test_default_colors(bg_str, fg_str):
    """ Test the default background/foreground colors. """
    cfg = RendererConfig(WIDTH, HEIGHT, bg_color=bg_str, init_line_color=fg_str)
    lcfg = LayoutConfig()
    nplots = 1

    r = MatplotlibRenderer(cfg, lcfg, nplots, None)
    verify(r, bg_str, fg_str)

    # Ensure default ChannelConfig(line_color=None) does not override line color
    chan = ChannelConfig(wav_path="")
    channels = [chan] * nplots
    r = MatplotlibRenderer(cfg, lcfg, nplots, channels)
    verify(r, bg_str, fg_str)


@all_colors
def test_line_colors(bg_str, fg_str):
    """ Test channel-specific line color overrides """
    cfg = RendererConfig(WIDTH, HEIGHT, bg_color=bg_str, init_line_color="#888888")
    lcfg = LayoutConfig()
    nplots = 1

    chan = ChannelConfig(wav_path="", line_color=fg_str)
    channels = [chan] * nplots
    r = MatplotlibRenderer(cfg, lcfg, nplots, channels)
    verify(r, bg_str, fg_str)


def verify(r: MatplotlibRenderer, bg_str, fg_str):
    r.render_frame([ALL_ZEROS])
    frame_colors: np.ndarray = np.frombuffer(r.get_frame(), dtype=np.uint8).reshape(
        (-1, RGB_DEPTH)
    )

    bg_u8 = to_bgra(bg_str)
    fg_u8 = to_bgra(fg_str)

    # Ensure background is correct
    np.testing.assert_equal(frame_colors[0], bg_u8)

    # Ensure foreground is present
    assert np.prod(
        frame_colors == fg_u8, axis=-1
    ).any(), "incorrect foreground, it might be 136 = #888888"

    np.testing.assert_equal(np.amax(frame_colors, axis=0), np.maximum(bg_u8, fg_u8))
    np.testing.assert_equal(np.amin(frame_colors, axis=0), np.minimum(bg_u8, fg_u8))


def to_bgra(c):
    # https://github.com/anntzer/mplcairo/issues/12#issuecomment-451035973
    # Cairo outputs bgrA.
    # ffmpeg bgr0 and rgb32 are equivalent, but rgb32 uses less CPU in ffplay.
    rgb = [round(c * 255) for c in matplotlib.colors.to_rgb(c)]
    bgra = rgb[::-1] + [255]
    return bgra
