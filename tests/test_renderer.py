from typing import Optional

import numpy as np
import pytest
from matplotlib.colors import to_rgb

from corrscope.channel import ChannelConfig
from corrscope.layout import LayoutConfig
from corrscope.outputs import RGB_DEPTH
from corrscope.renderer import RendererConfig, MatplotlibRenderer

WIDTH = 640
HEIGHT = 360

ALL_ZEROS = np.array([0, 0])

all_colors = pytest.mark.parametrize(
    "bg_str,fg_str,grid_str",
    [
        ("#000000", "#ffffff", None),
        ("#ffffff", "#000000", None),
        ("#0000aa", "#aaaa00", None),
        ("#aaaa00", "#0000aa", None),
        # Enabling gridlines enables Axes rectangles.
        # Make sure they don't draw *over* the global figure background.
        ("#0000aa", "#aaaa00", "#ff00ff"),  # beautiful magenta gridlines
        ("#aaaa00", "#0000aa", "#ff00ff"),
    ],
)

nplots = 2


@all_colors
def test_default_colors(bg_str, fg_str, grid_str):
    """ Test the default background/foreground colors. """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color=fg_str,
        grid_color=grid_str,
        line_width=2.0,
        antialiasing=False,
    )
    lcfg = LayoutConfig()

    r = MatplotlibRenderer(cfg, lcfg, nplots, None)
    verify(r, bg_str, fg_str, grid_str)

    # Ensure default ChannelConfig(line_color=None) does not override line color
    chan = ChannelConfig(wav_path="")
    channels = [chan] * nplots
    r = MatplotlibRenderer(cfg, lcfg, nplots, channels)
    verify(r, bg_str, fg_str, grid_str)


@all_colors
def test_line_colors(bg_str, fg_str, grid_str):
    """ Test channel-specific line color overrides """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color="#888888",
        grid_color=grid_str,
        line_width=2.0,
        antialiasing=False,
    )
    lcfg = LayoutConfig()

    chan = ChannelConfig(wav_path="", line_color=fg_str)
    channels = [chan] * nplots
    r = MatplotlibRenderer(cfg, lcfg, nplots, channels)
    verify(r, bg_str, fg_str, grid_str)


def verify(r: MatplotlibRenderer, bg_str, fg_str, grid_str: Optional[str]):
    r.render_frame([ALL_ZEROS] * nplots)
    frame_colors: np.ndarray = np.frombuffer(r.get_frame(), dtype=np.uint8).reshape(
        (-1, RGB_DEPTH)
    )

    bg_u8 = [round(c * 255) for c in to_rgb(bg_str)]
    fg_u8 = [round(c * 255) for c in to_rgb(fg_str)]
    all_colors = [bg_u8, fg_u8]

    if grid_str:
        grid_u8 = [round(c * 255) for c in to_rgb(grid_str)]
        all_colors.append(grid_u8)

    # Ensure background is correct
    bg_frame = frame_colors[0]
    assert (
        bg_frame == bg_u8
    ).all(), f"incorrect background, it might be grid_str={grid_str}"

    # Ensure foreground is present
    assert np.prod(
        frame_colors == fg_u8, axis=-1
    ).any(), "incorrect foreground, it might be 136 = #888888"

    # Ensure grid color is present
    if grid_str:
        assert np.prod(frame_colors == grid_u8, axis=-1).any(), "Missing grid_str"

    assert (np.amax(frame_colors, axis=0) == np.amax(all_colors, axis=0)).all()
    assert (np.amin(frame_colors, axis=0) == np.amin(all_colors, axis=0)).all()
