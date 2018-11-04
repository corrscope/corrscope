import numpy as np
import pytest
from matplotlib.colors import to_rgb

from ovgenpy.channel import ChannelConfig
from ovgenpy.outputs import RGB_DEPTH
from ovgenpy.renderer import RendererConfig, MatplotlibRenderer, LayoutConfig, \
    RendererLayout

WIDTH = 640
HEIGHT = 360


def test_config():
    with pytest.raises(ValueError):
        LayoutConfig(nrows=1, ncols=1)

    one_col = LayoutConfig(ncols=1)
    assert one_col

    one_row = LayoutConfig(nrows=1)
    assert one_row

    default = LayoutConfig()
    assert default.ncols == 1  # Should default to single-column layout
    assert default.nrows is None
    assert default.orientation == 'h'


@pytest.mark.parametrize('lcfg', [
    LayoutConfig(ncols=2),
    LayoutConfig(nrows=8),
])
@pytest.mark.parametrize('region_type', [str, tuple, list])
def test_hlayout(lcfg, region_type):
    nplots = 15
    layout = RendererLayout(lcfg, nplots)

    assert layout.ncols == 2
    assert layout.nrows == 8

    regions = layout.arrange(lambda row, col: region_type((row, col)))
    assert len(regions) == nplots

    assert regions[0] == region_type((0, 0))
    assert regions[1] == region_type((0, 1))
    assert regions[2] == region_type((1, 0))
    m = nplots - 1
    assert regions[m] == region_type((m // 2, m % 2))


@pytest.mark.parametrize('lcfg', [
    LayoutConfig(ncols=3, orientation='v'),
    LayoutConfig(nrows=3, orientation='v'),
])
@pytest.mark.parametrize('region_type', [str, tuple, list])
def test_vlayout(lcfg, region_type):
    nplots = 7
    layout = RendererLayout(lcfg, nplots)

    assert layout.ncols == 3
    assert layout.nrows == 3

    regions = layout.arrange(lambda row, col: region_type((row, col)))
    assert len(regions) == nplots

    assert regions[0] == region_type((0, 0))
    assert regions[2] == region_type((2, 0))
    assert regions[3] == region_type((0, 1))
    assert regions[6] == region_type((0, 2))


def test_renderer():
    """
    TODO check image output using:
    https://matplotlib.org/devel/testing.html#writing-an-image-comparison-test

    https://stackoverflow.com/a/27950953
    "[I]mage comparison tests end up bring more trouble than they are worth"
    """

    # 2 columns
    cfg = RendererConfig(WIDTH, HEIGHT)
    lcfg = LayoutConfig(ncols=2)
    nplots = 15

    r = MatplotlibRenderer(cfg, lcfg, nplots)

    # 2 columns, 8 rows
    assert r.layout.ncols == 2
    assert r.layout.nrows == 8


ALL_ZEROS = np.array([0,0])

all_colors = pytest.mark.parametrize('bg_str,fg_str', [
    ('#000000', '#ffffff'),
    ('#ffffff', '#000000'),
    ('#0000aa', '#aaaa00'),
    ('#aaaa00', '#0000aa'),
])


@all_colors
def test_default_colors(bg_str, fg_str):
    """ Test the default background/foreground colors. """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color=fg_str,
    )
    lcfg = LayoutConfig()
    nplots = 1

    r = MatplotlibRenderer(cfg, lcfg, nplots)
    verify(r, bg_str, fg_str)

    # Ensure default ChannelConfig(line_color=None) does not override line color
    r = MatplotlibRenderer(cfg, lcfg, nplots)
    chan = ChannelConfig(wav_path='')
    r.set_colors([chan] * nplots)
    verify(r, bg_str, fg_str)


@all_colors
def test_line_colors(bg_str, fg_str):
    """ Test channel-specific line color overrides """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color='#888888',
    )
    lcfg = LayoutConfig()
    nplots = 1

    r = MatplotlibRenderer(cfg, lcfg, nplots)
    chan = ChannelConfig(wav_path='', line_color=fg_str)
    r.set_colors([chan] * nplots)
    verify(r, bg_str, fg_str)


def verify(r: MatplotlibRenderer, bg_str, fg_str):
    r.render_frame([ALL_ZEROS])
    frame_colors: np.ndarray = \
        np.frombuffer(r.get_frame(), dtype=np.uint8).reshape((-1, RGB_DEPTH))

    bg_u8 = [round(c*255) for c in to_rgb(bg_str)]
    fg_u8 = [round(c*255) for c in to_rgb(fg_str)]

    # Ensure background is correct
    assert (frame_colors[0] == bg_u8).all()

    # Ensure foreground is present
    assert np.prod(frame_colors == fg_u8, axis=-1).any()

    assert (np.amax(frame_colors, axis=0) == np.maximum(bg_u8, fg_u8)).all()
    assert (np.amin(frame_colors, axis=0) == np.minimum(bg_u8, fg_u8)).all()
