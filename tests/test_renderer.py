import numpy as np
import pytest
from matplotlib.colors import to_rgb

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
def test_hlayout(lcfg):
    nplots = 15
    layout = RendererLayout(lcfg, nplots)

    assert layout.ncols == 2
    assert layout.nrows == 8

    # holy shit, passing tuples into a numpy array breaks things spectacularly, and it's
    # painfully difficult to stuff tuples into 1D array.
    # http://wesmckinney.com/blog/performance-quirk-making-a-1d-object-ndarray-of-tuples/
    regions = layout.arrange(lambda row, col: str((row, col)))
    assert len(regions) == nplots

    assert regions[0] == '(0, 0)'
    assert regions[1] == '(0, 1)'
    assert regions[2] == '(1, 0)'
    m = nplots - 1
    assert regions[m] == str((m // 2, m % 2))


@pytest.mark.parametrize('lcfg', [
    LayoutConfig(ncols=3, orientation='v'),
    LayoutConfig(nrows=3, orientation='v'),
])
def test_vlayout(lcfg):
    nplots = 7
    layout = RendererLayout(lcfg, nplots)

    assert layout.ncols == 3
    assert layout.nrows == 3

    regions = layout.arrange(lambda row, col: str((row, col)))
    assert len(regions) == nplots

    assert regions[0] == '(0, 0)'
    assert regions[2] == '(2, 0)'
    assert regions[3] == '(0, 1)'
    assert regions[6] == '(0, 2)'


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

@pytest.mark.parametrize('bg_str,fg_str', [
    ('#000000', '#ffffff'),
    ('#ffffff', '#000000'),
    ('#0000aa', '#aaaa00'),
    ('#aaaa00', '#0000aa'),
])
def test_colors(bg_str, fg_str):
    """ Ensure the rendered background/foreground colors are correct. """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color=fg_str,
        # line_width=5,
    )
    lcfg = LayoutConfig()
    nplots = 1

    r = MatplotlibRenderer(cfg, lcfg, nplots)
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


# TODO (integration test) ensure rendering to output works


def test_render_output():
    """ Ensure rendering to output does not raise exceptions. """

    from ovgenpy.ovgenpy import default_config
    from ovgenpy.outputs import FFmpegOutput, FFmpegOutputConfig

    cfg = default_config(render=RendererConfig(WIDTH, HEIGHT))
    renderer = MatplotlibRenderer(cfg.render, cfg.layout, nplots=1)
    output_cfg = FFmpegOutputConfig('-', '-f nut')
    out = FFmpegOutput(cfg, output_cfg)

    renderer.render_frame([ALL_ZEROS])
    out.write_frame(renderer.get_frame())

    assert out.close() == 0
