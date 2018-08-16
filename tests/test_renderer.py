from unittest.mock import patch

import pytest

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

# TODO: test get_frame()
def test_colors():
    pass    # TODO


# TODO (integration test) ensure rendering to output works
