import pytest

from corrscope.layout import LayoutConfig, RendererLayout
from corrscope.renderer import RendererConfig, MatplotlibRenderer
from tests.test_renderer import WIDTH, HEIGHT


def test_layout_config():
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


def test_renderer_layout():
    # 2 columns
    cfg = RendererConfig(WIDTH, HEIGHT)
    lcfg = LayoutConfig(ncols=2)
    nplots = 15

    r = MatplotlibRenderer(cfg, lcfg, nplots, None)

    # 2 columns, 8 rows
    assert r.layout.ncols == 2
    assert r.layout.nrows == 8
