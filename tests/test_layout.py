from typing import List

import numpy as np
import pytest

from corrscope.layout import LayoutConfig, RendererLayout, RegionSpec
from corrscope.renderer import RendererConfig, MatplotlibRenderer
from tests.test_renderer import WIDTH, HEIGHT, RENDER_Y_ZEROS


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
    assert default.orientation == "h"


@pytest.mark.parametrize("lcfg", [LayoutConfig(ncols=2), LayoutConfig(nrows=8)])
def test_hlayout(lcfg):
    nplots = 15
    layout = RendererLayout(lcfg, nplots, [1] * nplots)

    assert layout.wave_ncol == 2
    assert layout.wave_nrow == 8

    region2d: List[List[RegionSpec]] = layout.arrange(lambda arg: arg)
    assert len(region2d) == nplots
    for i, regions in enumerate(region2d):
        assert len(regions) == 1, (i, len(regions))

    np.testing.assert_equal(region2d[0][0].pos, (0, 0))
    np.testing.assert_equal(region2d[1][0].pos, (0, 1))
    np.testing.assert_equal(region2d[2][0].pos, (1, 0))

    m = nplots - 1
    np.testing.assert_equal(region2d[m][0].pos, (m // 2, m % 2))


@pytest.mark.parametrize(
    "lcfg",
    [LayoutConfig(ncols=3, orientation="v"), LayoutConfig(nrows=3, orientation="v")],
)
def test_vlayout(lcfg):
    nplots = 7
    layout = RendererLayout(lcfg, nplots, [1] * nplots)

    assert layout.wave_ncol == 3
    assert layout.wave_nrow == 3

    region2d: List[List[RegionSpec]] = layout.arrange(lambda arg: arg)
    assert len(region2d) == nplots
    for i, regions in enumerate(region2d):
        assert len(regions) == 1, (i, len(regions))

    np.testing.assert_equal(region2d[0][0].pos, (0, 0))
    np.testing.assert_equal(region2d[2][0].pos, (2, 0))
    np.testing.assert_equal(region2d[3][0].pos, (0, 1))
    np.testing.assert_equal(region2d[6][0].pos, (0, 2))


# FIXME test stereo layouts (h/v)


def test_renderer_layout():
    # 2 columns
    cfg = RendererConfig(WIDTH, HEIGHT)
    lcfg = LayoutConfig(ncols=2)
    nplots = 15

    r = MatplotlibRenderer(cfg, lcfg, nplots, None)
    r.render_frame([RENDER_Y_ZEROS] * nplots)
    layout = r.layout

    # 2 columns, 8 rows
    assert layout.wave_ncol == 2
    assert layout.wave_nrow == 8
