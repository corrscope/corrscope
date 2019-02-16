from typing import List

import hypothesis.strategies as hs
import numpy as np
import pytest
from hypothesis import given, settings

from corrscope.layout import LayoutConfig, RendererLayout, RegionSpec, Edges
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


# Small range to ensure many collisions.
# max_value = 3 to allow for edge-free space in center.
integers = hs.integers(-1, 3)


@given(nrows=integers, ncols=integers, row=integers, col=integers)
@settings(max_examples=500)
def test_edges(nrows: int, ncols: int, row: int, col: int):
    if not (nrows > 0 and ncols > 0 and 0 <= row < nrows and 0 <= col < ncols):
        with pytest.raises(ValueError):
            edges = Edges.at(nrows, ncols, row, col)
        return

    edges = Edges.at(nrows, ncols, row, col)
    assert bool(edges & Edges.Left) == (col == 0)
    assert bool(edges & Edges.Right) == (col == ncols - 1)
    assert bool(edges & Edges.Top) == (row == 0)
    assert bool(edges & Edges.Bottom) == (row == nrows - 1)


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
