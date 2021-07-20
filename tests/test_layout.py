from typing import List

import hypothesis.strategies as hs
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given, settings

from corrscope.layout import (
    LayoutConfig,
    RendererLayout,
    RegionSpec,
    Edges,
    Orientation,
    StereoOrientation,
)
from corrscope.renderer import RendererConfig, Renderer, RenderInput
from corrscope.util import ceildiv
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
    layout = RendererLayout(lcfg, [1] * nplots)

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
    npt.assert_equal(region2d[m][0].pos, (m // 2, m % 2))


@pytest.mark.parametrize(
    "lcfg",
    [LayoutConfig(ncols=3, orientation="v"), LayoutConfig(nrows=3, orientation="v")],
)
def test_vlayout(lcfg):
    nplots = 7
    layout = RendererLayout(lcfg, [1] * nplots)

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


@given(
    wave_nchans=hs.lists(hs.integers(1, 10), min_size=1, max_size=100),
    orientation=hs.sampled_from(Orientation),
    stereo_orientation=hs.sampled_from(StereoOrientation),
    nrow_ncol=hs.integers(1, 100),
    is_nrows=hs.booleans(),
)
def test_stereo_layout(
    orientation: Orientation,
    stereo_orientation: StereoOrientation,
    wave_nchans: List[int],
    nrow_ncol: int,
    is_nrows: bool,
):
    """
    Not-entirely-rigorous test for layout computation.
    Mind-numbingly boring to write (and read?).

    Honestly I prefer a good naming scheme in RendererLayout.arrange()
    over unit tests.

    - This is a regression test...
    - And an obstacle to refactoring or feature development.
    """
    # region Setup
    if is_nrows:
        nrows = nrow_ncol
        ncols = None
    else:
        nrows = None
        ncols = nrow_ncol

    lcfg = LayoutConfig(
        orientation=orientation,
        nrows=nrows,
        ncols=ncols,
        stereo_orientation=stereo_orientation,
    )
    nwaves = len(wave_nchans)
    layout = RendererLayout(lcfg, wave_nchans)
    # endregion

    # Assert layout dimensions correct
    assert layout.wave_ncol == ncols or ceildiv(nwaves, nrows)
    assert layout.wave_nrow == nrows or ceildiv(nwaves, ncols)

    region2d: List[List[RegionSpec]] = layout.arrange(lambda r_spec: r_spec)

    # Loop through layout regions
    assert len(region2d) == len(wave_nchans)
    for wave_i, wave_chans in enumerate(region2d):
        stereo_nchan = wave_nchans[wave_i]
        assert len(wave_chans) == stereo_nchan

        # Compute channel dims within wave.
        if stereo_orientation == StereoOrientation.overlay:
            chans_per_wave = [1, 1]
        elif stereo_orientation == StereoOrientation.v:  # pos[0]++
            chans_per_wave = [stereo_nchan, 1]
        else:
            assert stereo_orientation == StereoOrientation.h  # pos[1]++
            chans_per_wave = [1, stereo_nchan]

        # Sanity-check position of channel 0 relative to origin (wave grid).
        assert (np.add.reduce(wave_chans[0].pos) != 0) == (wave_i != 0)
        npt.assert_equal(wave_chans[0].pos % chans_per_wave, 0)

        for chan_j, chan in enumerate(wave_chans):
            # Assert 0 <= position < size.
            assert chan.pos.shape == chan.size.shape == (2,)
            assert (0 <= chan.pos).all()
            assert (chan.pos < chan.size).all()

            # Sanity-check position of chan relative to origin (wave grid).
            npt.assert_equal(
                chan.pos // chans_per_wave, wave_chans[0].pos // chans_per_wave
            )

            # Check position of region (relative to channel 0)
            chan_wave_pos = chan.pos - wave_chans[0].pos

            if stereo_orientation == StereoOrientation.overlay:
                npt.assert_equal(chan_wave_pos, [0, 0])
            elif stereo_orientation == StereoOrientation.v:  # pos[0]++
                npt.assert_equal(chan_wave_pos, [chan_j, 0])
            else:
                assert stereo_orientation == StereoOrientation.h  # pos[1]++
                npt.assert_equal(chan_wave_pos, [0, chan_j])

            # Check screen edges
            screen_edges = chan.screen_edges
            assert bool(screen_edges & Edges.Top) == (chan.row == 0)
            assert bool(screen_edges & Edges.Left) == (chan.col == 0)
            assert bool(screen_edges & Edges.Bottom) == (chan.row == chan.nrow - 1)
            assert bool(screen_edges & Edges.Right) == (chan.col == chan.ncol - 1)

            # Check stereo edges
            wave_edges = chan.wave_edges
            if stereo_orientation == StereoOrientation.overlay:
                assert wave_edges == ~Edges.NONE
            elif stereo_orientation == StereoOrientation.v:  # pos[0]++
                lr = Edges.Left | Edges.Right
                assert wave_edges & lr == lr
                assert bool(wave_edges & Edges.Top) == (chan.row % stereo_nchan == 0)
                assert bool(wave_edges & Edges.Bottom) == (
                    (chan.row + 1) % stereo_nchan == 0
                )
            else:
                assert stereo_orientation == StereoOrientation.h  # pos[1]++
                tb = Edges.Top | Edges.Bottom
                assert wave_edges & tb == tb
                assert bool(wave_edges & Edges.Left) == (chan.col % stereo_nchan == 0)
                assert bool(wave_edges & Edges.Right) == (
                    (chan.col + 1) % stereo_nchan == 0
                )


def test_renderer_layout():
    # 2 columns
    cfg = RendererConfig(WIDTH, HEIGHT)
    lcfg = LayoutConfig(ncols=2)
    nplots = 15

    datas = [RENDER_Y_ZEROS] * nplots
    r = Renderer(cfg, lcfg, datas, None, None)
    r.update_main_lines(RenderInput.wrap_datas(datas))
    layout = r.layout

    # 2 columns, 8 rows
    assert layout.wave_ncol == 2
    assert layout.wave_nrow == 8
