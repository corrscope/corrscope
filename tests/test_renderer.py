from unittest.mock import patch

import pytest

from ovgenpy.renderer import RendererConfig, MatplotlibRenderer


WIDTH = 640
HEIGHT = 360

def test_config():
    with pytest.raises(ValueError):
        RendererConfig(WIDTH, HEIGHT, nrows=1, ncols=1)

    one_col = RendererConfig(WIDTH, HEIGHT, ncols=1)
    assert one_col

    one_row = RendererConfig(WIDTH, HEIGHT, nrows=1)
    assert one_row

    default = RendererConfig(WIDTH, HEIGHT)
    assert default.ncols == 1   # Should default to single-column layout
    assert default.nrows is None


def test_renderer():
    """
    TODO check image output using:
    https://matplotlib.org/devel/testing.html#writing-an-image-comparison-test

    https://stackoverflow.com/a/27950953
    "[I]mage comparison tests end up bring more trouble than they are worth"
    """

    # 2 columns
    cfg = RendererConfig(WIDTH, HEIGHT, ncols=2)
    nplots = 15

    r = MatplotlibRenderer(cfg, nplots)

    # 2 columns, 8 rows
    assert r.layout.ncols == 2
    assert r.layout.nrows == 8


# TODO: test get_frame()
# (integration test) ensure rendering to output works
