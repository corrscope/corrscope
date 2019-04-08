from typing import Optional, TYPE_CHECKING, List

import matplotlib.colors
import numpy as np
import pytest

from corrscope.channel import ChannelConfig
from corrscope.corrscope import CorrScope, default_config, Arguments
from corrscope.layout import LayoutConfig
from corrscope.outputs import BYTES_PER_PIXEL, FFplayOutputConfig
from corrscope.renderer import RendererConfig, MatplotlibRenderer, LabelPosition, Font
from corrscope.wave import Flatten

if TYPE_CHECKING:
    import pytest_mock

parametrize = pytest.mark.parametrize


WIDTH = 64
HEIGHT = 64

RENDER_Y_ZEROS = np.zeros((2, 1))
RENDER_Y_STEREO = np.zeros((2, 2))
OPACITY = 2 / 3

all_colors = pytest.mark.parametrize(
    "bg_str,fg_str,grid_str,data",
    [
        ("#000000", "#ffffff", None, RENDER_Y_ZEROS),
        ("#ffffff", "#000000", None, RENDER_Y_ZEROS),
        ("#0000aa", "#aaaa00", None, RENDER_Y_ZEROS),
        ("#aaaa00", "#0000aa", None, RENDER_Y_ZEROS),
        # Enabling ~~beautiful magenta~~ gridlines enables Axes rectangles.
        # Make sure bg is disabled, so they don't overwrite global figure background.
        ("#0000aa", "#aaaa00", "#ff00ff", RENDER_Y_ZEROS),
        ("#aaaa00", "#0000aa", "#ff00ff", RENDER_Y_ZEROS),
        ("#0000aa", "#aaaa00", "#ff00ff", RENDER_Y_STEREO),
        ("#aaaa00", "#0000aa", "#ff00ff", RENDER_Y_STEREO),
    ],
)

NPLOTS = 2


@all_colors
def test_default_colors(bg_str, fg_str, grid_str, data):
    """ Test the default background/foreground colors. """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color=fg_str,
        grid_color=grid_str,
        stereo_grid_opacity=OPACITY,
        line_width=2.0,
        antialiasing=False,
    )
    lcfg = LayoutConfig()
    datas = [data] * NPLOTS

    r = MatplotlibRenderer(cfg, lcfg, datas, None)
    verify(r, bg_str, fg_str, grid_str, datas)

    # Ensure default ChannelConfig(line_color=None) does not override line color
    chan = ChannelConfig(wav_path="")
    channels = [chan] * NPLOTS
    r = MatplotlibRenderer(cfg, lcfg, datas, channels)
    verify(r, bg_str, fg_str, grid_str, datas)


@all_colors
def test_line_colors(bg_str, fg_str, grid_str, data):
    """ Test channel-specific line color overrides """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color="#888888",
        grid_color=grid_str,
        stereo_grid_opacity=OPACITY,
        line_width=2.0,
        antialiasing=False,
    )
    lcfg = LayoutConfig()
    datas = [data] * NPLOTS

    chan = ChannelConfig(wav_path="", line_color=fg_str)
    channels = [chan] * NPLOTS
    r = MatplotlibRenderer(cfg, lcfg, datas, channels)
    verify(r, bg_str, fg_str, grid_str, datas)


TOLERANCE = 3


def verify(
    r: MatplotlibRenderer,
    bg_str,
    fg_str,
    grid_str: Optional[str],
    datas: List[np.ndarray],
):
    r.update_main_lines(datas)
    frame_colors: np.ndarray = np.frombuffer(r.get_frame(), dtype=np.uint8).reshape(
        (-1, BYTES_PER_PIXEL)
    )

    bg_u8 = to_rgb(bg_str)
    fg_u8 = to_rgb(fg_str)
    all_colors = [bg_u8, fg_u8]

    if grid_str:
        grid_u8 = to_rgb(grid_str)
        all_colors.append(grid_u8)
    else:
        grid_u8 = bg_u8

    data = datas[0]
    assert (data.shape[1] > 1) == (data is RENDER_Y_STEREO)
    is_stereo = data.shape[1] > 1
    if is_stereo:
        stereo_grid_u8 = (grid_u8 * OPACITY + bg_u8 * (1 - OPACITY)).astype(int)
        all_colors.append(stereo_grid_u8)

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

    # Ensure stereo grid color is present
    if is_stereo:
        assert (
            np.min(np.sum(np.abs(frame_colors - stereo_grid_u8), axis=-1)) < TOLERANCE
        ), "Missing stereo gridlines"

    assert (np.amax(frame_colors, axis=0) == np.amax(all_colors, axis=0)).all()
    assert (np.amin(frame_colors, axis=0) == np.amin(all_colors, axis=0)).all()


def to_rgb(c) -> np.ndarray:
    to_rgb = matplotlib.colors.to_rgb
    return np.array([round(c * 255) for c in to_rgb(c)], dtype=int)


# Test label positioning and rendering
@parametrize("label_position", LabelPosition.__members__.values())
@parametrize("data", [RENDER_Y_ZEROS, RENDER_Y_STEREO])
def test_label_render(label_position: LabelPosition, data):
    """Test that text labels are drawn:
    - in the correct quadrant
    - with the correct color
    """
    font_str = "#FF00FF"
    font_u8 = to_rgb(font_str)

    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        antialiasing=False,
        font=Font(size=16, bold=True),
        label_position=label_position,
        font_color_override=font_str,
    )

    lcfg = LayoutConfig()

    nplots = 1
    labels = ["#"] * nplots
    datas = [data] * nplots

    r = MatplotlibRenderer(cfg, lcfg, datas, None)
    r.add_labels(labels)
    r.update_main_lines(datas)
    frame_buffer: np.ndarray = np.frombuffer(r.get_frame(), dtype=np.uint8).reshape(
        (r.h, r.w, BYTES_PER_PIXEL)
    )
    # Allow mutation
    frame_buffer = frame_buffer.copy()

    yslice = label_position.y.match(
        top=slice(None, r.h // 2), bottom=slice(r.h // 2, None)
    )
    xslice = label_position.x.match(
        left=slice(None, r.w // 2), right=slice(r.w // 2, None)
    )
    quadrant = frame_buffer[yslice, xslice]

    assert np.prod(quadrant == font_u8, axis=-1).any(), "Missing text"

    quadrant[:] = 0
    assert not np.prod(
        frame_buffer == font_u8, axis=-1
    ).any(), "Text appeared in wrong area of screen"


# Stereo *renderer* integration tests.
def test_stereo_render_integration(mocker: "pytest_mock.MockFixture"):
    """Ensure corrscope plays/renders in stereo, without crashing."""

    # Stub out FFplay output.
    mocker.patch.object(FFplayOutputConfig, "cls")

    # Render in stereo.
    cfg = default_config(
        channels=[ChannelConfig("tests/stereo in-phase.wav")],
        render_stereo=Flatten.Stereo,
        end_time=0.5,  # Reduce test duration
        render=RendererConfig(WIDTH, HEIGHT),
    )

    # Make sure it doesn't crash.
    corr = CorrScope(cfg, Arguments(".", [FFplayOutputConfig()]))
    corr.play()
