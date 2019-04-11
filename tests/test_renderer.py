from typing import Optional, TYPE_CHECKING, List

import attr
import hypothesis.strategies as hs
import numpy as np
import pytest
from hypothesis import given

from corrscope.channel import ChannelConfig
from corrscope.corrscope import CorrScope, default_config, Arguments
from corrscope.layout import LayoutConfig
from corrscope.outputs import FFplayOutputConfig
from corrscope.renderer import RendererConfig, Renderer, LabelPosition, Font
from corrscope.util import perr
from corrscope.wave import Flatten

if TYPE_CHECKING:
    import pytest_mock

parametrize = pytest.mark.parametrize


color_to_bytes = Renderer.color_to_bytes
BYTES_PER_PIXEL = Renderer.bytes_per_pixel

WIDTH = 64
HEIGHT = 64

RENDER_Y_ZEROS = np.full((2, 1), 0.5)
RENDER_Y_STEREO = np.full((2, 2), 0.5)
OPACITY = 2 / 3


def behead(string: str, header: str) -> str:
    if not string.startswith(header):
        raise ValueError(f"{string} does not start with {header}")
    return string[len(header) :]


def appearance_to_str(val):
    """Called once for each `appear` and `data`."""
    if isinstance(val, Appearance):
        # Remove class name.
        return behead(str(val), Appearance.__name__)
    if isinstance(val, np.ndarray):
        return "stereo" if val.shape[1] > 1 else "mono"
    return None


@attr.dataclass
class Appearance:
    bg_str: str
    fg_str: str
    grid_str: Optional[str]
    grid_line_width: float


all_colors = pytest.mark.parametrize(
    "appear, data",
    [
        (Appearance("#000000", "#ffffff", None, 1), RENDER_Y_ZEROS),
        (Appearance("#ffffff", "#000000", None, 2), RENDER_Y_ZEROS),
        (Appearance("#0000aa", "#aaaa00", None, 1), RENDER_Y_ZEROS),
        (Appearance("#aaaa00", "#0000aa", None, 2), RENDER_Y_ZEROS),
        (Appearance("#0000aa", "#aaaa00", "#ff00ff", 1), RENDER_Y_ZEROS),
        (Appearance("#aaaa00", "#0000aa", "#ff00ff", 1), RENDER_Y_ZEROS),
        (Appearance("#aaaa00", "#0000aa", "#ff00ff", 0), RENDER_Y_ZEROS),
        (Appearance("#0000aa", "#aaaa00", "#ff00ff", 1), RENDER_Y_STEREO),
        (Appearance("#aaaa00", "#0000aa", "#ff00ff", 1), RENDER_Y_STEREO),
    ],
    ids=appearance_to_str,
)


def get_renderer_config(appear: Appearance) -> RendererConfig:
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=appear.bg_str,
        init_line_color=appear.fg_str,
        grid_color=appear.grid_str,
        grid_line_width=appear.grid_line_width,
        stereo_grid_opacity=OPACITY,
        line_width=2.0,
        antialiasing=False,
    )
    return cfg


NPLOTS = 2
ORIENTATION = "h"
GRID_NPIXEL = WIDTH


@all_colors
def test_default_colors(appear: Appearance, data):
    """ Test the default background/foreground colors. """
    cfg = get_renderer_config(appear)
    lcfg = LayoutConfig(orientation=ORIENTATION)
    datas = [data] * NPLOTS

    r = Renderer(cfg, lcfg, datas, None)
    verify(r, appear, datas)

    # Ensure default ChannelConfig(line_color=None) does not override line color
    chan = ChannelConfig(wav_path="")
    channels = [chan] * NPLOTS
    r = Renderer(cfg, lcfg, datas, channels)
    verify(r, appear, datas)


@all_colors
def test_line_colors(appear: Appearance, data):
    """ Test channel-specific line color overrides """
    cfg = get_renderer_config(appear)
    lcfg = LayoutConfig(orientation=ORIENTATION)
    datas = [data] * NPLOTS

    # Move line color (appear.fg_str) from renderer cfg to individual channel.
    chan = ChannelConfig(wav_path="", line_color=appear.fg_str)
    channels = [chan] * NPLOTS
    cfg.init_line_color = "#888888"
    chan.line_color = appear.fg_str

    r = Renderer(cfg, lcfg, datas, channels)
    verify(r, appear, datas)


TOLERANCE = 3


def verify(r: Renderer, appear: Appearance, datas: List[np.ndarray]):
    bg_str = appear.bg_str
    fg_str = appear.fg_str
    grid_str = appear.grid_str
    grid_line_width = appear.grid_line_width

    r.update_main_lines(datas)
    frame_colors: np.ndarray = np.frombuffer(r.get_frame(), dtype=np.uint8).reshape(
        (-1, BYTES_PER_PIXEL)
    )

    bg_u8 = color_to_bytes(bg_str)
    fg_u8 = color_to_bytes(fg_str)
    all_colors = [bg_u8, fg_u8]

    is_grid = bool(grid_str and grid_line_width >= 1)

    if is_grid:
        grid_u8 = color_to_bytes(grid_str)
        all_colors.append(grid_u8)
    else:
        grid_u8 = np.array([1000] * BYTES_PER_PIXEL)

    data = datas[0]
    assert (data.shape[1] > 1) == (data is RENDER_Y_STEREO)
    is_stereo = is_grid and data.shape[1] > 1
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
    does_grid_appear_here = np.prod(frame_colors == grid_u8, axis=-1)
    does_grid_appear = does_grid_appear_here.any()
    assert does_grid_appear == is_grid, f"{does_grid_appear} != {is_grid}"

    if is_grid:
        assert np.sum(does_grid_appear_here) == pytest.approx(
            GRID_NPIXEL * grid_line_width, abs=GRID_NPIXEL * 0.1
        )

    # Ensure stereo grid color is present
    if is_stereo:
        assert (
            np.min(np.sum(np.abs(frame_colors - stereo_grid_u8), axis=-1)) < TOLERANCE
        ), "Missing stereo gridlines"

    assert (np.amax(frame_colors, axis=0) == np.amax(all_colors, axis=0)).all()
    assert (np.amin(frame_colors, axis=0) == np.amin(all_colors, axis=0)).all()


# Test label positioning and rendering
@parametrize("label_position", LabelPosition.__members__.values())
@parametrize("data", [RENDER_Y_ZEROS, RENDER_Y_STEREO])
@parametrize("hide_lines", [True, False])
def test_label_render(label_position: LabelPosition, data, hide_lines):
    """Test that text labels are drawn:
    - in the correct quadrant
    - with the correct color (defaults to init_line_color)
    - even if no lines are drawn at all
    """
    font_str = "#FF00FF"
    font_u8 = color_to_bytes(font_str)

    # If hide_lines: set line color to purple, draw text using the line color.
    # Otherwise: draw lines white, draw text purple,
    cfg_kwargs = {}
    if hide_lines:
        cfg_kwargs.update(init_line_color=font_str)

    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        antialiasing=False,
        label_font=Font(size=16, bold=True),
        label_position=label_position,
        label_color_override=font_str,
        **cfg_kwargs,
    )

    lcfg = LayoutConfig()

    nplots = 1
    labels = ["#"] * nplots
    datas = [data] * nplots

    r = Renderer(cfg, lcfg, datas, None)
    r.add_labels(labels)
    if not hide_lines:
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


@pytest.mark.parametrize(
    "target_int, res_divisor", [(50, 2.0), (51, 2.0), (100, 1.001)]
)
def test_res_divisor_rounding_fixed(target_int: int, res_divisor: float):
    verify_res_divisor_rounding(target_int, res_divisor, speed_hack=False)


@given(target_int=hs.integers(1, 10000), res_divisor=hs.floats(1, 100))
def test_res_divisor_rounding_hypothesis(target_int: int, res_divisor: float, mocker):
    verify_res_divisor_rounding(target_int, res_divisor, speed_hack=True, mocker=mocker)


def verify_res_divisor_rounding(
    target_int: int,
    res_divisor: float,
    speed_hack: bool,
    mocker: "pytest_mock.MockFixture" = None,
):
    """Ensure that pathological-case float rounding errors
    don't cause inconsistent dimensions and assertion errors."""
    target_dim = target_int + 0.5
    undivided_dim = round(target_dim * res_divisor)

    cfg = RendererConfig(undivided_dim, undivided_dim, res_divisor=res_divisor)
    cfg.before_preview()

    if speed_hack:
        mocker.patch.object(Renderer, "_save_background")
        datas = []
    else:
        datas = [RENDER_Y_ZEROS]

    try:
        renderer = Renderer(cfg, LayoutConfig(), datas, channel_cfgs=None)
        if not speed_hack:
            renderer.update_main_lines(datas)
            renderer.get_frame()
    except Exception:
        perr(cfg.divided_width)
        raise
