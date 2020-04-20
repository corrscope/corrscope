from typing import Optional, TYPE_CHECKING, List

import attr
import hypothesis.strategies as hs
import numpy as np
import pytest
from hypothesis import given

from corrscope.channel import ChannelConfig, Channel
from corrscope.corrscope import CorrScope, template_config, Arguments
from corrscope.layout import LayoutConfig
from corrscope.outputs import FFplayOutputConfig
from corrscope.renderer import (
    RendererConfig,
    Renderer,
    LabelPosition,
    Font,
    calc_limits,
    calc_xs,
    calc_center,
    AbstractMatplotlibRenderer,
    RendererFrontend,
    CustomLine,
    RenderInput,
)
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


my_dataclass = attr.dataclass(frozen=True, repr=False)


@my_dataclass
class NamedDebug:
    name: str

    def __init_subclass__(cls):
        my_dataclass(cls)

    def __repr__(self):
        return self.name

    if TYPE_CHECKING:

        def __init__(self, *args, **kwargs):
            pass


# "str" = HTML #FFFFFF color string.


class BG(NamedDebug):
    color: str


class FG(NamedDebug):
    color: str
    draw_fg: bool = True
    line_width: float = 2.0


class Grid(NamedDebug):
    line_width: float
    color: Optional[str]


class Debug(NamedDebug):
    viewport_width: float = 1


bg_black = BG("bg_black", "#000000")
bg_white = BG("bg_white", "#ffffff")
bg_blue = BG("bg_blue", "#0000aa")
bg_yellow = BG("bg_yellow", "#aaaa00")

fg_white = FG("fg_white", "#ffffff")
fg_white_thick = FG("fg_white_thick", "#ffffff", line_width=10.0)
fg_black = FG("fg_black", "#000000")
fg_NONE = FG("fg_NONE", "#ffffff", draw_fg=False)
fg_yellow = FG("fg_yellow", "#aaaa00")
fg_blue = FG("fg_blue", "#0000aa")


grid_0 = Grid("grid_0", 0, "#ff00ff")
grid_1 = Grid("grid_1", 1, "#ff00ff")
grid_10 = Grid("grid_10", 10, "#ff00ff")
grid_NONE = Grid("grid_NONE", 10, None)


debug_NONE = Debug("debug_NONE")
debug_wide = Debug("debug_wide", viewport_width=2)


@attr.dataclass(frozen=True)
class Appearance:
    bg: BG
    fg: FG
    grid: Grid
    debug: Debug = debug_NONE


def all_colors_to_str(val) -> Optional[str]:
    """Called once for each `appear` and `data`."""

    if isinstance(val, Appearance):
        args_tuple = attr.astuple(val, recurse=False)
        return f"appear=Appearance{args_tuple}"

    if isinstance(val, np.ndarray):
        data_type = "stereo" if val.shape[1] > 1 else "mono"
        return "data=" + data_type

    raise ValueError("Unrecognized all_colors parameter, not `appear` or `data`")


mono = RENDER_Y_ZEROS
stereo = RENDER_Y_STEREO

all_colors = pytest.mark.parametrize(
    "appear, data",
    [
        # Test with foreground disabled
        (Appearance(bg_black, fg_NONE, grid_NONE), mono),
        (Appearance(bg_blue, fg_NONE, grid_1), mono),
        # Test with grid disabled
        (Appearance(bg_black, fg_white, grid_NONE), mono),
        (Appearance(bg_white, fg_black, grid_NONE), mono),
        (Appearance(bg_blue, fg_yellow, grid_NONE), mono),
        (Appearance(bg_yellow, fg_blue, grid_NONE), mono),
        # Test FG line thickness
        (Appearance(bg_black, fg_white_thick, grid_NONE), mono),
        # Test various grid thicknesses
        (Appearance(bg_white, fg_black, grid_0), mono),
        (Appearance(bg_blue, fg_yellow, grid_1), mono),
        (Appearance(bg_blue, fg_yellow, grid_10), mono),
        # Test with stereo
        (Appearance(bg_black, fg_white, grid_NONE), stereo),
        (Appearance(bg_blue, fg_yellow, grid_0), stereo),
        (Appearance(bg_blue, fg_yellow, grid_10), stereo),
        # Test debugging (viewport width)
        (Appearance(bg_black, fg_white, grid_NONE, debug_wide), mono),
    ],
    ids=all_colors_to_str,
)


def get_renderer_config(appear: Appearance) -> RendererConfig:
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        # BG
        bg_color=appear.bg.color,
        # FG
        init_line_color=appear.fg.color,
        line_width=appear.fg.line_width,
        viewport_width=appear.debug.viewport_width,
        # Grid
        grid_color=appear.grid.color,
        grid_line_width=appear.grid.line_width,
        stereo_grid_opacity=OPACITY,
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

    r = Renderer(cfg, lcfg, datas, None, None)
    verify(r, appear, datas)

    # Ensure default ChannelConfig(line_color=None) does not override line color
    chan = ChannelConfig(wav_path="")
    channels = [chan] * NPLOTS
    r = Renderer(cfg, lcfg, datas, channels, None)
    verify(r, appear, datas)


@all_colors
def test_line_colors(appear: Appearance, data):
    """ Test channel-specific line color overrides """
    cfg = get_renderer_config(appear)
    lcfg = LayoutConfig(orientation=ORIENTATION)
    datas = [data] * NPLOTS

    # Move line color (appear.fg.color) from renderer cfg to individual channel.
    chan = ChannelConfig(wav_path="", line_color=appear.fg.color)
    channels = [chan] * NPLOTS
    cfg.init_line_color = "#888888"
    chan.line_color = appear.fg.color

    r = Renderer(cfg, lcfg, datas, channels, None)
    verify(r, appear, datas)


TOLERANCE = 3


def verify(r: Renderer, appear: Appearance, datas: List[Optional[np.ndarray]]):
    bg_str = appear.bg.color

    fg_str = appear.fg.color
    draw_fg = appear.fg.draw_fg
    fg_line_width = appear.fg.line_width

    grid_str = appear.grid.color
    grid_line_width = appear.grid.line_width

    viewport_width = appear.debug.viewport_width

    if draw_fg:
        r.update_main_lines(RenderInput.wrap_datas(datas))

    frame_colors: np.ndarray = np.frombuffer(r.get_frame(), dtype=np.uint8).reshape(
        (-1, BYTES_PER_PIXEL)
    )

    bg_u8 = color_to_bytes(bg_str)
    all_colors = [bg_u8]

    fg_u8 = color_to_bytes(fg_str)
    if draw_fg:
        all_colors.append(fg_u8)

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
    does_fg_appear_here = np.prod(frame_colors == fg_u8, axis=-1)
    does_fg_appear = does_fg_appear_here.any()
    # it might be 136 == #888888 == init_line_color
    assert does_fg_appear == draw_fg, f"{does_fg_appear} != {draw_fg}"

    if draw_fg:
        expected_fg_pixels = NPLOTS * (WIDTH / viewport_width) * fg_line_width
        assert does_fg_appear_here.sum() == pytest.approx(
            expected_fg_pixels, abs=expected_fg_pixels * 0.1
        )

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
@parametrize("label_position", list(LabelPosition.__members__.values()))
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

    r = Renderer(cfg, lcfg, datas, None, None)
    r.add_labels(labels)
    if not hide_lines:
        r.update_main_lines(RenderInput.wrap_datas(datas))

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
    cfg = template_config(
        channels=[ChannelConfig("tests/stereo in-phase.wav")],
        render_stereo=Flatten.Stereo,
        end_time=0.5,  # Reduce test duration
        render=RendererConfig(WIDTH, HEIGHT),
    )

    # Make sure it doesn't crash.
    corr = CorrScope(cfg, Arguments(".", [FFplayOutputConfig()]))
    corr.play()


# Image dimension tests
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
        mocker.patch.object(AbstractMatplotlibRenderer, "_save_background")
        datas = []
    else:
        datas = [RENDER_Y_ZEROS]

    try:
        renderer = Renderer(cfg, LayoutConfig(), datas, None, None)
        if not speed_hack:
            renderer.update_main_lines(RenderInput.wrap_datas(datas))
            renderer.get_frame()
    except Exception:
        perr(cfg.divided_width)
        raise


# X-axis stride tests
@given(N=hs.integers(2, 1000), stride=hs.integers(1, 1000))
def test_calc_limits_xs(N: int, stride: int):
    """Sanity check to ensure that data is drawn from 1 edge of screen to another,
    that calc_limits() and calc_xs() match.

    Test that calc_center() == "to the left of N//2".
    """
    min, max = calc_limits(N, stride)
    xs = calc_xs(N, stride)
    assert xs[0] == min
    assert xs[-1] == max

    center = calc_center(stride)
    assert np.mean([xs[N // 2 - 1], xs[N // 2]]) == pytest.approx(center)


# Debug plotting tests
@parametrize("integration", [False, True])
def test_renderer_knows_stride(mocker: "pytest_mock.MockFixture", integration: bool):
    """
    If Renderer draws both "main line" and "custom mono lines" at once,
    each line must have its x-coordinates multiplied by the stride.

    Renderer uses "main line stride = 1" by default,
    but this results in the main line appearing too narrow compared to debug lines.
    Make sure CorrScope.play() gives Renderer the correct values.
    """

    # Stub out FFplay output.
    mocker.patch.object(FFplayOutputConfig, "cls")

    subsampling = 2
    width_mul = 3

    chan_cfg = ChannelConfig("tests/sine440.wav", render_width=width_mul)
    corr_cfg = template_config(
        render_subsampling=subsampling, channels=[chan_cfg], end_time=0
    )

    if integration:
        corr = CorrScope(corr_cfg, Arguments(".", [FFplayOutputConfig()]))
        corr.play()
        assert corr.renderer.render_strides == [subsampling * width_mul]
    else:
        channel = Channel(chan_cfg, corr_cfg, channel_idx=0)
        data = channel.get_render_around(0)
        renderer = Renderer(
            corr_cfg.render, corr_cfg.layout, [data], [chan_cfg], [channel]
        )
        assert renderer.render_strides == [subsampling * width_mul]


# Multiple inheritance tests
def test_frontend_overrides_backend(mocker: "pytest_mock.MockFixture"):
    """
    class Renderer inherits from (RendererFrontend, backend).

    RendererFrontend.get_frame() is a wrapper around backend.get_frame()
    and should override it (RendererFrontend should come first in MRO).

    Make sure RendererFrontend methods overshadow backend methods.
    """

    # If RendererFrontend.get_frame() override is removed, delete this entire test.
    frontend_get_frame = mocker.spy(RendererFrontend, "get_frame")
    backend_get_frame = mocker.spy(AbstractMatplotlibRenderer, "get_frame")

    corr_cfg = template_config()
    chan_cfg = ChannelConfig("tests/sine440.wav")
    channel = Channel(chan_cfg, corr_cfg, channel_idx=0)
    data = channel.get_render_around(0)

    renderer = Renderer(corr_cfg.render, corr_cfg.layout, [data], [chan_cfg], [channel])
    renderer.update_main_lines([RenderInput.stub_new(data)])
    renderer.get_frame()

    assert frontend_get_frame.call_count == 1
    assert backend_get_frame.call_count == 1


def test_custom_line():
    def verify(line: CustomLine, xdata: list):
        line_xdata = line.xdata
        assert isinstance(line_xdata, np.ndarray)
        assert line_xdata.tolist() == xdata

    stride = 1
    noop = lambda x: None

    line = CustomLine(stride, [3], noop, noop)
    verify(line, [3])

    line.xdata = [4, 4]
    verify(line, [4, 4])
