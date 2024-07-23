"""
Backend implementations should not inherit from RendererFrontend,
since they don't need to know.

Implementation: Multiple inheritance:
Renderer inherits from (RendererFrontend, backend implementation).
Backend implementation does not know about RendererFrontend.
"""

import enum
import math
import os.path
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    Optional,
    List,
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
    Sequence,
    Type,
    Union,
    Tuple,
    Dict,
    DefaultDict,
    MutableSequence,
)

# DO NOT IMPORT MATPLOTLIB UNTIL WE DELETE mpl_config_dir!
import attr
import numpy as np

import corrscope.generate
from corrscope.channel import ChannelConfig, Channel
from corrscope.config import DumpableAttrs, with_units, TypedEnumDump
from corrscope.layout import (
    RendererLayout,
    LayoutConfig,
    unique_by_id,
    RegionSpec,
    Edges,
)
from corrscope.util import coalesce, obj_name

"""
On first import, matplotlib.font_manager spends nearly 10 seconds
building a font cache.

PyInstaller redirects matplotlib's font cache to a temporary folder,
deleted after the app exits. This is because in one-file .exe mode,
matplotlib-bundled fonts are extracted and deleted whenever the app runs,
and font cache entries point to invalid paths.

- https://github.com/pyinstaller/pyinstaller/issues/617
- https://github.com/pyinstaller/pyinstaller/blob/c06d853c0c4df7480d3fa921851354d4ee11de56/PyInstaller/loader/rthooks/pyi_rth_mplconfig.py#L35-L37

corrscope uses one-folder mode
and deletes all matplotlib-bundled fonts to save space. So reenable global font cache.
"""

mpl_config_dir = "MPLCONFIGDIR"
if mpl_config_dir in os.environ:
    del os.environ[mpl_config_dir]

# matplotlib.use() only affects pyplot. We don't use pyplot.

import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.image
import matplotlib.patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.spines import Spine
    from matplotlib.text import Text, Annotation


# Used by outputs.py.
ByteBuffer = Union[bytes, np.ndarray, memoryview]


def default_color() -> str:
    # import matplotlib.colors
    # colors = np.array([int(x, 16) for x in '1f 77 b4'.split()], dtype=float)
    # colors /= np.amax(colors)
    # colors **= 1/3
    #
    # return matplotlib.colors.to_hex(colors, keep_alpha=False)
    return "#ffffff"


T = TypeVar("T")


class LabelX(enum.Enum):
    Left = enum.auto()
    Right = enum.auto()

    def match(self, *, left: T, right: T) -> T:
        if self is self.Left:
            return left
        if self is self.Right:
            return right
        raise ValueError("failed match")


class LabelY(enum.Enum):
    Bottom = enum.auto()
    Top = enum.auto()

    def match(self, *, bottom: T, top: T) -> T:
        if self is self.Bottom:
            return bottom
        if self is self.Top:
            return top
        raise ValueError("failed match")


class LabelPosition(TypedEnumDump):
    def __init__(self, x: LabelX, y: LabelY):
        self.x = x
        self.y = y

    LeftBottom = (LabelX.Left, LabelY.Bottom)
    LeftTop = (LabelX.Left, LabelY.Top)
    RightBottom = (LabelX.Right, LabelY.Bottom)
    RightTop = (LabelX.Right, LabelY.Top)


class Font(DumpableAttrs, always_dump="*"):
    # Font file selection
    family: Optional[str] = None
    bold: bool = False
    italic: bool = False
    # Font size
    size: float = with_units("pt", default=28)
    # QFont implementation details
    toString: str = None


class RendererConfig(
    DumpableAttrs, always_dump="*", exclude="viewport_width viewport_height"
):
    width: int
    height: int
    line_width: float = with_units("px", default=1.5)
    line_outline_width: float = with_units("px", default=0.0)
    grid_line_width: float = with_units("px", default=1.0)

    @property
    def divided_width(self):
        return round(self.width / self.res_divisor)

    @property
    def divided_height(self):
        return round(self.height / self.res_divisor)

    bg_color: str = "#000000"
    bg_image: str = ""
    init_line_color: str = default_color()
    global_line_outline_color: str = "#000000"

    @property
    def global_line_color(self) -> str:
        return self.init_line_color

    # Whether to color lines by the pitch of the current note.
    global_color_by_pitch: bool = False
    # 12 colors, representing how to color pitches C through B.
    pitch_colors: List[str] = corrscope.generate.spectral_colors

    grid_color: Optional[str] = None
    stereo_grid_opacity: float = 0.25

    midline_color: str = "#404040"
    v_midline: bool = False
    h_midline: bool = False

    global_stereo_bars: bool = False
    stereo_bar_color: str = "#88ffff"

    # Label settings
    label_font: Font = attr.ib(factory=Font)

    label_position: LabelPosition = LabelPosition.LeftTop
    # The text will be located (label_padding_ratio * label_font.size) from the corner.
    label_padding_ratio: float = with_units("px/pt", default=0.5)
    label_color_override: Optional[str] = None

    @property
    def get_label_color(self):
        return coalesce(self.label_color_override, self.global_line_color)

    antialiasing: bool = True

    # Performance (skipped when recording to video)
    res_divisor: float = 1.0

    # Debugging only
    viewport_width: float = 1
    viewport_height: float = 1

    def __attrs_post_init__(self) -> None:
        # round(np.int32 / float) == np.float32, but we want int.
        assert isinstance(self.width, (int, float))
        assert isinstance(self.height, (int, float))
        assert len(self.pitch_colors) == 12, len(self.pitch_colors)

    # Both before_* functions should be idempotent, AKA calling twice does no harm.
    def before_preview(self) -> None:
        """Called *once* before preview. Does nothing."""
        pass

    def before_record(self) -> None:
        """Called *once* before recording video. Eliminates res_divisor."""
        self.res_divisor = 1


def gen_circular_cmap(colors: List[str]):
    colors = list(colors)

    # `colors` has 12 distinct entries.
    # LinearSegmentedColormap(colors) takes a real number `x` between 0 and 1,
    # and maps x=0 to colors[0] and x=1 to colors[-1].
    # pitch_cmap should be periodic, so `colors[0]` should appear at both x=0 and x=1.
    colors.append(colors[0])

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "12-tone spectrum", colors, N=256, gamma=1.0
    )
    return cmap


def freq_to_color(cmap, freq: Optional[float], fallback_color: str) -> str:
    if not freq:
        return fallback_color

    key_number = 12 * math.log2(freq / 440) + 69

    color = cmap(math.fmod(key_number, 12) / 12)
    return color


@attr.dataclass
class LineParam:
    color: str
    color_by_pitch: bool
    stereo_bars: bool


StereoLevels = Tuple[float, float]


@attr.dataclass
class RenderInput:
    # Should Renderer store a Wave and take an int?
    # Or take an array on each frame?
    data: np.ndarray
    stereo_levels: Optional[StereoLevels]
    freq_estimate: Optional[float]

    @staticmethod
    def stub_new(data: np.ndarray) -> "RenderInput":
        """
        Stable function to construct a RenderInput given only a data array.
        Used mainly for tests.
        """
        return RenderInput(data, None, None)

    @staticmethod
    def wrap_datas(datas: List[np.ndarray]) -> List["RenderInput"]:
        """
        Stable function to construct a list of RenderInput given only datas.
        Used mainly for tests.
        """
        return [RenderInput.stub_new(data) for data in datas]


UpdateOneLine = Callable[[np.ndarray], Any]


@attr.dataclass
class CustomLine:
    stride: int
    _xdata: np.ndarray = attr.ib(converter=np.array)

    @property
    def xdata(self) -> np.ndarray:
        return self._xdata

    @xdata.setter
    def xdata(self, value):
        self._xdata = np.array(value)

    set_xdata: UpdateOneLine
    set_ydata: UpdateOneLine


@property
@abstractmethod
def abstract_classvar(self) -> Any:
    """A ClassVar to be overriden by a subclass."""


@attr.dataclass
class RendererParams:
    """Serializable between processes."""

    cfg: RendererConfig
    lcfg: LayoutConfig

    data_shapes: List[Tuple[int, ...]]
    """[nwave] (nsamp, nchan), but tuple length is unchecked"""

    channel_cfgs: Optional[List[ChannelConfig]]
    render_strides: Optional[List[int]]
    labels: Optional[List[str]]

    @staticmethod
    def from_obj(
        cfg: RendererConfig,
        lcfg: "LayoutConfig",
        dummy_datas: List[np.ndarray],
        channel_cfgs: Optional[List["ChannelConfig"]],
        channels: Optional[List["Channel"]],
        cfg_dir: Optional[str] = None,
    ):
        if channels is not None:
            render_strides = [channel.render_stride for channel in channels]
            labels = [channel.label for channel in channels]
        else:
            render_strides = None
            labels = None

        # Resolve background image path relative to .yaml directory.
        if cfg_dir and cfg.bg_image:
            cfg = attr.evolve(cfg, bg_image=os.path.join(cfg_dir, cfg.bg_image))

        return RendererParams(
            cfg,
            lcfg,
            [data.shape for data in dummy_datas],
            channel_cfgs,
            render_strides,
            labels,
        )


class _RendererBase(ABC):
    """
    Renderer backend which takes data and produces images.
    Does not touch Wave or Channel.
    """

    cfg: RendererConfig
    lcfg: LayoutConfig
    w: int
    h: int

    pitch_cmap: Any
    nplots: int

    # [nplots] ...
    wave_nsamps: List[int]
    wave_nchans: List[int]
    _line_params: List[LineParam]
    render_strides: List[int]

    # Class attributes and methods
    bytes_per_pixel: int = abstract_classvar
    ffmpeg_pixel_format: str = abstract_classvar

    @staticmethod
    @abstractmethod
    def color_to_bytes(c: str) -> np.ndarray:
        """
        Returns integer ndarray of length RGB_DEPTH.
        This must return ndarray (not bytes or list),
        since the caller performs arithmetic on the return value.

        Only used for tests/test_renderer.py.
        """

    @classmethod
    def from_obj(cls, *args, **kwargs):
        return cls(RendererParams.from_obj(*args, **kwargs))

    # Instance initializer
    def __init__(self, params: RendererParams):
        cfg: RendererConfig = params.cfg
        self.cfg = cfg
        self.lcfg = params.lcfg

        self.w = cfg.divided_width
        self.h = cfg.divided_height

        # Maps a continuous variable from 0 to 1 (representing one octave) to a color.
        self.pitch_cmap = gen_circular_cmap(cfg.pitch_colors)

        data_shapes = params.data_shapes
        self.nplots = len(data_shapes)

        if self.nplots > 0:
            assert len(data_shapes[0]) == 2, data_shapes[0]
        self.wave_nsamps = [shape[0] for shape in data_shapes]
        self.wave_nchans = [shape[1] for shape in data_shapes]

        channel_cfgs = params.channel_cfgs
        if channel_cfgs is None:
            channel_cfgs = [ChannelConfig("") for _ in range(self.nplots)]

        if len(channel_cfgs) != self.nplots:
            raise ValueError(
                f"cannot assign {len(channel_cfgs)} colors to {self.nplots} plots"
            )

        self._line_params = [
            LineParam(
                color=coalesce(ccfg.line_color, cfg.global_line_color),
                color_by_pitch=coalesce(ccfg.color_by_pitch, cfg.global_color_by_pitch),
                stereo_bars=coalesce(ccfg.stereo_bars, cfg.global_stereo_bars),
            )
            for ccfg in channel_cfgs
        ]

        # Load channel strides.
        render_strides = params.render_strides
        if render_strides is not None:
            if len(render_strides) != self.nplots:
                raise ValueError(
                    f"cannot assign {len(render_strides)} channels to {self.nplots} plots"
                )
            self.render_strides = render_strides
        else:
            self.render_strides = [1] * self.nplots

    def is_stereo_bars(self, wave_idx: int):
        return self._line_params[wave_idx].stereo_bars

    # Instance functionality

    @abstractmethod
    def get_frame(self) -> ByteBuffer:
        ...


# See Wave.get_around() and designNotes.md.
# Viewport functions
def calc_limits(N: int, viewport_stride: float) -> Tuple[float, float]:
    halfN = N // 2
    max_x = N - 1
    return np.array([-halfN, -halfN + max_x]) * viewport_stride


def calc_center(viewport_stride: float) -> float:
    return -0.5 * viewport_stride


# Line functions
def calc_xs(N: int, stride: int) -> Sequence[float]:
    halfN = N // 2
    return (np.arange(N) - halfN) * stride


Point = float
Pixel = float

# Matplotlib multiplies all widths by (inch/72 units) (AKA "matplotlib points").
# To simplify code, render output at (72 px/inch), so 1 unit = 1 px.
# For font sizes, convert from font-pt to pixels.
# (Font sizes are used far less than pixel measurements.)

PX_INCH = 72
PIXELS_PER_PT = 96 / 72


def px_from_points(pt: Point) -> Pixel:
    return pt * PIXELS_PER_PT


@attr.dataclass(cmp=False)
class StereoBar:
    rect: Rectangle
    x_center: float
    x_range: float

    def set_range(self, left: float, right: float):
        left = -left

        x = self.x_center + left * self.x_range
        width = (right - left) * self.x_range

        self.rect.set_x(x)
        self.rect.set_width(width)


class AbstractMatplotlibRenderer(_RendererBase, ABC):
    """Matplotlib renderer which can use any backend (agg, mplcairo).
    To pick a backend, subclass and set _canvas_type at the class level.
    """

    _canvas_type: Type["FigureCanvasBase"] = abstract_classvar

    @staticmethod
    @abstractmethod
    def _canvas_to_bytes(canvas: "FigureCanvasBase") -> ByteBuffer:
        pass

    def __init__(self, params: RendererParams):
        super().__init__(params)
        self.color = 0

    # Output frames
    def get_frame(self) -> ByteBuffer:
        """Returns bytes with shape (h, w, self.bytes_per_pixel).
        The actual return value's shape may be flat.
        """
        buffer_rgb = bytes([self.color]) * (self.w * self.h * self.bytes_per_pixel)
        self.color = 255 - self.color

        return buffer_rgb


class MatplotlibAggRenderer(AbstractMatplotlibRenderer):
    # implements AbstractMatplotlibRenderer
    _canvas_type = FigureCanvasAgg

    @staticmethod
    def _canvas_to_bytes(canvas: FigureCanvasAgg) -> ByteBuffer:
        # In matplotlib >= 3.1, canvas.buffer_rgba() returns a zero-copy memoryview.
        # This is faster to print to screen than the previous bytes.
        # Also the APIs are incompatible.

        # Flatten all dimensions of the memoryview.
        return canvas.buffer_rgba().cast("B")

    # Implements _RendererBase.
    bytes_per_pixel = 4
    ffmpeg_pixel_format = "rgb0"

    @staticmethod
    def color_to_bytes(c: str) -> np.ndarray:
        from matplotlib.colors import to_rgba

        return np.array([round(c * 255) for c in to_rgba(c)], dtype=int)


# TODO: PlotConfig
# - align: left vs mid
# - shift/offset: bool


class RendererFrontend(_RendererBase, ABC):
    """Wrapper around _RendererBase implementations, providing a better interface."""

    def __init__(self, params: RendererParams):
        super().__init__(params)

        self._custom_lines = {}  # type: Dict[Any, CustomLine]
        self._vlines = {}  # type: Dict[Any, CustomLine]
        self._absolute = defaultdict(list)

    # Overrides implementations of _RendererBase.
    def get_frame(self) -> ByteBuffer:
        out = super().get_frame()

        for line in self._custom_lines.values():
            line.set_ydata(0 * line.xdata)

        for line in self._vlines.values():
            line.set_xdata(0 * line.xdata)
        return out

    _absolute: DefaultDict[int, MutableSequence[CustomLine]]

    def update_custom_line(
        self,
        name: str,
        wave_idx: int,
        stride: int,
        data: np.ndarray,
        xs: np.ndarray,
        absolute: bool,
    ):
        data = data.copy()
        key = (name, wave_idx)

        if key not in self._custom_lines:
            line = self._add_line_mono(name, wave_idx, stride, data)
            self._custom_lines[key] = line
            if absolute:
                self._absolute[wave_idx].append(line)
        else:
            line = self._custom_lines[key]

        line.xdata = xs
        line.set_xdata(xs)
        line.set_ydata(data)

    def update_vline(
        self, name: str, wave_idx: int, stride: int, x: int, *, absolute: bool = True
    ):
        key = (name, wave_idx)
        if key not in self._vlines:
            line = self._add_vline_mono(name, wave_idx, stride)
            self._vlines[key] = line
            if absolute:
                self._absolute[wave_idx].append(line)
        else:
            line = self._vlines[key]

        line.xdata = [x * stride] * 2
        line.set_xdata(line.xdata)

    def move_viewport(self, wave_idx: int, viewport_center: float):
        for line in self._absolute[wave_idx]:
            line.xdata -= viewport_center
            line.set_xdata(line.xdata)

    def _add_line_mono(
        self,
        name: str,
        wave_idx: int,
        stride: int,
        dummy_data: np.ndarray,
    ) -> CustomLine:
        xs = np.zeros_like(dummy_data)
        ys = np.zeros_like(dummy_data)
        return self._add_xy_line_mono(name, wave_idx, xs, ys, stride)

    def _add_vline_mono(self, name: str, wave_idx: int, stride: int) -> CustomLine:
        return self._add_xy_line_mono(name, wave_idx, [0, 0], [-1, 1], stride)


class Renderer(RendererFrontend, MatplotlibAggRenderer):
    pass
