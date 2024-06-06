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
    def update_main_lines(
        self, inputs: List[RenderInput], trigger_samples: List[int]
    ) -> None:
        ...

    @abstractmethod
    def get_frame(self) -> ByteBuffer:
        ...

    @abstractmethod
    def add_labels(self, labels: List[str]) -> Any:
        ...

    # Primarily used by RendererFrontend, not outside world.
    @abstractmethod
    def _update_lines_stereo(self, inputs: List[RenderInput]) -> None:
        ...

    @abstractmethod
    def _add_xy_line_mono(
        self,
        name: str,
        wave_idx: int,
        xs: Sequence[float],
        ys: Sequence[float],
        stride: int,
    ) -> CustomLine:
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

        dict.__setitem__(
            matplotlib.rcParams, "lines.antialiased", self.cfg.antialiasing
        )

        self._setup_axes(self.wave_nchans)

        if params.labels is not None:
            self.add_labels(params.labels)

        self._artists = []

    _fig: "Figure"

    _artists: List["Artist"]

    # [wave][chan] Axes
    # Primary, used to draw oscilloscope lines and gridlines.
    _wave_chan_to_axes: List[List["Axes"]]  # set by set_layout()

    # _axes_mono[wave] = Axes
    # Secondary, used for titles and debug plots.
    _wave_to_mono_axes: List["Axes"]

    # Fields updated by _update_lines_stereo():
    # [wave][chan] Line2D
    _wave_chan_to_line: "Optional[List[List[Line2D]]]" = None

    # Only for stereo channels, if stereo bars are enabled.
    _wave_to_stereo_bar: "List[Optional[StereoBar]]"

    def _setup_axes(self, wave_nchans: List[int]) -> None:
        """
        Creates a flat array of Matplotlib Axes, with the new layout.
        Sets up each Axes with correct region limits.
        """

        # Only read by unit tests.
        self.layout = RendererLayout(self.lcfg, wave_nchans)
        layout_mono = RendererLayout(self.lcfg, [1] * self.nplots)

        if hasattr(self, "_fig"):
            raise Exception("I don't currently expect to call _setup_axes() twice")
            # plt.close(self.fig)

        cfg = self.cfg

        self._fig = Figure()
        self._canvas_type(self._fig)

        px_inch = PX_INCH / cfg.res_divisor
        self._fig.set_dpi(px_inch)

        """
        Requirements:
        - px_inch /= res_divisor (to scale visual elements correctly)
        - int(set_size_inches * px_inch) == self.w,h
            - matplotlib uses int instead of round. Who knows why.
        - round(set_size_inches * px_inch) == self.w,h
            - just in case matplotlib changes its mind

        Solution:
        - (set_size_inches * px_inch) == self.w,h + 0.25
        - set_size_inches == (self.w,h + 0.25) / px_inch
        """
        offset = 0.25
        self._fig.set_size_inches(
            (self.w + offset) / px_inch, (self.h + offset) / px_inch
        )

        real_dims = self._fig.canvas.get_width_height()
        assert (self.w, self.h) == real_dims, [(self.w, self.h), real_dims]
        del real_dims

        # Setup background
        self._fig.set_facecolor(cfg.bg_color)

        if cfg.bg_image:
            img = mpl.image.imread(cfg.bg_image)

            ax = self._fig.add_axes([0, 0, 1, 1])

            # Hide black borders around screen edge.
            ax.set_axis_off()

            # Size image to fill screen pixel-perfectly. Somehow, matplotlib requires
            # showing the image 1 screen-pixel smaller than the full area.

            # Get image dimensions (in ipx).
            w = img.shape[1]
            h = img.shape[0]

            # Setup axes to fit image to screen (while maintaining square pixels).
            # Axes automatically expand their limits to maintain square coordinates,
            # while imshow() stretches images to the full area supplied.
            ax.set_xlim(0, w)
            ax.set_ylim(0, h)

            # Calculate (image pixels per screen pixel). Since we fit the image
            # on-screen, pick the minimum of the horizontal/vertical zoom factors.
            zoom = min(self.w / w, self.h / h)
            ipx_per_spx = 1 / zoom

            # imshow() takes coordinates in axes units (here, ipx) and renders to
            # screen pixels. To workaround matplotlib stretching images off-screen,
            # we need an extent 1 spx smaller than full scale. So subtract 1 spx
            # (converted to ipx) from dimensions.
            ax.imshow(img, extent=(0, w - ipx_per_spx, 0, h - ipx_per_spx))

        # Create Axes (using self.lcfg, wave_nchans)
        # [wave][chan] Axes
        self._wave_chan_to_axes = self.layout.arrange(self._axes_factory)

        # _axes_mono[wave] = Axes
        self._wave_to_mono_axes = []

        """
        When calling _axes_factory() with the same position twice, we should pass a
        different label to get a different Axes, to avoid warning:
        
        >>> Adding an axes using the same arguments as a previous axes
        currently reuses the earlier instance.
        In a future version, a new instance will always be created and returned.
        Meanwhile, this warning can be suppressed, and the future behavior ensured,
        by passing a unique label to each axes instance.

        <<< ax=fig.add_axes(label=) is unused, even if you call ax.legend().
        """
        # Returns 2D list of [self.nplots][1]Axes.
        axes_mono_2d = layout_mono.arrange(self._axes_factory, label="mono")
        for axes_list in axes_mono_2d:
            (axes,) = axes_list  # type: Axes

            # Pick colormap used for debug lines (_add_xy_line_mono()).
            # List of colors at
            # https://matplotlib.org/gallery/color/colormap_reference.html
            # Discussion at https://github.com/matplotlib/matplotlib/issues/10840
            cmap: ListedColormap = matplotlib.colormaps["Accent"]
            colors = cmap.colors
            axes.set_prop_cycle(color=colors)

            self._wave_to_mono_axes.append(axes)

        # Setup axes
        for wave_idx, N in enumerate(self.wave_nsamps):
            chan_to_axes = self._wave_chan_to_axes[wave_idx]

            # Calculate the bounds of an Axes object to match the scale of calc_xs()
            # (unless cfg.viewport_width != 1).
            viewport_stride = self.render_strides[wave_idx] * cfg.viewport_width
            xlims = calc_limits(N, viewport_stride)
            ylim = cfg.viewport_height

            def scale_axes(ax: "Axes"):
                ax.set_xlim(*xlims)
                ax.set_ylim(-ylim, ylim)

            scale_axes(self._wave_to_mono_axes[wave_idx])

            # When using overlay stereo, all channels map to the same Axes object.
            for ax in unique_by_id(chan_to_axes):
                scale_axes(ax)

                # Setup midlines (depends on max_x and wave_data)
                midline_color = cfg.midline_color
                midline_width = cfg.grid_line_width

                # Not quite sure if midlines or gridlines draw on top
                kw = dict(color=midline_color, linewidth=midline_width)
                if cfg.v_midline:
                    ax.axvline(x=calc_center(viewport_stride), **kw)
                if cfg.h_midline:
                    ax.axhline(y=0, **kw)

        self._save_background()

    transparent = "#00000000"

    # satisfies RegionFactory
    def _axes_factory(self, r: RegionSpec, label: str = "") -> "Axes":
        cfg = self.cfg

        # Calculate plot positions (relative to bottom-left) as fractions of the screen.
        width = 1 / r.ncol
        left = r.col / r.ncol
        assert 0 <= left < 1

        height = 1 / r.nrow
        # We index rows from top down, but matplotlib positions plots from bottom up.
        # The final row (row = nrow-1) is located at the bottom of the graph, at y=0.
        bottom = (r.nrow - (r.row + 1)) / r.nrow
        assert 0 <= bottom < 1

        # Disabling xticks/yticks is unnecessary, since we hide Axises.
        ax = self._fig.add_axes(
            [left, bottom, width, height], xticks=[], yticks=[], label=label
        )

        grid_color = cfg.grid_color
        if grid_color:
            # Initialize borders
            # Hide Axises
            # (drawing them is very slow, and we disable ticks+labels anyway)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Background color
            # ax.patch.set_fill(False) sets _fill=False,
            # then calls _set_facecolor(...) "alpha = self._alpha if self._fill else 0".
            # It is no faster than below.
            ax.set_facecolor(self.transparent)

            # Set border colors
            for spine in ax.spines.values():  # type: Spine
                spine.set_linewidth(cfg.grid_line_width)
                spine.set_color(grid_color)

            def hide(key: str):
                ax.spines[key].set_visible(False)

            # Hide all borders except bottom-right.
            hide("top")
            hide("left")

            # If bottom of screen, hide bottom. If right of screen, hide right.
            if r.screen_edges & Edges.Bottom:
                hide("bottom")
            if r.screen_edges & Edges.Right:
                hide("right")

            # If our Axes is a stereo track, dim borders between channels. (Show
            # borders between waves at full opacity.)
            if cfg.stereo_grid_opacity > 0:
                dim_color = matplotlib.colors.to_rgba_array(grid_color)[0]
                dim_color[-1] = cfg.stereo_grid_opacity

                def dim(key: str):
                    ax.spines[key].set_color(dim_color)

            else:
                dim = hide

            # If not bottom of wave, dim bottom. If not right of wave, dim right.
            if not r.wave_edges & Edges.Bottom:
                dim("bottom")
            if not r.wave_edges & Edges.Right:
                dim("right")

        else:
            ax.set_axis_off()

        return ax

    # Protected API
    def __add_lines_stereo(self, inputs: List[RenderInput]):
        cfg = self.cfg
        strides = self.render_strides

        # Plot lines over background
        line_width = cfg.line_width

        # Foreach wave, plot dummy data.
        lines2d = []
        wave_to_stereo_bar = []
        for wave_idx, input in enumerate(inputs):
            wave_data = input.data
            line_params = self._line_params[wave_idx]

            # [nsamp][nchan] Amplitude
            wave_zeros = np.zeros_like(wave_data)

            chan_to_axes = self._wave_chan_to_axes[wave_idx]
            wave_lines = []

            xs = calc_xs(len(wave_zeros), strides[wave_idx])
            line_color = line_params.color

            # Foreach chan
            for chan_idx, chan_zeros in enumerate(wave_zeros.T):
                ax = chan_to_axes[chan_idx]

                chan_line: Line2D = ax.plot(
                    xs, chan_zeros, color=line_color, linewidth=line_width
                )[0]

                if cfg.line_outline_width > 0:
                    chan_line.set_path_effects(
                        [
                            mpl.patheffects.Stroke(
                                linewidth=cfg.line_width + 2 * cfg.line_outline_width,
                                foreground=cfg.global_line_outline_color,
                            ),
                            mpl.patheffects.Normal(),
                        ]
                    )

                wave_lines.append(chan_line)

            lines2d.append(wave_lines)
            self._artists.extend(wave_lines)

            # Add stereo bars if enabled and track is stereo.
            if input.stereo_levels:
                assert self._line_params[wave_idx].stereo_bars
                ax = self._wave_to_mono_axes[wave_idx]

                viewport_stride = self.render_strides[wave_idx] * cfg.viewport_width
                x_center = calc_center(viewport_stride)

                xlim = ax.get_xlim()
                x_range = (xlim[1] - xlim[0]) / 2

                y_bottom = ax.get_ylim()[0]

                h = abs(y_bottom) / 16
                stereo_rect = Rectangle((x_center, y_bottom - h), 0, 2 * h)
                stereo_rect.set_color(cfg.stereo_bar_color)
                stereo_rect.set_linewidth(0)
                ax.add_patch(stereo_rect)

                stereo_bar = StereoBar(stereo_rect, x_center, x_range)

                wave_to_stereo_bar.append(stereo_bar)
                self._artists.append(stereo_rect)
            else:
                wave_to_stereo_bar.append(None)

        self._wave_chan_to_line = lines2d
        self._wave_to_stereo_bar = wave_to_stereo_bar

    def _update_lines_stereo(self, inputs: List[RenderInput]) -> None:
        """
        Preconditions:
        - inputs[wave] = ndarray, [samp][chan] = f32
        """
        if self._wave_chan_to_line is None:
            self.__add_lines_stereo(inputs)

        lines2d = self._wave_chan_to_line
        nplots = len(lines2d)
        ndata = len(inputs)
        if nplots != ndata:
            raise ValueError(
                f"incorrect data to plot: {nplots} plots but {ndata} dummy_datas"
            )

        # Draw waveform data
        # Foreach wave
        for wave_idx, input in enumerate(inputs):
            wave_data = input.data
            freq_estimate = input.freq_estimate

            wave_lines = lines2d[wave_idx]

            color_by_pitch = self._line_params[wave_idx].color_by_pitch

            # If we color notes by pitch, then on every frame,
            # recompute the color based on current pitch.
            # If no sound is detected, fall back to the default color.
            # If we don't color notes by pitch,
            # just keep the initial color and never overwrite it.
            if color_by_pitch:
                fallback_color = self._line_params[wave_idx].color
                color = freq_to_color(self.pitch_cmap, freq_estimate, fallback_color)

            # Foreach chan
            for chan_idx, chan_data in enumerate(wave_data.T):
                chan_line = wave_lines[chan_idx]
                chan_line.set_ydata(chan_data)
                if color_by_pitch:
                    chan_line.set_color(color)

            stereo_bar = self._wave_to_stereo_bar[wave_idx]
            stereo_levels = inputs[wave_idx].stereo_levels
            assert bool(stereo_bar) == bool(
                stereo_levels
            ), f"wave {wave_idx}: plot={stereo_bar} != values={stereo_levels}"
            if stereo_bar:
                stereo_bar.set_range(*stereo_levels)

    def _add_xy_line_mono(
        self,
        name: str,
        wave_idx: int,
        xs: Sequence[float],
        ys: Sequence[float],
        stride: int,
    ) -> CustomLine:
        """Add a debug line, which can be repositioned every frame."""
        cfg = self.cfg

        # Plot lines over background
        line_width = cfg.line_width

        ax = self._wave_to_mono_axes[wave_idx]
        mono_line: Line2D = ax.plot(xs, ys, linewidth=line_width)[0]
        print(f"{name} {wave_idx} has color {mono_line.get_color()}")

        self._artists.append(mono_line)

        # noinspection PyTypeChecker
        return CustomLine(stride, xs, mono_line.set_xdata, mono_line.set_ydata)

    # Channel labels
    def add_labels(self, labels: List[str]) -> List["Text"]:
        """
        Updates background, adds text.
        Do NOT call after calling self.add_lines().
        """
        nlabel = len(labels)
        if nlabel != self.nplots:
            raise ValueError(
                f"incorrect labels: {self.nplots} plots but {nlabel} labels"
            )

        cfg = self.cfg
        color = cfg.get_label_color

        size_pt = cfg.label_font.size
        distance_px = cfg.label_padding_ratio * size_pt

        @attr.dataclass
        class AxisPosition:
            pos_axes: float
            offset_px: float
            align: str

        xpos = cfg.label_position.x.match(
            left=AxisPosition(0, distance_px, "left"),
            right=AxisPosition(1, -distance_px, "right"),
        )
        ypos = cfg.label_position.y.match(
            bottom=AxisPosition(0, distance_px, "bottom"),
            top=AxisPosition(1, -distance_px, "top"),
        )

        pos_axes = (xpos.pos_axes, ypos.pos_axes)
        offset_pt = (xpos.offset_px, ypos.offset_px)

        out: List["Text"] = []
        for label_text, ax in zip(labels, self._wave_to_mono_axes):
            # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.annotate.html
            # Annotation subclasses Text.
            text: "Annotation" = ax.annotate(
                label_text,
                # Positioning
                xy=pos_axes,
                xycoords="axes fraction",
                xytext=offset_pt,
                textcoords="offset points",
                horizontalalignment=xpos.align,
                verticalalignment=ypos.align,
                # Cosmetics
                color=color,
                fontsize=px_from_points(size_pt),
                fontfamily=cfg.label_font.family,
                fontweight=("bold" if cfg.label_font.bold else "normal"),
                fontstyle=("italic" if cfg.label_font.italic else "normal"),
            )
            out.append(text)

        self._save_background()
        return out

    # Output frames
    def get_frame(self) -> ByteBuffer:
        """Returns bytes with shape (h, w, self.bytes_per_pixel).
        The actual return value's shape may be flat.
        """
        self._redraw_over_background()

        canvas = self._fig.canvas

        # Agg is the default noninteractive backend except on OSX.
        # https://matplotlib.org/faq/usage_faq.html
        if not isinstance(canvas, self._canvas_type):
            raise RuntimeError(
                f"oh shit, cannot read data from {obj_name(canvas)} != {self._canvas_type.__name__}"
            )

        buffer_rgb = self._canvas_to_bytes(canvas)
        assert len(buffer_rgb) == self.w * self.h * self.bytes_per_pixel

        return buffer_rgb

    # Pre-rendered background
    bg_cache: Any  # "matplotlib.backends._backend_agg.BufferRegion"

    def _save_background(self) -> None:
        """Draw static background."""
        # https://stackoverflow.com/a/8956211
        # https://matplotlib.org/api/animation_api.html#funcanimation
        fig = self._fig

        fig.canvas.draw()
        self.bg_cache = fig.canvas.copy_from_bbox(fig.bbox)

    def _redraw_over_background(self) -> None:
        """Redraw animated elements of the image."""

        # Both FigureCanvasAgg and FigureCanvasCairo, but not FigureCanvasBase,
        # support restore_region().
        canvas: FigureCanvasAgg = self._fig.canvas
        canvas.restore_region(self.bg_cache)

        for artist in self._artists:
            artist.axes.draw_artist(artist)

        # canvas.blit(self._fig.bbox) is unnecessary when drawing off-screen.


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

    # New methods.
    def update_main_lines(
        self, inputs: List[RenderInput], trigger_samples: List[int]
    ) -> None:
        datas = [input.data for input in inputs]

        self._update_lines_stereo(inputs)
        assert len(datas) == len(trigger_samples)
        for i, (data, trigger) in enumerate(zip(datas, trigger_samples)):
            self.move_viewport(i, trigger)  # - len(data) / 2

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
