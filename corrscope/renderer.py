import enum
import os
from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING, Any, Callable, TypeVar

import attr
import matplotlib  # do NOT import anything else until we call matplotlib.use().
import matplotlib.colors
import numpy as np

from corrscope.config import DumpableAttrs, with_units, TypedEnumDump
from corrscope.layout import (
    RendererLayout,
    LayoutConfig,
    unique_by_id,
    RegionSpec,
    Edges,
)
from corrscope.outputs import BYTES_PER_PIXEL, ByteBuffer
from corrscope.util import coalesce

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

matplotlib.use("agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.spines import Spine
    from matplotlib.text import Text, Annotation
    from corrscope.channel import ChannelConfig


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
    size: float = with_units("pt", default=20)
    # QFont implementation details
    toString: str = None


class RendererConfig(DumpableAttrs, always_dump="*"):
    width: int
    height: int
    line_width: float = with_units("px", default=1.5)
    grid_line_width: float = with_units("px", default=1.0)

    @property
    def divided_width(self):
        return round(self.width / self.res_divisor)

    @property
    def divided_height(self):
        return round(self.height / self.res_divisor)

    bg_color: str = "#000000"
    init_line_color: str = default_color()

    grid_color: Optional[str] = None
    stereo_grid_opacity: float = 0.5

    midline_color: Optional[str] = None
    v_midline: bool = False
    h_midline: bool = False

    # Label settings
    label_font: Font = attr.ib(factory=Font)

    label_position: LabelPosition = LabelPosition.LeftTop
    # The text will be located (label_padding_ratio * label_font.size) from the corner.
    label_padding_ratio: float = with_units("px/pt", default=0.5)
    label_color_override: Optional[str] = None

    @property
    def get_label_color(self):
        return coalesce(self.label_color_override, self.init_line_color)

    antialiasing: bool = True

    # Performance (skipped when recording to video)
    res_divisor: float = 1.0

    def __attrs_post_init__(self) -> None:
        # round(np.int32 / float) == np.float32, but we want int.
        assert isinstance(self.width, (int, float))
        assert isinstance(self.height, (int, float))

    def before_preview(self) -> None:
        """ Called *once* before preview. Does nothing. """
        pass

    def before_record(self) -> None:
        """ Called *once* before recording video. Eliminates res_divisor. """
        self.res_divisor = 1


@attr.dataclass
class LineParam:
    color: str


UpdateLines = Callable[[List[np.ndarray]], None]

# TODO rename to Plotter
class Renderer(ABC):
    def __init__(
        self,
        cfg: RendererConfig,
        lcfg: "LayoutConfig",
        dummy_datas: List[np.ndarray],
        channel_cfgs: Optional[List["ChannelConfig"]],
    ):
        self.cfg = cfg
        self.lcfg = lcfg

        self.w = cfg.divided_width
        self.h = cfg.divided_height

        self.nplots = len(dummy_datas)

        if self.nplots > 0:
            assert len(dummy_datas[0].shape) == 2, dummy_datas[0].shape
        self.wave_nsamps = [data.shape[0] for data in dummy_datas]
        self.wave_nchans = [data.shape[1] for data in dummy_datas]

        # Load line colors.
        if channel_cfgs is not None:
            if len(channel_cfgs) != self.nplots:
                raise ValueError(
                    f"cannot assign {len(channel_cfgs)} colors to {self.nplots} plots"
                )
            line_colors = [cfg.line_color for cfg in channel_cfgs]
        else:
            line_colors = [None] * self.nplots

        self._line_params = [
            LineParam(color=coalesce(color, cfg.init_line_color))
            for color in line_colors
        ]

    _update_main_lines: Optional[UpdateLines] = None

    def update_main_lines(self, datas: List[np.ndarray]) -> None:
        if self._update_main_lines is None:
            self._update_main_lines = self.add_lines(datas)

        self._update_main_lines(datas)

    @abstractmethod
    def add_lines(self, dummy_datas: List[np.ndarray]) -> UpdateLines:
        ...

    @abstractmethod
    def get_frame(self) -> ByteBuffer:
        ...

    @abstractmethod
    def add_labels(self, labels: List[str]) -> Any:
        ...


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


class MatplotlibRenderer(Renderer):
    """
    Renderer backend which takes data and produces images.
    Does not touch Wave or Channel.

    If __init__ reads cfg, cfg cannot be hotswapped.

    Reasons to hotswap cfg: RendererCfg:
    - GUI preview size
    - Changing layout
    - Changing #smp drawn (samples_visible)
    (see RendererCfg)

        Original OVGen does not support hotswapping.
        It disables changing options during rendering.

    Reasons to hotswap trigger algorithms:
    - changing scan_nsamp (cannot be hotswapped, since correlation buffer is incompatible)
    So don't.
    """

    def __init__(self, *args, **kwargs):
        Renderer.__init__(self, *args, **kwargs)

        dict.__setitem__(
            matplotlib.rcParams, "lines.antialiased", self.cfg.antialiasing
        )

        self._setup_axes(self.wave_nchans)

        self._artists: List["Artist"] = []

    _fig: "Figure"

    # _axes2d[wave][chan] = Axes
    # Primary, used to draw oscilloscope lines and gridlines.
    _axes2d: List[List["Axes"]]  # set by set_layout()

    # _axes_mono[wave] = Axes
    # Secondary, used for titles and debug plots.
    _axes_mono: List["Axes"]

    def _setup_axes(self, wave_nchans: List[int]) -> None:
        """
        Creates a flat array of Matplotlib Axes, with the new layout.
        Sets up each Axes with correct region limits.
        """

        self.layout = RendererLayout(self.lcfg, wave_nchans)
        self.layout_mono = RendererLayout(self.lcfg, [1] * self.nplots)

        if hasattr(self, "_fig"):
            raise Exception("I don't currently expect to call _setup_axes() twice")
            # plt.close(self.fig)

        cfg = self.cfg

        self._fig = Figure()
        FigureCanvasAgg(self._fig)

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

        # Setup background
        self._fig.set_facecolor(cfg.bg_color)

        # Create Axes (using self.lcfg, wave_nchans)
        # _axes2d[wave][chan] = Axes
        self._axes2d = self.layout.arrange(self._axes_factory)

        """
        Adding an axes using the same arguments as a previous axes
        currently reuses the earlier instance.
        In a future version, a new instance will always be created and returned.
        Meanwhile, this warning can be suppressed, and the future behavior ensured,
        by passing a unique label to each axes instance.

        ax=fig.add_axes(label=) is unused, even if you call ax.legend().
        """
        # _axes_mono[wave] = Axes
        self._axes_mono = []
        # Returns 2D list of [self.nplots][1]Axes.
        axes_mono_2d = self.layout_mono.arrange(self._axes_factory, label="mono")
        for axes_list in axes_mono_2d:
            assert len(axes_list) == 1
            self._axes_mono.extend(axes_list)

        # Setup axes
        for idx, N in enumerate(self.wave_nsamps):
            wave_axes = self._axes2d[idx]
            max_x = N - 1

            def scale_axes(ax: "Axes"):
                ax.set_xlim(0, max_x)
                ax.set_ylim(-1, 1)

            scale_axes(self._axes_mono[idx])
            for ax in unique_by_id(wave_axes):
                scale_axes(ax)

                # Setup midlines (depends on max_x and wave_data)
                midline_color = cfg.midline_color
                midline_width = cfg.grid_line_width

                # Not quite sure if midlines or gridlines draw on top
                kw = dict(color=midline_color, linewidth=midline_width)
                if cfg.v_midline:
                    # See Wave.get_around() docstring.
                    # wave_data[N//2] == self[sample], usually > 0.
                    ax.axvline(x=N // 2 - 0.5, **kw)
                if cfg.h_midline:
                    ax.axhline(y=0, **kw)

        self._save_background()

    transparent = "#00000000"

    # satisfies RegionFactory
    def _axes_factory(self, r: RegionSpec, label: str = "") -> "Axes":
        cfg = self.cfg

        width = 1 / r.ncol
        left = r.col / r.ncol
        assert 0 <= left < 1

        height = 1 / r.nrow
        bottom = (r.nrow - r.row - 1) / r.nrow
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

            # Hide all axes except bottom-right.
            hide("top")
            hide("left")

            # If bottom of screen, hide bottom. If right of screen, hide right.
            if r.screen_edges & Edges.Bottom:
                hide("bottom")
            if r.screen_edges & Edges.Right:
                hide("right")

            # Dim stereo gridlines
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

    # Public API
    def add_lines(self, dummy_datas: List[np.ndarray]) -> UpdateLines:
        cfg = self.cfg

        # Plot lines over background
        line_width = cfg.line_width

        # Foreach wave, plot dummy data.
        lines2d = []
        for wave_idx, wave_data in enumerate(dummy_datas):
            wave_zeros = np.zeros_like(wave_data)

            wave_axes = self._axes2d[wave_idx]
            wave_lines = []

            # Foreach chan
            for chan_idx, chan_zeros in enumerate(wave_zeros.T):
                ax = wave_axes[chan_idx]
                line_color = self._line_params[wave_idx].color
                chan_line: Line2D = ax.plot(
                    chan_zeros, color=line_color, linewidth=line_width
                )[0]
                wave_lines.append(chan_line)

            lines2d.append(wave_lines)
            self._artists.extend(wave_lines)

        return lambda datas: self._update_lines(lines2d, datas)

    @staticmethod
    def _update_lines(lines2d: "List[List[Line2D]]", datas: List[np.ndarray]) -> None:
        """
        Preconditions:
        - lines2d[wave][chan] = Line2D
        - datas[wave] = ndarray, [samp][chan] = FLOAT
        """
        nplots = len(lines2d)
        ndata = len(datas)
        if nplots != ndata:
            raise ValueError(
                f"incorrect data to plot: {nplots} plots but {ndata} dummy_datas"
            )

        # Draw waveform data
        # Foreach wave
        for wave_idx, wave_data in enumerate(datas):
            wave_lines = lines2d[wave_idx]

            # Foreach chan
            for chan_idx, chan_data in enumerate(wave_data.T):
                chan_line = wave_lines[chan_idx]
                chan_line.set_ydata(chan_data)

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
        for label_text, ax in zip(labels, self._axes_mono):
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
        """ Returns ndarray of shape w,h,3. """
        self._redraw_over_background()

        canvas = self._fig.canvas

        # Agg is the default noninteractive backend except on OSX.
        # https://matplotlib.org/faq/usage_faq.html
        if not isinstance(canvas, FigureCanvasAgg):
            raise RuntimeError(
                f"oh shit, cannot read data from {type(canvas)} != FigureCanvasAgg"
            )

        buffer_rgb = canvas.tostring_rgb()
        assert len(buffer_rgb) == self.w * self.h * BYTES_PER_PIXEL

        return buffer_rgb

    # Pre-rendered background
    bg_cache: Any  # "matplotlib.backends._backend_agg.BufferRegion"

    def _save_background(self) -> None:
        """ Draw static background. """
        # https://stackoverflow.com/a/8956211
        # https://matplotlib.org/api/animation_api.html#funcanimation
        fig = self._fig

        fig.canvas.draw()
        self.bg_cache = fig.canvas.copy_from_bbox(fig.bbox)

    def _redraw_over_background(self) -> None:
        """ Redraw animated elements of the image. """

        canvas: FigureCanvasAgg = self._fig.canvas
        canvas.restore_region(self.bg_cache)

        for artist in self._artists:
            artist.axes.draw_artist(artist)

        # canvas.blit(self._fig.bbox) is unnecessary when drawing off-screen.
