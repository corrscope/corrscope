import os
from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

import attr
import matplotlib
import numpy as np

from corrscope.config import register_config
from corrscope.layout import RendererLayout, LayoutConfig, EdgeFinder
from corrscope.outputs import RGB_DEPTH, ByteBuffer
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

corrscope uses one-folder mode, does not use fonts yet,
and deletes all matplotlib-bundled fonts to save space. So reenable global font cache.
"""

mpl_config_dir = "MPLCONFIGDIR"
if mpl_config_dir in os.environ:
    del os.environ[mpl_config_dir]

matplotlib.use("agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from corrscope.channel import ChannelConfig


def default_color():
    # import matplotlib.colors
    # colors = np.array([int(x, 16) for x in '1f 77 b4'.split()], dtype=float)
    # colors /= np.amax(colors)
    # colors **= 1/3
    #
    # return matplotlib.colors.to_hex(colors, keep_alpha=False)
    return "#8edeff"


@register_config(always_dump="*")
class RendererConfig:
    width: int
    height: int
    line_width: float = 1.5

    bg_color: str = "#000000"
    init_line_color: str = default_color()
    grid_color: Optional[str] = None


@attr.dataclass
class LineParam:
    color: str


# TODO rename to Plotter
class Renderer(ABC):
    def __init__(
        self,
        cfg: RendererConfig,
        lcfg: "LayoutConfig",
        nplots: int,
        channel_cfgs: Optional[List["ChannelConfig"]],
    ):
        self.cfg = cfg
        self.nplots = nplots
        self.layout = RendererLayout(lcfg, nplots)

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

    @abstractmethod
    def render_frame(self, datas: List[np.ndarray]) -> None:
        ...

    @abstractmethod
    def get_frame(self) -> ByteBuffer:
        ...


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

    DPI = 96

    def __init__(self, *args, **kwargs):
        Renderer.__init__(self, *args, **kwargs)

        # Flat array of nrows*ncols elements, ordered by cfg.rows_first.
        self._fig: "Figure"
        self._axes: List["Axes"]  # set by set_layout()
        self._lines: Optional[List["Line2D"]] = None  # set by render_frame() first call

        self._set_layout()  # mutates self

    transparent = "#00000000"

    def _set_layout(self) -> None:
        """
        Creates a flat array of Matplotlib Axes, with the new layout.
        Opens a window showing the Figure (and Axes).

        Inputs: self.cfg, self.fig
        Outputs: self.nrows, self.ncols, self.axes
        """

        # Create Axes
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
        if hasattr(self, "_fig"):
            raise Exception("I don't currently expect to call _set_layout() twice")
            # plt.close(self.fig)

        grid_color = self.cfg.grid_color
        axes2d: np.ndarray["Axes"]
        self._fig = Figure()
        FigureCanvasAgg(self._fig)

        axes2d = self._fig.subplots(
            self.layout.nrows,
            self.layout.ncols,
            squeeze=False,
            # Remove axis ticks (which slow down rendering)
            subplot_kw=dict(xticks=[], yticks=[]),
            # Remove gaps between Axes TODO borders shouldn't be half-visible
            gridspec_kw=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0),
        )

        ax: "Axes"
        if grid_color:
            # Initialize borders
            for ax in axes2d.flatten():
                # Hide Axises
                # (drawing them is very slow, and we disable ticks+labels anyway)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Background color
                ax.set_facecolor(self.transparent)

                # Set border colors
                for spine in ax.spines.values():
                    spine.set_color(grid_color)

                # gridspec_kw indexes from bottom-left corner.
                # Only show bottom-left borders (x=0, y=0)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # Hide bottom-left edges for speed.
            edge_axes: EdgeFinder["Axes"] = EdgeFinder(axes2d)
            for ax in edge_axes.bottoms:
                ax.spines["bottom"].set_visible(False)
            for ax in edge_axes.lefts:
                ax.spines["left"].set_visible(False)

        else:
            # Remove Axis from Axes
            for ax in axes2d.flatten():
                ax.set_axis_off()

        # Generate arrangement (using nplots, cfg.orientation)
        self._axes = self.layout.arrange(lambda row, col: axes2d[row, col])

        # Setup figure geometry
        self._fig.set_dpi(self.DPI)
        self._fig.set_size_inches(self.cfg.width / self.DPI, self.cfg.height / self.DPI)

    def render_frame(self, datas: List[np.ndarray]) -> None:
        ndata = len(datas)
        if self.nplots != ndata:
            raise ValueError(
                f"incorrect data to plot: {self.nplots} plots but {ndata} datas"
            )

        # Initialize axes and draw waveform data
        if self._lines is None:
            self._fig.set_facecolor(self.cfg.bg_color)
            line_width = self.cfg.line_width

            self._lines = []
            for idx, data in enumerate(datas):
                # Setup colors
                line_param = self._line_params[idx]
                line_color = line_param.color

                # Setup axes
                ax = self._axes[idx]
                ax.set_xlim(0, len(data) - 1)
                ax.set_ylim(-1, 1)

                # Plot line
                line = ax.plot(data, color=line_color, linewidth=line_width)[0]
                self._lines.append(line)

        # Draw waveform data
        else:
            for idx, data in enumerate(datas):
                line = self._lines[idx]
                line.set_ydata(data)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def get_frame(self) -> ByteBuffer:
        """ Returns ndarray of shape w,h,3. """
        canvas = self._fig.canvas

        # Agg is the default noninteractive backend except on OSX.
        # https://matplotlib.org/faq/usage_faq.html
        if not isinstance(canvas, FigureCanvasAgg):
            raise RuntimeError(
                f"oh shit, cannot read data from {type(canvas)} != FigureCanvasAgg"
            )

        w = self.cfg.width
        h = self.cfg.height
        assert (w, h) == canvas.get_width_height()

        buffer_rgb = canvas.tostring_rgb()
        assert len(buffer_rgb) == w * h * RGB_DEPTH

        return buffer_rgb
