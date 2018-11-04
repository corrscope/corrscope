from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING, Any

import matplotlib
import numpy as np

from ovgenpy.config import register_config
from ovgenpy.layout import RendererLayout, LayoutConfig
from ovgenpy.outputs import RGB_DEPTH
from ovgenpy.util import coalesce

matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from ovgenpy.channel import ChannelConfig


def default_color():
    colors = np.array([int(x, 16) for x in '1f 77 b4'.split()], dtype=float)
    colors /= np.amax(colors)
    colors **= 1/3

    return tuple(colors.tolist())   # tolist() converts np.float64 to float


@register_config(always_dump='bg_color init_line_color line_width')
class RendererConfig:
    width: int
    height: int

    bg_color: Any = 'black'
    init_line_color: Any = default_color()
    line_width: Optional[float] = None  # TODO

    create_window: bool = False


class Renderer(ABC):
    def __init__(self, cfg: RendererConfig, lcfg: 'LayoutConfig', nplots: int):
        self.cfg = cfg
        self.nplots = nplots
        self.layout = RendererLayout(lcfg, nplots)

    @abstractmethod
    def set_colors(self, channel_cfgs: List['ChannelConfig']) -> None: ...

    @abstractmethod
    def render_frame(self, datas: List[np.ndarray]) -> None: ...

    @abstractmethod
    def get_frame(self) -> bytes: ...


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
        self._fig: 'Figure' = None
        self._axes: List['Axes'] = None        # set by set_layout()
        self._lines: List['Line2D'] = None     # set by render_frame() first call

        self._line_colors: List = [None] * self.nplots

        self._set_layout()   # mutates self

    def _set_layout(self) -> None:
        """
        Creates a flat array of Matplotlib Axes, with the new layout.
        Opens a window showing the Figure (and Axes).

        Inputs: self.cfg, self.fig
        Outputs: self.nrows, self.ncols, self.axes
        """

        # Create Axes
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
        if self._fig:
            raise Exception("I don't currently expect to call set_layout() twice")
            # plt.close(self.fig)

        axes2d: np.ndarray['Axes']
        self._fig, axes2d = plt.subplots(
            self.layout.nrows, self.layout.ncols,
            squeeze=False,
            # Remove gaps between Axes
            gridspec_kw=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        )

        # remove Axis from Axes
        for ax in axes2d.flatten():
            ax.set_axis_off()

        # Generate arrangement (using nplots, cfg.orientation)
        self._axes = self.layout.arrange(lambda row, col: axes2d[row, col])

        # Setup figure geometry
        self._fig.set_dpi(self.DPI)
        self._fig.set_size_inches(
            self.cfg.width / self.DPI,
            self.cfg.height / self.DPI
        )
        if self.cfg.create_window:
            plt.show(block=False)

    def set_colors(self, channel_cfgs: List['ChannelConfig']):
        if len(channel_cfgs) != self.nplots:
            raise ValueError(
                f"cannot assign {len(channel_cfgs)} colors to {self.nplots} plots"
            )

        if self._lines is not None:
            raise ValueError(
                f'cannot set line colors after calling render_frame()'
            )
        self._line_colors = [cfg.line_color for cfg in channel_cfgs]

    def render_frame(self, datas: List[np.ndarray]) -> None:
        ndata = len(datas)
        if self.nplots != ndata:
            raise ValueError(
                f'incorrect data to plot: {self.nplots} plots but {ndata} datas')

        # Initialize axes and draw waveform data
        if self._lines is None:
            self._fig.set_facecolor(self.cfg.bg_color)
            line_width = self.cfg.line_width

            self._lines = []
            for idx, data in enumerate(datas):
                # Setup colors
                line_color = coalesce(self._line_colors[idx], self.cfg.init_line_color)

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

    def get_frame(self) -> bytes:
        """ Returns ndarray of shape w,h,3. """
        canvas = self._fig.canvas

        # Agg is the default noninteractive backend except on OSX.
        # https://matplotlib.org/faq/usage_faq.html
        if not isinstance(canvas, FigureCanvasAgg):
            raise RuntimeError(
                f'oh shit, cannot read data from {type(canvas)} != FigureCanvasAgg')

        w = self.cfg.width
        h = self.cfg.height
        assert (w, h) == canvas.get_width_height()

        buffer_rgb = canvas.tostring_rgb()
        assert len(buffer_rgb) == w * h * RGB_DEPTH

        return buffer_rgb

