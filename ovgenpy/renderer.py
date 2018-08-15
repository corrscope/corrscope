from typing import Optional, List, TYPE_CHECKING, TypeVar, Callable, Any

import matplotlib
import numpy as np

from ovgenpy.config import register_config
from ovgenpy.outputs import RGB_DEPTH
from ovgenpy.util import ceildiv, coalesce

matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from ovgenpy.channel import ChannelConfig


@register_config(always_dump='init_bg_color init_line_color line_width')
class RendererConfig:
    width: int
    height: int

    init_bg_color: Any = 'black'
    init_line_color: Any = 'white'
    # line_width: Optional[float] = None  # TODO

    create_window: bool = False


class MatplotlibRenderer:
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

    def __init__(self, cfg: RendererConfig, lcfg: 'LayoutConfig', nplots: int):
        self.cfg = cfg
        self.nplots = nplots
        self.layout = RendererLayout(lcfg, nplots)

        # Flat array of nrows*ncols elements, ordered by cfg.rows_first.
        self.fig: 'Figure' = None
        self.axes: List['Axes'] = None        # set by set_layout()
        self.lines: List['Line2D'] = None     # set by render_frame() first call

        self.bg_colors: List = [None] * nplots
        self.line_colors: List = [None] * nplots

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
        if self.fig:
            raise Exception("I don't currently expect to call set_layout() twice")
            plt.close(self.fig)

        axes2d: np.ndarray['Axes']
        self.fig, axes2d = plt.subplots(
            self.layout.nrows, self.layout.ncols,
            squeeze=False,
            # Remove gaps between Axes
            gridspec_kw=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        )

        # remove Axis from Axes
        for ax in axes2d.flatten():
            ax.set_axis_off()   # FIXME ax.get_xaxis().set_visible(False)

        # Generate arrangement (using nplots, cfg.orientation)
        self.axes = self.layout.arrange(lambda row, col: axes2d[row, col])

        # Setup figure geometry
        self.fig.set_dpi(self.DPI)
        self.fig.set_size_inches(
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
        self.bg_colors = [coalesce(cfg.bg_color, self.cfg.init_bg_color)
                          for cfg in channel_cfgs]
        self.line_colors = [coalesce(cfg.line_color, self.cfg.init_line_color)
                            for cfg in channel_cfgs]

    def render_frame(self, datas: List[np.ndarray]) -> None:
        ndata = len(datas)
        if self.nplots != ndata:
            raise ValueError(
                f'incorrect data to plot: {self.nplots} plots but {ndata} datas')

        # Initialize axes and draw waveform data
        if self.lines is None:
            self.lines = []
            for idx, data in enumerate(datas):
                # Setup colors
                bg_color = self.bg_colors[idx]
                line_color = self.line_colors[idx]

                # Setup axes
                ax = self.axes[idx]
                ax.set_xlim(0, len(data) - 1)
                ax.set_ylim(-1, 1)
                ax.set_facecolor(bg_color)

                # Plot line
                line = ax.plot(data, color=line_color)[0]
                self.lines.append(line)

        # Draw waveform data
        else:
            for idx, data in enumerate(datas):
                line = self.lines[idx]
                line.set_ydata(data)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_frame(self) -> np.ndarray:
        """ Returns ndarray of shape w,h,3. """
        canvas = self.fig.canvas

        # Agg is the default noninteractive backend except on OSX.
        # https://matplotlib.org/faq/usage_faq.html
        if not isinstance(canvas, FigureCanvasAgg):
            raise RuntimeError(
                f'oh shit, cannot read data from {type(canvas)} != FigureCanvasAgg')

        w = self.cfg.width
        h = self.cfg.height
        assert (w, h) == canvas.get_width_height()

        buffer_rgb: np.ndarray = np.frombuffer(canvas.tostring_rgb(), np.uint8)     # TODO Pycharm type inference error
        assert buffer_rgb.size == w * h * RGB_DEPTH

        return buffer_rgb


@register_config
class LayoutConfig:
    nrows: Optional[int] = None
    ncols: Optional[int] = None
    orientation: str = 'h'

    def __post_init__(self):
        if not self.nrows:
            self.nrows = None
        if not self.ncols:
            self.ncols = None

        if self.nrows and self.ncols:
            raise ValueError('cannot manually assign both nrows and ncols')

        if not self.nrows and not self.ncols:
            self.ncols = 1


Region = TypeVar('Region')
RegionFactory = Callable[[int, int], Region]   # f(row, column) -> Region


class RendererLayout:
    VALID_ORIENTATIONS = ['h', 'v']

    def __init__(self, cfg: LayoutConfig, nplots: int):
        self.cfg = cfg
        self.nplots = nplots

        # Setup layout
        self.nrows, self.ncols = self._calc_layout()

        self.orientation = cfg.orientation
        if self.orientation not in self.VALID_ORIENTATIONS:
            raise ValueError(f'Invalid orientation {self.orientation} not in '
                             f'{self.VALID_ORIENTATIONS}')

    def _calc_layout(self):
        """
        Inputs: self.cfg, self.waves
        :return: (nrows, ncols)
        """
        cfg = self.cfg

        if cfg.nrows:
            nrows = cfg.nrows
            if nrows is None:
                raise ValueError('invalid cfg: rows_first is True and nrows is None')
            ncols = ceildiv(self.nplots, nrows)
        else:
            ncols = cfg.ncols
            if ncols is None:
                raise ValueError('invalid cfg: rows_first is False and ncols is None')
            nrows = ceildiv(self.nplots, ncols)

        return nrows, ncols

    def arrange(self, region_factory: RegionFactory) -> List[Region]:
        """ Generates an array of regions.

        index, row, column are fed into region_factory in a row-major order [row][col].
        The results are possibly reshaped into column-major order [col][row].
        """
        nspaces = self.nrows * self.ncols
        inds = np.arange(nspaces)
        rows, cols = np.unravel_index(inds, (self.nrows, self.ncols))

        row_col = np.array([rows, cols]).T
        regions = np.array([region_factory(*rc) for rc in row_col])  # type: np.ndarray[Region]
        regions2d = regions.reshape((self.nrows, self.ncols))   # type: np.ndarray[Region]

        # if column major:
        if self.orientation == 'v':
            regions2d = regions2d.T

        return regions2d.flatten()[:self.nplots].tolist()
