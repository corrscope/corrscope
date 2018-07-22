from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ovgenpy.outputs import RGB_DEPTH
from ovgenpy.util import ceildiv

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D



@dataclass
class RendererConfig:
    width: int
    height: int

    nrows: Optional[int] = None
    ncols: Optional[int] = None

    def __post_init__(self):
        if not self.nrows:
            self.nrows = None
        if not self.ncols:
            self.ncols = None

        if self.nrows and self.ncols:
            raise ValueError('cannot manually assign both nrows and ncols')

        if not self.nrows and not self.ncols:
            self.ncols = 1


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

    def __init__(self, cfg: RendererConfig, nplots: int, create_window: bool):
        self.cfg = cfg
        self.nplots = nplots
        self.create_window = create_window

        # Setup layout
        # "ncols=1" is good for vertical layouts.
        # But "nrows=X" is good for left-to-right grids.

        self.nrows = 0
        self.ncols = 0

        # Flat array of nrows*ncols elements, ordered by cfg.rows_first.
        self.fig: 'Figure' = None
        self.axes: List['Axes'] = None        # set by set_layout()
        self.lines: List['Line2D'] = None     # set by render_frame() first call

        self.set_layout()   # mutates self

    def set_layout(self) -> None:
        """
        Creates a flat array of Matplotlib Axes, with the new layout.
        Opens a window showing the Figure (and Axes).

        Inputs: self.cfg, self.fig
        Outputs: self.nrows, self.ncols, self.axes
        """

        self.nrows, self.ncols = self._calc_layout()

        # Create Axes
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
        if self.fig:
            plt.close(self.fig)     # FIXME

        axes2d: np.ndarray['Axes']
        self.fig, axes2d = plt.subplots(
            self.nrows, self.ncols,
            squeeze=False,
            # Remove gaps between Axes
            gridspec_kw=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        )

        # remove Axis from Axes
        for ax in axes2d.flatten():
            ax.set_axis_off()

        # if column major:
        if self.cfg.ncols:
            axes2d = axes2d.T

        self.axes: List['Axes'] = axes2d.flatten().tolist()[:self.nplots]

        # Setup figure geometry
        self.fig.set_dpi(self.DPI)
        self.fig.set_size_inches(
            self.cfg.width / self.DPI,
            self.cfg.height / self.DPI
        )
        if self.create_window:
            plt.show(block=False)

    def _calc_layout(self) -> Tuple[int, int]:
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

    def render_frame(self, datas: List[np.ndarray]) -> None:
        ndata = len(datas)
        if self.nplots != ndata:
            raise ValueError(
                f'incorrect data to plot: {self.nplots} plots but {ndata} datas')

        # Initialize axes and draw waveform data
        if self.lines is None:
            self.lines = []
            for idx, data in enumerate(datas):
                ax = self.axes[idx]
                ax.set_xlim(0, len(data) - 1)
                ax.set_ylim(-1, 1)

                line = ax.plot(data)[0]
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
        np.reshape(buffer_rgb, (w, h, RGB_DEPTH))
        assert buffer_rgb.size == w * h * RGB_DEPTH

        return buffer_rgb
        # # TODO https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imsave.html to
        # # in-memory stream as png
        #
        # # or imsave(arr=...)
        #
        # # TODO http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
        #
        # raise NotImplementedError
