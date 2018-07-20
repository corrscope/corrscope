from typing import Optional, List, Tuple

import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ovgenpy.util import ceildiv


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

    def __init__(self, cfg: RendererConfig, nplots: int):
        self.cfg = cfg
        self.nplots = nplots
        self.fig: Figure = None

        # Setup layout
        # "ncols=1" is good for vertical layouts.
        # But "nrows=X" is good for left-to-right grids.

        self.nrows = 0
        self.ncols = 0

        # Flat array of nrows*ncols elements, ordered by cfg.rows_first.
        self.axes: List[Axes] = None        # set by set_layout()
        self.lines: List[Line2D] = None     # set by render_frame() first call

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

        axes2d: np.ndarray[Axes]
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

        self.axes: List[Axes] = axes2d.flatten().tolist()[:self.nplots]

        # Setup figure geometry
        self.fig.set_dpi(self.DPI)
        self.fig.set_size_inches(
            self.cfg.width / self.DPI,
            self.cfg.height / self.DPI
        )
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
