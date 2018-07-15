from itertools import count
from typing import NamedTuple, Optional, List, Tuple, TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ovgenpy.util import ceildiv

if TYPE_CHECKING:
    from ovgenpy.wave import Wave


class RendererConfig(NamedTuple):
    width: int
    height: int

    samples_visible: int

    rows_first: bool

    nrows: Optional[int] = None     # TODO set to 1
    ncols: Optional[int] = None

    # TODO backend: FigureCanvasBase = FigureCanvasAgg


class MatplotlibRenderer:
    """
    TODO disable antialiasing
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

    def __init__(self, cfg: RendererConfig, waves: List['Wave']):
        self.cfg = cfg
        self.waves = waves
        self.nwaves = len(waves)
        self.fig: Figure = None

        # Setup layout

        self.nrows = 0
        self.ncols = 0
        # Flat array of nrows*ncols elements, ordered by cfg.rows_first.
        self.axes: List[Axes] = None
        self.lines: List[Line2D] = None

        self.set_layout()   # mutates self

    def set_layout(self) -> None:
        """
        Inputs: self.cfg, self.waves, self.fig
        Outputs: self.nrows, self.ncols, self.axes

        Creates a flat array of Matplotlib Axes, with the new layout.
        """

        self.nrows, self.ncols = self.calc_layout()

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
        if not self.cfg.rows_first:
            axes2d = axes2d.T

        self.axes: List[Axes] = axes2d.flatten().tolist()[:self.nwaves]

        # Create oscilloscope line objects
        self.lines = []
        for ax in self.axes:
            # Setup axes limits
            ax.set_xlim(0, self.cfg.samples_visible)
            ax.set_ylim(-1, 1)

            line = ax.plot([0] * self.cfg.samples_visible)[0]
            self.lines.append(line)

        # Setup figure geometry
        self.fig.set_dpi(self.DPI)
        self.fig.set_size_inches(
            self.cfg.width / self.DPI,
            self.cfg.height / self.DPI
        )
        plt.show(block=False)

    def calc_layout(self) -> Tuple[int, int]:
        """
        Inputs: self.cfg, self.waves
        :return: (nrows, ncols)
        """
        cfg = self.cfg

        if cfg.rows_first:
            nrows = cfg.nrows
            if nrows is None:
                raise ValueError('invalid cfg: rows_first is True and nrows is None')
            ncols = ceildiv(self.nwaves, nrows)
        else:
            ncols = cfg.ncols
            if ncols is None:
                raise ValueError('invalid cfg: rows_first is False and ncols is None')
            nrows = ceildiv(self.nwaves, ncols)

        return nrows, ncols

    def render_frame(self, center_smps: List[int]) -> None:
        ncenters = len(center_smps)
        if self.nwaves != ncenters:
            raise ValueError(
                f'incorrect wave offsets: {self.nwaves} waves but {ncenters} offsets')

        for idx, wave, center_smp in zip(count(), self.waves, center_smps):
            # Draw waveform data
            line = self.lines[idx]
            data = wave.get_around(center_smp, self.cfg.samples_visible)
            line.set_ydata(data)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
