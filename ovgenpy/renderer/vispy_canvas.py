from typing import Tuple, TYPE_CHECKING, List

import numpy as np
from vispy import app, gloo, visuals
from vispy.gloo.util import _screenshot
from vispy.visuals.transforms import STTransform

from ovgenpy.utils.keyword_dataclasses import dataclass

if TYPE_CHECKING:
    from ovgenpy.renderer import RendererConfig
    from ovgenpy.layout import RendererLayout

RGBA_DEPTH = 4


@dataclass
class CanvasConfig:
    width: int
    height: int

    nrows: int
    ncols: int

    # TODO colors


# TODO move into .renderer
class MyCanvas(app.Canvas):

    # self._fig, axes2d = plt.subplots(self.layout.nrows, self.layout.ncols...)
    def __init__(self, cfg: 'CanvasConfig'):
        self.cfg = cfg

        # (800, 600) is landscape.
        # x,y = w,h
        size = (cfg.width, cfg.height)
        app.Canvas.__init__(self, show=False, size=size)

        # Texture where we render the scene.
        self.rendertex = gloo.Texture2D(shape=self.size + (RGBA_DEPTH,))
        # FBO.
        self.fbo = gloo.FrameBuffer(self.rendertex, gloo.RenderBuffer(self.size))

    # lines_coords[chan] is a 2D ndarray.
    # lines_coords[chan][0] = xs, precomputed in create_lines()
    # lines_coords[chan][1] = ys, updated once per frame.
    lines_coords: List[np.ndarray]

    # lines_ys[chan] = lines_coords[chan][1]
    lines_ys: List[np.ndarray]

    # Vispy line objects.
    _lines: List[visuals.LineVisual]
    # All draw()able Vispy elements.
    visuals: list

    def create_lines(self, lines_nsamp: List[int]):
        self.lines_coords = []
        self.lines_ys = []
        self._lines = []

        # A bit confusing, be sure to check for bugs.

        for i, nsamp in enumerate(lines_nsamp):
            # Create line coordinates (x, y).
            line_coords = np.empty((nsamp, 2))
            self.lines_coords.append(line_coords)

            # xs ranges from 0..1 inclusive.
            line_coords[:, 0] = np.linspace(0, 1, nsamp)

            # ys ranges from -1..1 inclusive.
            line_coords[:, 1] = 0
            self.lines_ys.append(line_coords[:, 1])

            # Create line.
            line = visuals.LineVisual(pos=line_coords)  # TODO color, width
            self._lines.append(line)

            # Compute proper grid coordinate.
            x = ...
            y = ...

            line.transform = STTransform(translate=[x, y])

            # redraw the canvas if any visuals request an update
            line.events.update.connect(lambda evt: self.update())


        # All drawable elements.
        self.visuals = self._lines
        self.on_resize(None)

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        for visual in self.visuals:
            visual.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, event):
        """ Called by canvas.events.draw(). """
        with self.fbo:
            # TODO why is `set_viewport` redundant with `on_resize` above?
            gloo.set_viewport(0, 0, *self.size)
            gloo.clear('black')
            for visual in self.visuals:
                visual.draw()
            self.im = _screenshot((0, 0, self.size[0], self.size[1]))


def grid(cfg: 'CanvasConfig') -> List[STTransform]:
    """
    Generates a row-major grid of transformations.
    Compare with Matplotlib Figure.subplots().

    vispy coords:
    top left = (0,0)
    coord = (x, y) = (right, down)
    TODO
    """
    transforms = []

    xticks = fenceposts(cfg.width, cfg.ncols)
    widths = np.diff(xticks)

    yticks = fenceposts(cfg.height, cfg.nrows)
    heights = np.diff(yticks)

    # Matplotlib uses GridSpec to generate coordinates.
    for yidx in range(cfg.nrows):
        y = yticks[yidx]
        height = heights[yidx]
        # ys: Rescale -1 to y+height-1, 1 to y(+1?)
        yscale = -(height//2 - 1)
        y += height//2

        for xidx in range(cfg.ncols):
            x = xticks[xidx]
            width = widths[xidx]
            # xs: Rescale 0 to x, 1 to x+width-1.
            xscale = width

            tf = STTransform(scale=(xscale, yscale), translate=(x, y))
            transforms.append(tf)

    return transforms

def fenceposts(stop: int, n: int) -> 'np.ndarray[int]':
    """ Returns n+1 elements ranging from 0 to stop, inclusive. """
    pts = np.linspace(0, stop, n + 1)
    return pts.astype(int)
