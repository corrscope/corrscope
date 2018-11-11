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
    def __init__(self, cfg: 'RendererConfig', layout: 'RendererLayout'):
        # (800, 600) is landscape.
        # x,y = w,h
        size = (cfg.width, cfg.height)
        app.Canvas.__init__(self, show=False, size=size)

        # Subplot layout
        self.nrows = layout.nrows
        self.ncols = layout.ncols

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
        yticks = fenceposts(self.physical_size[1], self.nrows)
        xticks = fenceposts(self.physical_size[0], self.ncols)

        for i, nsamp in enumerate(lines_nsamp):
            # Create line coordinates (x, y).
            # xs ranges from 0..1 inclusive.
            # ys ranges from -1..1 inclusive.

            line_coords = np.empty((nsamp, 2))
            line_coords[:, 0] = np.linspace(0, 1, nsamp)
            line_coords[:, 1] = 0

            line = visuals.LineVisual(pos=line_coords)  # TODO color, width

            # Set line position and size.
            x = ...
            y = ...

            line.transform = STTransform(translate=[x, y])

            # redraw the canvas if any visuals request an update
            line.events.update.connect(lambda evt: self.update())

            self.lines_coords.append(line_coords)
            self.lines_ys.append(line_coords[:, 1])
            self._lines.append(line)

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
            gloo.set_viewport(0, 0, *self.size)
            gloo.clear('black')
            for visual in self.visuals:
                visual.draw()
            self.im = _screenshot((0, 0, self.size[0], self.size[1]))


def fenceposts(max: int, n: int) -> 'np.ndarray[int]':
    """ Returns n+1 elements ranging from 0 to max, inclusive. """
    pts = np.linspace(0, max, n + 1)
    return pts.astype(int)
