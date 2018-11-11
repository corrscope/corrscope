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
class CanvasParam:
    """ Unlike other config classes, this is internal and not saved to YAML. """
    width: int
    height: int

    # nrows: int
    # ncols: int

    # TODO colors


# TODO move into .renderer
class MyCanvas(app.Canvas):
    """ Canvas which draws a grid of lines.
    Modify `lines_ys` to change plotted data.
    """

    # self._fig, axes2d = plt.subplots(self.layout.nrows, self.layout.ncols...)
    def __init__(self, cfg: 'CanvasParam'):
        self.cfg = cfg

        # (800, 600) is landscape.
        # x,y = w,h
        size = (cfg.width, cfg.height)
        app.Canvas.__init__(self, show=False, size=size)

        # Texture where we render the scene.
        self._rendertex = gloo.Texture2D(shape=self.size + (RGBA_DEPTH,))
        # FBO.
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(self.size))

    # lines_ys[chan] = lines_coords[chan][1]
    _lines_ys: List[np.ndarray] = None

    def set_ys(self, lines_ys: List[np.ndarray]):
        """ Assigns a list of ydata. """
        for i, ys in enumerate(lines_ys):
            self._lines_ys[i] = lines_ys
        self.update()   # TODO ????

    # lines_coords[chan] is a 2D ndarray.
    # lines_coords[chan][0] = xs, precomputed in create_lines()
    # lines_coords[chan][1] = ys, updated once per frame.
    _lines_coords: List[np.ndarray]

    # Vispy line objects.
    _lines: List[visuals.LineVisual]
    # All draw()able Vispy elements.
    _visuals: list

    def create_lines(self, lines_nsamp: List[int], layout: RendererLayout):
        self._lines_coords = []
        self._lines_ys = []
        self._lines = []

        # 1D list of Vispy transforms, satisfying layout.
        transforms = transform_grid(self.cfg, layout)

        for i, nsamp in enumerate(lines_nsamp):
            # Create line coordinates (x, y).
            line_coords = np.empty((nsamp, 2))
            self._lines_coords.append(line_coords)

            # xs ranges from 0..1 inclusive.
            line_coords[:, 0] = np.linspace(0, 1, nsamp)

            # ys ranges from -1..1 inclusive.
            line_coords[:, 1] = 0
            self._lines_ys.append(line_coords[:, 1])

            # Create line and transform to correct position.
            line = visuals.LineVisual(pos=line_coords)  # TODO color, width
            line.transform = transforms[i]
            self._lines.append(line)

            # redraw the canvas if any visuals request an update
            # TODO unneeded??? line.events.update.connect(lambda evt: self.update())

        self._visuals = self._lines
        self.on_resize(None)

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        for visual in self._visuals:
            visual.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, event):
        """ Called by canvas.events.draw(). """
        with self._fbo:
            # TODO why is `set_viewport` redundant with `on_resize` above?
            gloo.set_viewport(0, 0, *self.size)
            gloo.clear('black')
            for visual in self._visuals:
                visual.draw()
            self.im = _screenshot((0, 0, self.size[0], self.size[1]))


def transform_grid(cfg: 'CanvasParam', layout: 'RendererLayout') -> List[STTransform]:
    """
    Returns a 1D list of Vispy transforms, satisfying the layout given.
    Compare with Matplotlib Figure.subplots().

    vispy coords:
    top left = (0,0)
    coord = (x, y) = (right, down)
    TODO
    """

    # below[row,col] = transform
    transforms = np.empty((layout.nrows, layout.ncols), object)  # type: np.ndarray[STTransform]

    xticks = fenceposts(cfg.width, cfg.ncols)
    widths = np.diff(xticks)

    yticks = fenceposts(cfg.height, cfg.nrows)
    heights = np.diff(yticks)

    for yrow in range(layout.nrows):
        y = yticks[yrow]
        height = heights[yrow]
        # ys: Rescale -1 to y+height-1, 1 to y(+1?)
        yscale = -(height//2 - 1)
        y += height//2

        for xcol in range(layout.ncols):
            x = xticks[xcol]
            width = widths[xcol]
            # xs: Rescale 0 to x, 1 to x+width-1.
            xscale = width

            tf = STTransform(scale=(xscale, yscale), translate=(x, y))
            transforms[yrow, xcol] = tf

    # Now apply `layout`.
    arrangement = layout.arrange(lambda row, col: transforms[row, col])
    return arrangement


def fenceposts(stop: int, n: int) -> 'np.ndarray[int]':
    """ Returns n+1 elements ranging from 0 to stop, inclusive. """
    pts = np.linspace(0, stop, n + 1)
    return pts.astype(int)
