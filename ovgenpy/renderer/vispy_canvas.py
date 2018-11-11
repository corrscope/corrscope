from typing import TYPE_CHECKING, List

import numpy as np
from vispy import app, gloo, visuals
from vispy.gloo.util import _screenshot
from vispy.visuals.transforms import STTransform

import ovgenpy.renderer
from ovgenpy.utils.keyword_dataclasses import dataclass

if TYPE_CHECKING:
    from ovgenpy.layout import RendererLayout

RGBA_DEPTH = 4


# Removing this line results in stderr messages:
# WARNING: QOpenGLContext::swapBuffers() called with non-exposed window, behavior is undefined
# IDK if pyglet also invokes undefined behavior under circumstances.
app.use_app('pyglet')


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
        app.Canvas.__init__(self, show=False, size=size, dpi=ovgenpy.renderer.DPI)

        # I have no clue why I need to swap width/height for texture/FBO.
        size_swapped = self.size[::-1]

        # Texture where we render the scene.
        self._rendertex = gloo.Texture2D(shape=size_swapped + (RGBA_DEPTH,))
        # FBO.
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(size_swapped))

    # lines_ys[chan] = lines_coords[chan][1]
    _lines_ys: List[np.ndarray] = None

    def set_ys(self, lines_ys: List[np.ndarray]):
        """ Assigns a list of ydata. """
        for i, ys in enumerate(lines_ys):
            self._lines_ys[i][:] = ys
        self.update()

    # lines_coords[chan] is a 2D ndarray.
    # lines_coords[chan][0] = xs, precomputed in create_lines()
    # lines_coords[chan][1] = ys, updated once per frame.
    _lines_coords: List[np.ndarray]

    # Vispy line objects.
    _lines: List[visuals.LineVisual]
    # All draw()able Vispy elements.
    _visuals: list

    def create_lines(self, lines_nsamp: List[int], layout: 'RendererLayout'):
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

            # Redrawing is handled by set_ys(), not visuals.
            # Potential optimization: let visuals decide to not redraw the
            # exact same data?

        self._visuals = self._lines
        self.set_visual_viewport()

    def set_visual_viewport(self):
        # Ignore self.physical_size altogether,
        # it's only meaningful when rendering to GUI (but I render to texture)
        vp = (0, 0, *self.size)
        self.context.set_viewport(*vp)
        for visual in self._visuals:
            visual.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, event):
        """ Called by canvas.events.draw(). """
        with self._fbo:
            gloo.clear('black')
            for visual in self._visuals:
                visual.draw()
            self.im = _screenshot(alpha=False)


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

    xticks = fenceposts(cfg.width, layout.ncols)
    widths = np.diff(xticks)

    yticks = fenceposts(cfg.height, layout.nrows)
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
