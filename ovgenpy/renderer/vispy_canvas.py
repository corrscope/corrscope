from typing import Tuple, TYPE_CHECKING, List

import numpy as np
from vispy import app, gloo, visuals
from vispy.gloo.util import _screenshot
from vispy.visuals.transforms import STTransform, NullTransform

if TYPE_CHECKING:
    from ovgenpy.renderer import RendererConfig

RGBA_DEPTH = 4


# TODO move into .renderer
class MyCanvas(app.Canvas):

    # self._fig, axes2d = plt.subplots(self.layout.nrows, self.layout.ncols...)
    def __init__(self, cfg: 'RendererConfig'):
        size = (cfg.height, cfg.width)  # eg. (800, 600) is landscape.
        app.Canvas.__init__(self, show=False, size=size)

        # Texture where we render the scene.
        self.rendertex = gloo.Texture2D(shape=self.size + (RGBA_DEPTH,))
        # FBO.
        self.fbo = gloo.FrameBuffer(self.rendertex, gloo.RenderBuffer(self.size))

    lines_pos: List[np.ndarray]
    # line_pos[chan] is a 2D ndarray.
    # line_pos[chan][0] = xs, precomputed in create_lines()
    # line_pos[chan][1] = ys, updated once per frame.

    _lines: List[visuals.LineVisual]

    def create_lines(self, lines_nsamp: List[int]):
        self.lines_pos = []

        for nsamp in lines_nsamp:
            line_pos = np.empty((nsamp, 2))
            line_pos

        for i, line in enumerate(self._lines):
            x = ...
            y = ...

            line.transform = STTransform(translate=[x, y])

            # redraw the canvas if any visuals request an update
            line.events.update.connect(lambda evt: self.update())

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
