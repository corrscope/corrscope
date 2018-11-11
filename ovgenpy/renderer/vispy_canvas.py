from typing import Tuple, TYPE_CHECKING

from vispy import app, gloo, visuals
from vispy.gloo.util import _screenshot
from vispy.visuals.transforms import STTransform, NullTransform

if TYPE_CHECKING:
    from ovgenpy.renderer import RendererConfig

RGBA_DEPTH = 4


# TODO move into .renderer
class MyCanvas(app.Canvas):
    def __init__(self, cfg: 'RendererConfig'):
        size = (cfg.height, cfg.width)  # eg. (800, 600) is landscape.
        app.Canvas.__init__(self, show=False, size=size)

        # Texture where we render the scene.
        self.rendertex = gloo.Texture2D(shape=self.size + (RGBA_DEPTH,))
        # FBO.
        self.fbo = gloo.FrameBuffer(self.rendertex, gloo.RenderBuffer(self.size))






        self.lines = [
            # GL-method lines:
            visuals.LineVisual(pos=pos, color=color, method='gl'),
            visuals.LineVisual(pos=pos, color=(0, 0.5, 0.3, 1), method='gl'),
            visuals.LineVisual(pos=pos, color=color, width=5, method='gl'),
        ]
        counts = [0, 0]
        for i, line in enumerate(self.lines):
            # arrange lines in a grid
            tidx = (line.method == 'agg')
            x = 400 * tidx
            y = 140 * (counts[tidx] + 1)
            counts[tidx] += 1
            line.transform = STTransform(translate=[x, y])
            # redraw the canvas if any visuals request an update
            line.events.update.connect(lambda evt: self.update())

        self.texts = [
            visuals.TextVisual(
                'GL', bold=True, font_size=24, color='w', pos=(200, 40)
            )
        ]
        for text in self.texts:
            text.transform = NullTransform()
        self.visuals = self.lines + self.texts
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
