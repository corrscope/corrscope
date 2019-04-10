# Design Notes

## Renderer to Output

- `renderer.py` produces a frame-buffer format which is passed to `outputs.py`. This is a cross-cutting concern, and the 2 modules must agree on bytes per pixel and pixel format.
- Each renderer returns a different pixel format, and I switch renderers more often than outputs.
- `tests/test_renderer.py` must convert hexadecimal colors to the right pixel format, for comparison.

### Solution

- Each `BaseRenderer` subclass exposes classvars `bytes_per_pixel` and `ffmpeg_pixel_format`.
- `renderer.py` exposes `Renderer = preferred subclass`.
- `outputs.py` imports `renderer.Renderer` and uses the format stored within.
- Additionally, `tests/test_renderer.py` uses `Renderer.color_to_bytes()`
