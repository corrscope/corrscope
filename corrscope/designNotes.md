# Design Notes (cross-cutting concerns)

## Renderer to Output

- `renderer.py` produces a frame-buffer format which is passed to `outputs.py`. This is a cross-cutting concern, and the 2 modules must agree on bytes per pixel and pixel format.
- Each renderer returns a different pixel format, and I switch renderers more often than outputs.
- `tests/test_renderer.py` must convert hexadecimal colors to the right pixel format, for comparison.

### Solution

- Each `BaseRenderer` subclass exposes classvars `bytes_per_pixel` and `ffmpeg_pixel_format`.
- `renderer.py` exposes `Renderer = preferred subclass`.
- `outputs.py` imports `renderer.Renderer` and uses the format stored within.
- Additionally, `tests/test_renderer.py` uses `Renderer.color_to_bytes()`

## Wave/Trigger/Render Centering

Assume stride=1 for simplicity. If stride > 1, get_around(x) may not include [x] at all!

If data is centered around `x` with length `N`:
- `halfN = N//2`
- data contains range `[x - halfN, x - halfN + N)`.

The following functions follow this convention:

- `Wave.get_around(x, N)` returns `result[N//2] == self[sample]`
    - See `Wave.get_around()` docstring.
- `Trigger.get_trigger(x)` calls `get_around(x)` where `result[N//2] == self[sample]`.
    - Contract: if `get_trigger(x) == x`, then `data[[x-1, x]] == [<0, >0]`.
- `Renderer._setup_axes()` watches x-range `[x - halfN, x - halfN + N) * stride`.
