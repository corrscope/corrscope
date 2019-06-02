# Module/Class Structure and Organization

When corrscope is launched, it first executes `__main__.py` and `cli.py`.

## cli.py

If -w is present, it writes a .yaml file. If -p are present, it runs `corrscope.CorrScope` directly. If neither is present, it imports and runs the `gui` subdirectory.

## CorrScope, Config

`corrscope.py` defines classes `CorrScope`, `Config` (and `Arguments`).

- `CorrScope` is the main loop of the program, and only communicates with the GUI through `Arguments`. `CorrScope` requires a `Config` and `Arguments`.
    - `Arguments` is constructed by the GUI and used to update rendering progress dialogs, etc.
    - `Config` is a dataclass (see `config.py`) which can be edited through the GUI, or loaded/saved to a YAML file.
- When `cli.py` creates new configs, `default_config()` is used as a template to supply values. When loading existing YAML files, only dataclass default values are used.

-----

`Config` holds `channels: List[ChannelConfig]`, which store all per-channel settings.

`CorrScope` turns `channels` into a list of `Channel` objects. Each channel uses its own `ChannelConfig` parameters to create:

- self.trigger_wave: Wave
- self.render_wave: Wave
- self.trigger: MainTrigger

-----

Each frame:

`CorrScope` reads data from Wave objects, asks MainTrigger

Each channel has its own `trigger_wave`, `render_wave`, and `trigger` instance. For each channel, `trigger_wave` is used by `trigger` to find a "trigger time" near the frame's center time.

The `Renderer` and `Output` instances are shared among all channels.

- `Renderer` turns a list of N-channel "wave file portions" (from `render_wave`) into a RGB video frame.
- `Output` sends a RGB video frame to ffmpeg, ffplay, or (in the future) an interactive GUI player window.

## config.py

See docstring at top of file.

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
