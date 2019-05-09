# Corrscope Help

Homepage at <https://github.com/jimbo1qaz/corrscope>.

*Corrscope is named because it cross-correlates the input wave and a history buffer, to maximize wave alignment between frames.*

## Program Layout

![Screenshot of Corrscope and video preview](images/corrscope-screenshot.png?raw=true)

- Global options: Left pane
- Trigger options: Top-right pane
- Per-channel options: Bottom-right table

## Correlation Trigger Explanation

Unlike other triggering algorithms (which only depend on the current frame's worth of data), Corrscope uses a correlation trigger, which remembers several forms of data between frames, to help align future frames.

### Options

All tabs are located in the left pane.

- Global
    - `Trigger Width` (also controllable via per-channel "Trigger Width ×")
- Trigger, Wave Alignment
    <!-- - `Buffer Strength` -->
    - `Buffer Responsiveness`
    - `Mean Responsiveness`
    - `Pitch Tracking`
- Trigger, Edge Triggering
    - `Edge Direction`
    - `Edge Strength`
- Trigger, Post Triggering
    - Post Trigger
    - `Post Trigger Radius`

### Variables Remembered

- `buffer`: array of samples, containing `Trigger Width` (around 40 milliseconds) recent "trigger outputs". Starts out as all zeros.
- `mean`: real number, estimated from recent "trigger inputs". Starts out at 0.

### Obtaining Data (each frame)

On each frame, corrscope fetches [from the channel] a buffer of mono `data`, centered at the current time. The amount of data used is controlled by `Trigger Width`, which should be increased to keep low bass stable.

- If `Edge Direction` is "Falling (-1)", then both the main and post trigger will receive negated data from the wave, causing both to search for falling edges (instead of rising edges).

### Sign Triggering

Some waves do not have clear edges. For example, triangle waves do not have clear rising edges (leading to suboptimal triggering), and NES triangles have 15 small rising edges, causing corrscope to jump between them.

If `Sign Triggering` is set to nonzero `strength`, corrscope computes `peak = max(abs(data))`. It adds `peak * strength` to positive parts of `data`, subtracts `peak * strength` from negative parts of `data`, and heavily amplifies parts of the wave near zero. This helps the correlation trigger locate zero-crossings exactly.

### Mean/Period

To remove DC offset from the wave, corrscope calculates the `mean` of input `data` and subtracts this averaged `mean` from `data`.

Corrscope then estimates the fundamental `period` of the waveform, using autocorrelation.

Corrscope multiplies `data` by `data window` to taper off the edges towards zero, and avoid using data over 1 frame old.

### (optional) Pitch Tracking

If `Pitch Tracking` is enabled:

If `period` changes significantly:

- Cross-correlate the log-frequency spectrum of `data` with `buffer`.
- Rescale `buffer` until its pitch matches `data`.

Pitch Tracking may get confused when `data` moves from 1 note to another over the course of multiple frames. If the right half of `buffer` changes to a new note while the left half is still latched onto the old note, the next frame will latch onto the mistriggered right half of the buffer. To prevent issues, you should consider reducing `Buffer Responsiveness` (so `buffer` will not "learn" the wrong pitch, and instead be rescaled to align with the new note).

### Correlation Triggering (uses `buffer`)

Precomputed: `edge_finder`, which is computed once and reused for every frame.

Corrscope cross-correlates `data` with `buffer + edge_finder` to produce a "buffer similarity + edge" score for each possible `data` triggering location. Corrscope then picks the location in `data` with the highest score, then sets `position` to be used for rendering.

### (Optional) Post Triggering

If post triggering is enabled:
- We recalculate the `post mean` of data around our new `position` value. If `position` is a good trigger position (and there are no nearby discontinuities like note changes), then `post mean` should be stable and not jitter.
- The post trigger is called with `position` and returns a new `position`, which overwrites the original variable.

#### Zero Crossing Trigger

Setting Post Trigger to "Zero Crossing Trigger" causes corrscope to "slide" towards edges. The maximum distance per frame is determined by GUI `Post Trigger Radius` (in samples).

- Grab some new data around `position`. Subtract `post mean` from the new data.
- If the data is positive, search left for a rising edge. If the data is negative, search right for a rising edge.
- If no edge is found within `Post Trigger Radius` samples, return `position` ± `Post Trigger Radius`.

### Updating Buffer

- Corrscope fetches [from the channel] a buffer of mono `triggered data`, centered at `position`
- `triggered data` is multiplied by a Gaussian window with width 0.5 × period
- Corrscope updates `buffer` = `buffer` + `Buffer Responsiveness` × (`triggered data` - `buffer`).
