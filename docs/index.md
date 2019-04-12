# Corrscope Help

*Corrscope is named because it cross-correlates the input wave and a history buffer, to maximize wave alignment between frames.*

## Program Layout

![Screenshot of Corrscope and video preview](images/corrscope-screenshot.png?raw=true)

- Global options: Left pane
- Trigger options: Top-right pane
- Per-channel options: Bottom-right table

## Correlation Trigger Explanation

Unlike other triggering algorithms (which only depend on the current frame's worth of data), Corrscope uses a correlation trigger, which remembers several forms of data between frames, to help align future frames.

### Options

- Left pane, Global
    - `Trigger Width` (also controllable via per-channel "Trigger Width ×")
- Top pane, Wave Alignment
    <!-- - `Buffer Strength` -->
    - `Buffer Responsiveness`
    - `Mean Responsiveness`
    - `Pitch Tracking`
- Top pane, Edge Search
    - `Edge Direction`
    - `Edge Strength`

### Variables Remembered

- `buffer`: array of samples, containing `Trigger Width` (around 40 milliseconds) recent "trigger outputs". Starts out as all zeros.
- `mean`: real number, estimated from recent "trigger inputs". Starts out at 0.

### Obtaining Data (each frame)

On each frame, corrscope fetches [from the channel] a buffer of mono `data`, centered at the current time. The amount of data used is controlled by `Trigger Width`, which should be increased to keep low bass stable.

To remove DC offset from the wave, corrscope calculates the `new mean` of input `data`. Because each frame's mean may jitter depending on what window of the wave was selected, update saved `mean` = `mean` + `Mean Responsiveness` × (`new mean` - `mean`). Then corrscope subtracts this averaged `mean` from `data`.

Corrscope then estimates the fundamental `period` of the waveform, using autocorrelation.

### (optional) Pitch Tracking

If `Pitch Tracking` is enabled:

If `period` changes significantly, corrscope computes the spectrums of `data` and `data` from 2 frames ago, and cross-correlates them to estimate the pitch change over the last 2 frames. It then resamples (horizontally scales) `buffer` to match this pitch change.

### Correlation Triggering (uses `buffer`)

Corrscope multiplies `data` by `data window` to taper off the edges towards zero, and avoid using data over 1 frame old.

Precomputed: `edge_finder`, which is computed once and reused for every frame.

Corrscope cross-correlates `data` with `buffer + edge_finder` to produce a "buffer similarity + edge" score for each possible `data` triggering location. Corrscope then picks the location in `data` with the highest score, then sets `position` to be used for rendering.

### (Optional) Post Triggering

If post triggering is enabled, the post trigger is called with `position` and returns a new `position`, which overwrites the original variable.

### Updating Buffer

- Corrscope fetches [from the channel] a buffer of mono `triggered data`, centered at `position`
- `triggered data` is multiplied by a Gaussian window with width 0.5 × period
- Corrscope updates `buffer` = `buffer` + `Buffer Responsiveness` × (`triggered data` - `buffer`).
