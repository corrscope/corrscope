# Corrscope Help

Homepage at <https://github.com/jimbo1qaz/corrscope>. Report bugs at https://github.com/jimbo1qaz/corrscope/issues/ or https://discord.gg/CCJZCjc.

*Corrscope is named because it cross-correlates the input wave and a history buffer, to maximize wave alignment between frames.*

## Tutorial

![Screenshot of Corrscope and video preview](images/corrscope-screenshot.png?raw=true)

- Global options: Left pane
- Trigger options: Top-right pane
- Per-channel options: Bottom-right table

Start by adding channels to be visualized: look at the bottom-right table and click the "Add..." button.

To add audio to play in the background, look at the top-right "FFmpeg Options" and click the Master Audio "Browse..." button.

To make the waves taller, go to the left panel's General tab and edit Amplification. Afterwards, click the Appearance tab and customize the appearance of the oscilloscope. (Note that colored lines will be discolored and blurred by Youtube's chroma subsampling.)

Click Preview to launch a live preview of the oscilloscope with audio.

Click Render to render the oscilloscope to video.

Click Save to save the current project configuration to a file. These project files can loaded in corrscope, previewed or rendered from the command line, or shared with corrscope's author when reporting issues.

## Configuring the Trigger

Unlike other triggering algorithms (which only depend on the current frame's worth of data), Corrscope uses a correlation trigger, which remembers the waveforms of past frames to help align future frames.

Corrscope's triggering algorithm is configurable, allowing it to track many types of waves well. However it may be intimidating to newcomers. This will provide several types of waves, along with suggestions for how to tune trigger options.

Triggering options are found on the left panel. Trigger Width is located in the General tab. All other options are found on the Trigger tab. (Per-channel triggering options are found in the table.)

### Sampled Trumpets and Trigger Direction (screenshot from [Tales of Phantasia](https://www.youtube.com/watch?v=GdM03JV_Vw0))

![Screenshot of trumpets in corrscope](images/trumpet.png?raw=true)

Sampled trumpets generally consist of a sharp falling edge, followed by gibberish with one or more rising edges.

- Set "Trigger Direction" to "Falling (-1)", which will track the falling edge well. (Using a rising-edge trigger will result in poor results, since the gibberish will vary between notes, especially for SNES SPC music using the echo functionality.)
- Slope trigger is useful, since the trumpet has a narrow tall positive section, followed by a narrow tall negative section.

### Complex Waves and Trigger Direction (screenshot from [Midori Mizuno - Sinkhole](https://www.youtube.com/watch?v=ElWHUp0BIDw))

![Screenshot of complex wave in corrscope](images/complex-bass.png?raw=true)

Corrscope's standard "edge trigger" does not look for "steep edges" but instead "sign changes". It operates by maximizing `(signed area in right half) - (signed area in left half)`. This waveform has a clear falling edge from positive to negative, but no clear edge from negative to positive.

Either:

- Set "Trigger Direction" to "Falling (-1)".
- Alternatively set "Trigger Direction" to "Rising (+1)", set "Edge Strength" to 0, and increase "Slope Strength". This will latch onto the small rising edge.

### NES Triangle Waves

NES triangle waves are stair-stepped. In theory, Area Trigger would work and properly locate the best zero-crossing on each frame. However, corrscope subtracts DC from each frame, and the exact amount of DC (positive or negative) fluctuates. Subtracting various amounts of DC causes corrscope to jump between different rising edges located near zero-crossing.

Try the following:

- Use any "Trigger Direction" you prefer. Rising and Falling both work equally well.
- Set "Sign Triggering" to 1 or so. This causes corrscope to preprocess the waveform and enhance edges located near the zero crossing. This occurs before DC is subtracted.
- Afterwards, set "Edge Strength" to nonzero (and optionally enable "Slope Strength"). This will pick up the newly created steep edges located at zero crossings.

NES triangle waves have 15 rising/falling edges. The NES high-pass removes DC and low frequencies, causing waveforms to decay towards y=0. As a result, "which edge crosses y=0" changes with pitch.

- Reduce "Buffer Strength" to 0 (or up to 0.5). Corrscope's buffer needs to be disabled, to prevent it from remembering "which edge used to cross y=0".

### FDS FM Waves

FDS FM changes the width of waves, but not their height.

The NES high-pass removes DC and low frequencies, causing waveforms to decay towards y=0. If FDS waves contain anything other than pulse/saw, "which part of the wave crosses y=0" may change with FM and pitch.

- Experiment with "Trigger Direction".
- Use nonzero Slope Strength and low (or zero) Edge Strength, to reliably locate the sharpest edge in a waveform. This is because sharp edges are preserved by FM, whereas the width of waves is not.
  - If you have multiple steep rising/falling edges,

### Yamaha FM and SNES/sampled Waves

- Experiment with "Trigger Direction".
- Try using Slope Strength, Edge Strength, or a combination of both.
- Reduce both relative to Buffer Strength to track evolving waves better (but center new/existing waves less strongly). To restore centering, you can enable Post Triggering and experiment with the radius.

## Options

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

If `Sign Strength` (Sign Triggering on the GUI) is set to nonzero `strength`, corrscope computes `peak = max(abs(data))`. It adds `peak * strength` to positive parts of `data`, subtracts `peak * strength` from negative parts of `data`, and heavily amplifies parts of the wave near zero. This helps the correlation trigger locate zero-crossings exactly.

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
