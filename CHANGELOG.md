## 0.8.0 (unreleased)

### Features

- Add option to color lines by pitch (#386)
- Add support for background images (#388, @Sanqui)
- Add support for line outlines (#388, @Sanqui)

### Major Changes

- Rewrite the trigger algorithm to enhance determinism (#403)
    - Triggering still makes mistakes, especially when DC offset varies within a frame (eg. NES 75% pulse changing volumes). This may be addressed in the future.
    - Changed default triggering settings as well.

### Changelog

- Fix passing absolute .wav paths into corrscope CLI (#398)
- Fix preview error when clearing "Trigger/Render Width" table cells (#407)

## 0.7.1

### Major Changes

- Change `-r/--render` command line flag to take an output path (#382)
- Render videos in BT.709 colorspace by default (#384)

### Changelog

- Update NumPy so `poetry install` on Python 3.8+ won't build NumPy from source (#371)
- Fix longstanding crash when prefs.yaml is corrupted, reset settings instead (#377)
- Atomically save prefs.yaml to prevent file corruption (#377)
- Fix issue where foobar2000 WAV files fail with message "ValueError: Incomplete wav chunk." (#379)
- Build Win32 binaries as well as Win64 (#381)
- Build official Win32/Win64 binaries on Python 3.8 (the last release to support Windows 7) (#381)
- Add .mkv/.webm extensions to "Render to Video" dialog (#383)

## 0.7.0

Long delayed as well. I haven't been around corrscope in a while. Background images and spectral coloring are not in this release, but I wanted to push this out because it enables chroma subsampling (reduces support queries) and fixes the FFmpeg URL.

### Major Changes

- Enable chroma subsampling by default (may affect saved projects) (#331)
- Improve FPS by reducing rendering overhead (#335)

### Changelog

- Add menu items linking to config folder and Github repository (#343)
- Fix FFmpeg URL, switch to static FFmpeg to reduce user error (#332, #358)
- Fix bug where videos were truncated if first channel was shorter than the rest (#360)

## 0.6.1

Long delayed... sorry.

### Major Changes

- Fix bug where narrow pulse waves were erroneously detected as silence (#306)
- Fix Windows-only crash when opening a non-ASCII path and restarting corrscope (#311)
- Fix bug where unrecognized fonts would cause corrscope to crash (#313)
- Fix bug where `pip install corrscope` failed on Linux because `PyQt5-sip` was pinned to 4.x (#319)

### Changelog

- On Windows, use locale-specific font, not hard-coded Segoe UI (#322)

## 0.6.0

### Features

- Rewrite pitch tracking to avoid false negatives (#274)
    - Previously, we rescaled the *buffer* to maximize spectral similarity between *data 2 frames ago* and data now.
    - Now we rescale the buffer to maximize spectral similarity between the buffer and data now.
- Improve period calculation, add maximum frequency cap (#294)
    - Fixes incorrectly high frequency with low bass notes
    - Fixes incorrectly high frequency with treble-heavy waveforms

### Major Changes

- Update default options for new projects (#275)
    - Stereo grid opacity = 0.25
    - Render FPS Divisor = 2 (preview-only, faster)
    - Trigger Width = 60 ms
- Always enable midline color, remove color checkbox (#291)
    - Can be disabled separately for h/v
- Enable grid color #55aaff for new projects (#300)

### Changelog

- Increase GUI maximum Trigger/Render Width to 200 ms
- Update trigger GUI, merge all edge-related triggers (#299)
- Rewrite FPS printing code
- Add test to ensure cancelling render terminates FFmpeg quickly
- Add support for excluding fields from always_dump="*" (#268)
    - Don't dump viewport_width/height by default

## 0.5.1

This is a bugfix release, since master has regressions in pitch tracking.

### Changelog

- Improve GUI dialog path defaults (#277)
- Display all GUI errors in dialog box, instead of crashing (#279)
- Display dialog and terminate ffmpeg, when closing project with preview/render active (#280)


## 0.5.0

### Breaking Changes

- Reorganize GUI, move trigger options to tab
- Improve NES triangle triggering, switch data window to Gaussian (#244)
- Remove mean responsiveness (always set to 1)
    - To improve triangle waves, use sign triggering instead.

### Features

- Add sign triggering (37d2c08a)
- Add support for per-channel labels (#256)
    - Some fonts may not work or display the wrong weight, due to Matplotlib issues.
- Add configurable grid line width (#265)
- Add Ctrl+Tab or Ctrl+PageUp/Down shortcuts to switch GUI tabs (#246)

### Changelog

- Quit GUI when pressing Ctrl-C in terminal (#252)
- Rewrite resolution division system to use internal DPI (#264)
- Refactor renderer API (c8239558)
- Add renderer debugging visualizations (development only)


## 0.4.0

### Breaking Changes

- Set default mean responsiveness to 0.05 instead of 1 (even in unmodified older files, oops)
- Always use full-resolution rendering, when rendering to file (make trigger/render subsampling preview-only)
- Remove buffer falloff from GUI (defaults to 0.5)
- Lag prevention is no longer increased, when trigger subsampling or trigger width × are >1

### Features

- Add Help menu (online help manual)
- Add custom stereo downmix modes
    - Allow left-only triggering, downmixing specific channels to mono, etc.
- Add post-triggering for finding zero-crossing edges
- Add optional slope-based triggering
    - Previous edge-triggering was area-based and located zero crossings

### Changelog

- Add trigger "buffer responsiveness" option
- Remove dependency on more_itertools


## 0.3.1

### Breaking Changes
- Fix time-traveling bug by reverting "Increase trigger diameter to improve bass stability" from 0.3.0

### Changelog

- Rebuild UI in Python, not .ui XML
- Show stack trace dialog when loading config fails
- Show stack trace dialog if exceptions raised before playback begins


## 0.3.0

### Breaking Changes

- Increase trigger diameter to improve bass stability
- Line width is now measured in pixels. (previously, 1 pt = 4/3 px)
- Tweak default config settings: white lines, gray midlines, 1 column, vertical orientation.
    - Enable preview-only resolution divisor by default, to boost speed

### Features

- Add pitch-tracking trigger checkbox
    - Waves should no longer jump around when pitch changes.
    - Rapidly repeating pitch changes (less than 6 frames apart) are skipped for performance.
    - Pitch tracking may increase CPU usage on noise channels. See https://github.com/corrscope/corrscope/issues/213 for details
- Add stereo rendering support
    - Located in stereo tab in GUI
- Add per-channel amplification support
- Add color picker to GUI
- Add unit suffixes to GUI spinboxes

### Changelog

- Prevent YAML dump from line-breaking long paths
- Dump Config.show_internals to YAML by default
- Add non-GUI option to disable antialiasing (does not improve performance)


## 0.2.0 and before

See https://github.com/corrscope/corrscope/releases.
