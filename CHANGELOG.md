## 0.5.0

### Breaking Changes

### Features

### Changelog


## 0.4.0

### Breaking Changes

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
    - Pitch tracking may increase CPU usage on noise channels. See https://github.com/jimbo1qaz/corrscope/issues/213 for details
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

See https://github.com/jimbo1qaz/corrscope/releases.
