## 0.4.0-pre

### Breaking Changes
...

### Features
...

### Changelog
...

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
