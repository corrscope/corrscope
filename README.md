# Corrscope

[![Appveyor build status](https://ci.appveyor.com/api/projects/status/awiajnwd6a4uhu37/branch/master?svg=true)](https://ci.appveyor.com/project/nyanpasu64/corrscope/branch/master)
[![Latest release](https://img.shields.io/github/v/release/corrscope/corrscope?include_prereleases)](https://github.com/corrscope/corrscope/releases)
[![PyPI release](https://img.shields.io/pypi/v/corrscope.svg)](https://pypi.org/project/corrscope/)

Corrscope renders oscilloscope views of WAV files recorded from chiptune (game music from retro sound chips).

Corrscope uses "waveform correlation" to track complex waves (including SNES and Sega Genesis/FM synthesis) which jump around on other oscilloscope programs.

Sample results can be found on my Youtube channel at https://www.youtube.com/nyanpasu64/videos.

Documentation is available at https://corrscope.github.io/corrscope/.

![Screenshot of Corrscope and video preview](docs/images/corrscope-screenshot.png?raw=true)

## Status

Corrscope is currently in semi-active development. The program basically works and I will fix bugs as they are discovered. Features will be added (and feature requests may be accepted) on a case-by-case basis. For technical support or feedback, contact me at Discord (https://discord.gg/CCJZCjc), or alternatively in the issue tracker (using the "Support/feedback" template). Pull requests may be accepted if they're clean.

## Dependencies

- FFmpeg

## Installation

- Releases (recommended): https://github.com/corrscope/corrscope/releases
- Dev Builds: https://ci.appveyor.com/project/nyanpasu64/corrscope/history

On Windows, download Windows binary releases (.7z files), then double-click `corrscope.exe` or run `corrscope (args)` via CLI.

On other operating systems, download cross-platform Python packages (.whl or .tar.gz), then install Python 3.6+ and run `pip install FILENAME.whl`, then run `corr (args)`.

## Installing from PyPI via Pip (cross-platform, releases)

Install Python 3.6 or above (3.5 will not work). Note that Python versions other than 3.8 and 3.9 are untested.

```shell
# Installs into per-user Python environment.
pip3 install --user corrscope
corr (args)
```

## Running from Source Code (cross-platform, dev master)

Install Python 3.6 or above (3.5 will not work), and Poetry.

```shell
# Installs into an isolated environment.
# Install Poetry (only do this once)
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
cd path/to/corrscope
poetry install # --develop is implied
poetry run corr (args)
```

## GUI Tutorial

1. Open GUI:
    - `corrscope.exe` to create new project
    - `corrscope.exe file.yaml` to open existing project
1. Add audio to play back
    - On the right side of the window, click "Browse" to pick a master audio file.
1. Add oscilloscope channels
    - On the right side of the window, click "Add" to add WAV files to be viewed.
1. Edit settings
    - Global settings on the left side of the window
    - Per-channel on the right side
1. Play or render to MP4/etc. video (requires ffmpeg)
    - Via toolbar or menu

## Command-line Tutorial

1. Create YAML:
    - `corrscope split*.wav --audio master.wav -w`
    - Specify all channels on the command line.
    - `-a` or `--audio` specifies master audio track.
    - Creates file `master.yaml`.

1. Edit `master.yaml` to change settings.

1. Play (requires ffmpeg):
    - `corrscope master.yaml -p/--play`

1. Render and encode video (requires ffmpeg)
    - `corrscope master.yaml -r/--render file.mp4` (other file extensions supported)

## Contributing

Issues, feature requests, and pull requests are accepted.

This project uses [Black code formatting](https://github.com/ambv/black). Either pull request authors can reformat code before creating a PR, or maintainers can reformat code before merging.

You can install a Git pre-commit hook to apply Black formatting before each commit. Open a terminal/cmd in this repository and run:

```sh
pip install --user pre-commit
pre-commit install
```
