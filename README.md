# Corrscope

[![Build status](https://ci.appveyor.com/api/projects/status/awiajnwd6a4uhu37/branch/master?svg=true)](https://ci.appveyor.com/project/nyanpasu64/ovgenpy/branch/master)
[![codecov](https://codecov.io/gh/nyanpasu64/corrscope/branch/master/graph/badge.svg)](https://codecov.io/gh/nyanpasu64/corrscope)
[![Latest pre-release](https://img.shields.io/github/release-pre/nyanpasu64/corrscope.svg)](https://github.com/nyanpasu64/corrscope/releases)

Corrscope renders oscilloscope views of WAV files recorded from chiptune (game music from retro sound chips).

Corrscope uses "waveform correlation" to track complex waves (including SNES and Sega Genesis/FM synthesis) which jump around on other oscilloscope programs.

Sample results can be found on my Youtube channel at https://www.youtube.com/channel/UCIjb87rjJZxtNsHUdKXMsww/videos.

<!-- screenshot here -->

## Dependencies

- FFmpeg

## Installation

- Releases (recommended): https://github.com/nyanpasu64/corrscope/releases
- Dev Builds: https://ci.appveyor.com/project/nyanpasu64/ovgenpy/history
    - Download Windows binary releases (zip files), then double-click `corrscope.exe` or run `corrscope (args)` via CLI.
    - Download cross-platform Python packages (whl), then install Python 3.6+ and run `pip install *.whl`.

## Running from Source Code (cross-platform)

Install Python 3.6 or above (3.5 will not work), and Poetry.

```shell
# Install Poetry (only do this once)
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
# Install corrscope
cd path/to/corrscope
poetry install --develop corrscope
poetry run corr (args)
```

<!--
### Conda (possibly installs pyqt5 twice and breaks env)

```shell
conda create -n ovgenpy python=3.6 pip numpy scipy matplotlib pyqt=5
pip install -e .
python -m corrscope (args)
```
-->

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

1. Render and encode MP4 video (requires ffmpeg)
    - `corrscope master.yaml -r/--render`

## Contributing

Issues, feature requests, and pull requests are accepted.

This project uses [Black code formatting](https://github.com/ambv/black). Either pull request authors can reformat code before creating a PR, or maintainers can reformat code before merging.

You can install a Git pre-commit hook to apply Black formatting before each commit. Open a terminal/cmd in this repository and run:

```sh
pip install --user pre-commit
pre-commit install
```
