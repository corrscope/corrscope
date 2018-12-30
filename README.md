# Corrscope
Python program to render wave files into oscilloscope views, featuring improved correlation-based triggering algorithm

<!-- screenshot here -->

## Dependencies

- FFmpeg

## Installation

- Releases (recommended): https://github.com/jimbo1qaz/corrscope/releases
- Dev Builds: https://ci.appveyor.com/project/jimbo1qaz/ovgenpy/history
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
