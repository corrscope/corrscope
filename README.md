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

### Installing Prebuilt Windows Binaries

On Windows, download Windows binary releases (.7z files) from the [Releases page](https://github.com/corrscope/corrscope/releases), then double-click `corrscope.exe` or run `corrscope (args)` via CLI.

### Installing from PyPI via pipx (cross-platform, releases)

pipx creates an isolated environment for each program, and adds their binaries into PATH. I find this most reliable in practice, though it runs into issues after upgrading system Python in-place.

- Install Python 3.8 or above.
- Install pipx using either your Linux package manager, `pip3 install --user pipx`, or `pip install --user pipx`.
- Run `pipx install corrscope[qt5]`
    - On Linux, to add support for native Qt themes, instead run `pipx install --system-site-packages corrscope[qt5]`
    - On M1 Mac, instead run `pipx install corrscope[qt6]`
- Open a terminal and run `corr (args)`.

### Installing from PyPI via Pip (cross-platform, releases)

pip installs packages into a per-user Python environment. This has the disadvantage that each program you install influences the packages seen by other programs. It might run into issues when upgrading system Python in-place; I haven't tested.

- Install Python 3.8 or above.
- If necessary, install pip using your Linux package manager.
- Run `pip3 install --user corrscope[qt5]`
    - On M1 Mac, instead run `pip3 install --user corrscope[qt6]`
- Open a terminal and run `corr (args)`.

### Dev builds (Windows)

Windows dev builds are compiled automatically, and available at https://ci.appveyor.com/project/nyanpasu64/corrscope/history.

Installing dev builds on non-Windows platforms without cloning the repo (eg. Git URLs or .whl files) is not supported yet. Fixes are welcome.

### Running from Source Code (cross-platform, dev master)

Install Python 3.8 or above, and Poetry. My preference is to run `pipx install poetry`. You can alternatively use the Poetry installer via `curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`, but in the past, updating via `poetry self update` has broken and left me with no Poetry at all, requiring reinstalling.

```shell
cd path/to/corrscope
poetry install -E qt5  # --develop is implied
# On M1 Mac, instead run `poetry install -E qt6`.
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

## Mac-specific Issues

### Preview audio cutting out

When you preview a video in Corrscope, it sends video frames to ffplay, which opens a video player window and also plays synchronized audio. On Mac (at least my M1 MacBook Air running macOS 12.3), switching windows can cause ffplay to stutter and temporarily or semi-permanently stop playing audio (until you restart the preview). There is no fix for this issue at the moment.

Rendering does not stutter on M1, since neither corrscope nor ffmpeg are affected by app switching, or App Nap.

### Gatekeeper

On Mac, if you render a video file, in some cases (eg. IINA video player) you may not be able to open the resulting files. Gatekeeper will print an error saying '"filename.mp4" cannot be opened because it is from an unidentified developer.'. If you see this message, try opening the file again. Once you silence the error once, it should not reappear.

## Contributing

Issues, feature requests, and pull requests are accepted.

This project uses [Black code formatting](https://github.com/ambv/black). Either pull request authors can reformat code before creating a PR, or maintainers can reformat code before merging.

You can install a Git pre-commit hook to apply Black formatting before each commit. Open a terminal/cmd in this repository and run:

```sh
pip install --user pre-commit
pre-commit install
```
