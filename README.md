# ovgenpy
Python program to render wave files into oscilloscope views, featuring improved correlation-based triggering algorithm

## Dependencies

- Python 3.6 (3.5 and 3.7 will not work)
- FFmpeg

## Installation

### Binary bundles:
- Coming soon

### Pipsi:
```shell
curl https://raw.githubusercontent.com/mitsuhiko/pipsi/master/get-pipsi.py | python3
pipsi install -e .
# and pray that python3 points to 3.6
```

TODO https://github.com/jimbo1qaz/ovgenpy/issues/74

### Conda:
```shell
conda create -n ovgenpy python=3.6 pip numpy scipy matplotlib
pip install -e .
```

## Usage

`python -m ovgenpy [OPTIONS] [FILES]...`

FILES can be one or more .wav files (or wildcards), one folder, or one .yaml config.

-   -a, --audio FILE  Input path for master audio file

Create YAML:
- `...ovgenpy split*.wav -a/--audio master.wav -w` writes to `master.yaml`

Edit the YAML file to change settings.

Play (requires ffmpeg):
- `...ovgenpy file.yaml -p/--play`

Render and encode MP4 video (requires ffmpeg)
- `...ovgenpy file.yaml -r/--render`

GUI will be added soon™.
