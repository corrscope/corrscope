# ovgenpy
Python program to render wave files into oscilloscope views, featuring improved correlation-based triggering algorithm

## Dependencies

- Python 3.6 or above (3.5 will not work)
- FFmpeg

## Installation

<!--### Binary bundles
- Coming soon

### Pipsi
```shell
curl https://raw.githubusercontent.com/mitsuhiko/pipsi/master/get-pipsi.py | python3
pipsi install -e .
# and pray that python3 points to 3.6
```

doesn't work yet, see https://github.com/nyanpasu64/ovgenpy/issues/74
-->
### Conda
```shell
conda create -n ovgenpy python=3.6 pip numpy scipy matplotlib pyqt=5
pip install -e .
```

## Command-line Tutorial

1. Create YAML:
    - `python -m ovgenpy split*.wav --audio master.wav -w`
    - Specify all channels on the command line.
    - `-a` or `--audio` specifies master audio track.
    - Creates file `master.yaml`.

1. Edit `master.yaml` to change settings.

1. Play (requires ffmpeg):
    - `...ovgenpy master.yaml -p/--play`

1. Render and encode MP4 video (requires ffmpeg)
- `...ovgenpy master.yaml -r/--render`
