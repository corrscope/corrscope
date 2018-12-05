from itertools import count
from pathlib import Path
from typing import Optional, List, Tuple, Union

import click

from ovgenpy.channel import ChannelConfig
from ovgenpy.config import OvgenError, yaml
from ovgenpy.outputs import IOutputConfig, FFplayOutputConfig, FFmpegOutputConfig
from ovgenpy.ovgenpy import default_config, Config, Ovgen


Folder = click.Path(exists=True, file_okay=False)
File = click.Path(exists=True, dir_okay=False)
OutFile = click.Path(dir_okay=False)


# https://github.com/pallets/click/issues/473
# @platformio requires some functionality which doesn't work in Click 6.
# Click 6 is marked as stable, but http://click.pocoo.org/ redirects to /5/.
# wat


# If multiple `--` names are supplied to @click.option, the last one will be used.
# possible_names = [('-', 'w'), ('--', 'write')]
# possible_names.sort(key=lambda x: len(x[0]))
# name = possible_names[-1][1].replace('-', '_').lower()


# List of recognized Config file extensions.
YAML_EXTS = ['.yaml']
# Default extension when writing Config.
YAML_NAME = YAML_EXTS[0]

# Default output extension
VIDEO_NAME = '.mp4'


def get_path(audio_file: Union[None, str, Path], ext: str) -> Path:
    # Write file to current working dir, not audio dir.
    if audio_file:
        name = Path(audio_file).name
    else:
        name = 'ovgenpy'

    # Add extension
    return Path(name).with_suffix(ext)


PROFILE_DUMP_NAME = 'cprofile'


@click.command()
# Inputs
@click.argument('files', nargs=-1)
# Override default .yaml settings (only if YAML file not supplied)
# Incorrect [option] name order: https://github.com/pallets/click/issues/793
@click.option('--audio', '-a', type=File, help=
        'Config: Input path for master audio file')
# Disables GUI
@click.option('--write', '-w', is_flag=True, help=
        "Write config YAML file to current directory (don't open GUI).")
@click.option('--play', '-p', is_flag=True, help=
        "Preview (don't open GUI).")
@click.option('--render', '-r', is_flag=True, help=
        "Render and encode MP4 video (don't open GUI).")
# Debugging
@click.option('--profile', is_flag=True, help=
        'Debug: Write CProfiler snapshot')
def main(
        files: Tuple[str],
        # cfg
        audio: Optional[str],
        # gui
        write: bool,
        play: bool,
        render: bool,
        profile: bool,
):
    """Intelligent oscilloscope visualizer for .wav files.

    FILES can be one or more .wav files (or wildcards), one folder, or one
    .yaml config.
    """
    # GUI:
    # ovgenpy
    # ovgenpy file.yaml
    # ovgenpy wildcard/wav/folder ... [--options]
    #
    # CLI:
    # ovgenpy wildcard/wav/folder ... [--options] --write-cfg file.yaml [--play]
    # ovgenpy wildcard/wav/folder ... --play
    # ovgenpy file.yaml --play
    # ovgenpy file.yaml --write-yaml
    #
    # - You can specify as many wildcards or wav files as you want.
    # - You can only supply one folder, with no files/wildcards.

    show_gui = not any([write, play, render])

    # Create cfg: Config object.
    cfg: Optional[Config] = None
    cfg_path: Optional[Path] = None
    cfg_dir: Optional[str] = None  # Changing to Path will take a lot of refactoring.

    wav_list: List[Path] = []
    for name in files:
        path = Path(name)

        # Windows likes to raise OSError when path contains *
        try:
            is_dir = path.is_dir()
        except OSError:
            is_dir = False
        if is_dir:
            # Add a directory.
            if len(files) > 1:
                # Warning is technically optional, since wav_prefix has been removed.
                raise click.ClickException(
                    f'Cannot supply multiple arguments when providing folder {path}')
            matches = sorted(path.glob('*.wav'))
            wav_list += matches
            break

        elif path.suffix in YAML_EXTS:
            # Load a YAML file to cfg, and skip default_config().
            if len(files) > 1:
                raise click.ClickException(
                    f'Cannot supply multiple arguments when providing config {path}')
            cfg = yaml.load(path)
            cfg_path = path
            cfg_dir = str(path.parent)
            break

        else:
            # Load one or more wav files.
            matches = sorted(Path().glob(name))
            if not matches:
                matches = [path]
                if not path.exists():
                    raise click.ClickException(
                        f'Supplied nonexistent file or wildcard: {path}')
            wav_list += matches

    if not cfg:
        channels = [ChannelConfig(str(wav_path)) for wav_path in wav_list]

        cfg = default_config(
            master_audio=audio,
            # fps=default,
            channels=channels,
            # width_ms...trigger=default,
            # amplification...render=default,
        )
        cfg_dir = '.'

    if show_gui:
        raise OvgenError('GUI not implemented')
    else:
        if not files:
            raise click.ClickException('Must specify files or folders to play')
        if write:
            write_path = get_path(audio, YAML_NAME)
            yaml.dump(cfg, write_path)

        outputs = []  # type: List[IOutputConfig]

        if play:
            outputs.append(FFplayOutputConfig())

        if render:
            video_path = get_path(cfg_path or audio, VIDEO_NAME)
            outputs.append(FFmpegOutputConfig(video_path))

        if outputs:
            assert Ovgen  # to prevent PyCharm from deleting the import
            command = 'Ovgen(cfg, cfg_dir, outputs).play()'
            if profile:
                import cProfile

                # Pycharm can't load CProfile files with dots in the name.
                profile_dump_name = Path(files[0]).name.split('.')[0]
                profile_dump_name += f'-{PROFILE_DUMP_NAME}'

                # Write stats to unused filename
                for i in count():
                    path = Path(profile_dump_name + str(i))
                    if not path.exists():
                        break

                # noinspection PyUnboundLocalVariable
                cProfile.runctx(command, globals(), locals(), str(path))

            else:
                exec(command)
