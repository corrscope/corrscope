import sys
from glob import glob
from pathlib import Path
from typing import Optional, List, Tuple, Union, cast, TypeVar

import click

import corrscope
from corrscope.channel import ChannelConfig
from corrscope.config import yaml
from corrscope.corrscope import template_config, CorrScope, Config, Arguments
from corrscope.outputs import IOutputConfig, FFplayOutputConfig
from corrscope.settings.paths import MissingFFmpegError
from corrscope.utils.profile_wrapper import run_profile

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
YAML_EXTS = [".yaml"]
# Default extension when writing Config.
YAML_NAME = YAML_EXTS[0]

# Default output extension, only used in GUI and unit tests
VIDEO_NAME = ".mp4"


DEFAULT_NAME = corrscope.app_name


def _get_file_name(cfg_path: Optional[Path], cfg: Config, ext: str) -> str:
    """
    Returns a file path with extension (but no dir).
    Defaults to "corrscope.ext".
    """
    name = get_file_stem(cfg_path, cfg, default=DEFAULT_NAME)
    return name + ext


T = TypeVar("T")


def get_file_stem(cfg_path: Optional[Path], cfg: Config, default: T) -> Union[str, T]:
    """
    Returns a "name" (no dir or extension) for saving file or video.
    Defaults to `default`.

    Used by GUI as well.
    """
    if cfg_path:
        # Current file was saved.
        file_path_or_name = Path(cfg_path)

    # Current file was never saved.
    # Master audio and all channels may/not be absolute paths.
    elif cfg.master_audio:
        file_path_or_name = Path(cfg.master_audio)
    elif len(cfg.channels) > 0:
        file_path_or_name = Path(cfg.channels[0].wav_path)
    else:
        return default

    return file_path_or_name.stem


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# fmt: off
@click.command(context_settings=CONTEXT_SETTINGS)
# Inputs
@click.argument('files', nargs=-1)
# Override default .yaml settings (only if YAML file not supplied)
# Incorrect [option] name order: https://github.com/pallets/click/issues/793
@click.option('--audio', '-a', type=File, help=
        'Input path for master audio file')
# Disables GUI
@click.option('--write', '-w', is_flag=True, help=
        "Write config YAML file to current directory (don't open GUI).")
@click.option('--play', '-p', is_flag=True, help=
        "Preview (don't open GUI).")
@click.option('--render', '-r', type=OutFile, help=
        "Render and encode video to file (don't open GUI).")
# Debugging
@click.option('--profile', is_flag=True, help=
        'Debug: Write CProfiler snapshot')
@click.version_option(corrscope.__version__)
# fmt: on
def main(
    files: Tuple[str],
    # cfg
    audio: Optional[str],
    # gui
    write: bool,
    play: bool,
    render: Optional[str],
    profile: bool,
):
    """Intelligent oscilloscope visualizer for .wav files.

    FILES can be one or more .wav files (or wildcards), one folder, or one
    .yaml config.
    """
    # GUI:
    # corrscope
    # corrscope file.yaml
    # corrscope wildcard/wav/folder ... [--options]
    #
    # CLI:
    # corrscope wildcard/wav/folder ... [--options] --write-cfg file.yaml [--play]
    # corrscope wildcard/wav/folder ... --play
    # corrscope file.yaml --play
    # corrscope file.yaml --write-yaml
    #
    # - You can specify as many wildcards or wav files as you want.
    # - You can only supply one folder, with no files/wildcards.

    show_gui = not any([write, play, render])

    # Gather data for cfg: Config object.
    CfgOrPath = Union[Config, Path]

    cfg_or_path: Union[Config, Path, None] = None
    cfg_dir: Optional[str] = None

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
                    f"Cannot supply multiple arguments when providing folder {path}"
                )
            matches = sorted(path.glob("*.wav"))
            wav_list += matches
            break

        elif path.suffix in YAML_EXTS:
            # Load a YAML file to cfg, and skip template_config().
            if len(files) > 1:
                raise click.ClickException(
                    f"Cannot supply multiple arguments when providing config {path}"
                )
            cfg_or_path = path
            cfg_dir = str(path.parent)
            break

        else:
            # Load one or more wav files.
            matches = sorted([Path(s) for s in glob(name)])
            if not matches:
                matches = [path]
                if not path.exists():
                    raise click.ClickException(
                        f"Supplied nonexistent file or wildcard: {path}"
                    )
            wav_list += matches

    if not cfg_or_path:
        # cfg and cfg_dir are always initialized together.
        channels = [ChannelConfig(str(wav_path)) for wav_path in wav_list]

        cfg_or_path = template_config(
            master_audio=audio,
            # fps=default,
            channels=channels,
            # width_ms...trigger=default,
            # amplification...render=default,
        )
        cfg_dir = "."

    assert cfg_or_path is not None
    assert cfg_dir is not None
    if show_gui:

        def command():
            from corrscope import gui

            return gui.gui_main(cast(CfgOrPath, cfg_or_path))

        if profile:
            run_profile(command, "gui")
        else:
            command()

    else:
        if not files:
            raise click.UsageError("Must specify files or folders to play")

        if isinstance(cfg_or_path, Config):
            cfg = cfg_or_path
            cfg_path = None
        elif isinstance(cfg_or_path, Path):
            cfg = yaml.load(cfg_or_path)
            cfg_path = cfg_or_path
        else:
            assert False, cfg_or_path

        if write:
            # `corrscope file.yaml -w` should write back to same path.
            write_path = _get_file_name(cfg_path, cfg, ext=YAML_NAME)
            yaml.dump(cfg, Path(write_path))

        outputs = []  # type: List[IOutputConfig]

        if play:
            outputs.append(FFplayOutputConfig())

        if render:
            video_path = render
            outputs.append(cfg.get_ffmpeg_cfg(video_path))

        if outputs:
            arg = Arguments(cfg_dir=cfg_dir, outputs=outputs)
            command = lambda: CorrScope(cfg, arg).play()
            if profile:
                first_song_name = Path(files[0]).name
                run_profile(command, first_song_name)
            else:
                try:
                    command()
                except MissingFFmpegError as e:
                    # Tell user how to install ffmpeg (__str__).
                    print(e, file=sys.stderr)
