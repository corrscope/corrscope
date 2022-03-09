"""
- Test command-line parsing and Config generation.
- Integration tests (see conftest.py).
"""
import os.path
import shlex
from os.path import abspath
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest import mock

import click
import pytest
from click.testing import CliRunner

import corrscope.channel
from corrscope import cli
from corrscope.cli import YAML_NAME
from corrscope.config import yaml
from corrscope.corrscope import Arguments, Config, CorrScope
from corrscope.outputs import FFmpegOutputConfig
from corrscope.util import pushd

if TYPE_CHECKING:
    import pytest_mock


def call_main(argv):
    return CliRunner().invoke(
        cli.main, argv, catch_exceptions=False, standalone_mode=False
    )


# corrscope configuration sinks


def yaml_sink(_mocker, command: str):
    """Mocks yaml.dump() and returns call args. Also tests dumping and loading."""
    with mock.patch.object(yaml, "dump") as dump:

        argv = shlex.split(command) + ["-w"]
        call_main(argv)

        dump.assert_called_once()
        (cfg, stream), kwargs = dump.call_args

        assert isinstance(cfg, Config)

    yaml_dump = yaml.dump(cfg)

    # YAML Representer.ignore_aliases() should return True for Enums.
    # If it returns True for other types, "null" is dumped explicitly, which is ugly.
    assert "end_time: null" not in yaml_dump

    cfg_round_trip = yaml.load(yaml_dump)
    assert cfg_round_trip == cfg, yaml_dump

    return (cfg, stream)


def player_sink(mocker: "pytest_mock.MockFixture", command: str):
    CorrScope = mocker.patch.object(cli, "CorrScope")

    argv = shlex.split(command) + ["-p"]
    call_main(argv)

    CorrScope.assert_called_once()
    args, kwargs = CorrScope.call_args
    cfg = args[0]

    assert isinstance(cfg, Config)
    return (cfg,)


@pytest.fixture(params=[yaml_sink, player_sink])
def any_sink(request) -> Callable[["pytest_mock.MockFixture", str], tuple]:
    sink = request.param
    return sink


# corrscope configuration sources


def test_no_files(any_sink, mocker):
    with pytest.raises(click.ClickException):
        any_sink(mocker, "")


@pytest.mark.parametrize("wav_dir", ". tests".split())
def test_file_dirs(any_sink, mocker, wav_dir):
    """Ensure loading files from `dir` places `dir/*.wav` in config."""
    wavs = Path(wav_dir).glob("*.wav")
    wavs = sorted(str(x) for x in wavs)

    cfg = any_sink(mocker, wav_dir)[0]
    assert isinstance(cfg, Config)

    assert [chan.wav_path for chan in cfg.channels] == wavs


@pytest.mark.parametrize("wav_dir", ". tests".split())
def test_absolute_dir(any_sink, mocker, wav_dir):
    """Ensure loading files from `abspath(dir)` functions properly."""

    wavs = Path(wav_dir).glob("*.wav")
    wavs = sorted(abspath(str(x)) for x in wavs)

    cfg = any_sink(mocker, shlex.quote(abspath(wav_dir)))[0]
    assert isinstance(cfg, Config)

    assert [chan.wav_path for chan in cfg.channels] == wavs


def test_absolute_files(any_sink, mocker):
    """On Linux, the shell expands wildcards. Ensure loading absolute .wav paths
    functions properly."""

    wav_dir = "tests"

    wavs = Path(wav_dir).glob("*.wav")
    wavs = sorted(abspath(str(x)) for x in wavs)

    cfg = any_sink(mocker, shlex.join(wavs))[0]
    assert isinstance(cfg, Config)

    assert [chan.wav_path for chan in cfg.channels] == wavs


def test_absolute_wildcard(any_sink, mocker):
    """On Windows, the program is expected to expand wildcards. Ensure loading
    absolute paths containing wildcards functions properly."""

    wav_dir = "tests"

    wavs = Path(wav_dir).glob("*.wav")
    wavs = sorted(abspath(str(x)) for x in wavs)

    wildcard_wav = os.path.join(abspath(wav_dir), "*.wav")
    cfg = any_sink(mocker, shlex.quote(wildcard_wav))[0]
    assert isinstance(cfg, Config)

    assert [chan.wav_path for chan in cfg.channels] == wavs


def q(path: Path) -> str:
    return shlex.quote(str(path))


def test_write_dir(mocker):
    """Loading `--audio another/dir` should write YAML to current dir.
    Writing YAML to audio dir: causes relative paths (relative to pwd) to break."""

    audio_path = Path("tests/sine440.wav")
    arg_str = f"tests -a {q(audio_path)}"

    cfg, outpath = yaml_sink(mocker, arg_str)  # type: Config, Path
    assert isinstance(outpath, Path)

    # Ensure YAML config written to current dir.
    assert outpath.parent == Path()
    assert outpath.name == str(outpath)
    assert str(outpath) == audio_path.with_suffix(YAML_NAME).name

    # Ensure config paths are valid.
    assert outpath.parent / cfg.master_audio == audio_path


# Integration tests


@pytest.mark.usefixtures("Popen")
def test_load_yaml_another_dir(mocker, Popen):
    """YAML file located in `another/dir` should resolve `master_audio`, `channels[].
    wav_path`, and video `path` from `another/dir`."""

    subdir = "tests"
    wav = "sine440.wav"
    mp4 = "sine440.mp4"
    with pushd(subdir):
        arg_str = f"{wav} -a {wav}"
        cfg, outpath = yaml_sink(mocker, arg_str)  # type: Config, Path

    cfg.begin_time = 100  # To skip all actual rendering

    # Log execution of CorrScope().play()
    Wave = mocker.spy(corrscope.channel, "Wave")

    # Same function as used in cli.py and gui/__init__.py.
    output = cfg.get_ffmpeg_cfg(mp4)

    corr = CorrScope(cfg, Arguments(subdir, [output]))
    corr.play()

    # The .wav path (specified in Config) should be resolved relative to the config
    # file.
    wav_abs = abspath(f"{subdir}/{wav}")

    # The output video path (specified in CLI --render) should be resolved relative to
    # the shell's working directory.
    mp4_abs = abspath(mp4)

    # Test `wave_path`
    args, kwargs = Wave.call_args
    (wave_path,) = args
    assert wave_path == wav_abs

    # Test output `master_audio` and video `path`
    args, kwargs = Popen.call_args
    argv = args[0]
    assert argv[-1] == mp4_abs
    assert f"-i {wav_abs}" in " ".join(argv)
