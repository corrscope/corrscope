import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import click
import pytest
from click.testing import CliRunner

import ovgenpy.channel
from ovgenpy import cli
from ovgenpy.cli import YAML_NAME
from ovgenpy.config import yaml
from ovgenpy.ovgenpy import Config, Ovgen
from ovgenpy.util import pushd


if TYPE_CHECKING:
    import pytest_mock


def call_main(argv):
    return CliRunner().invoke(cli.main, argv, catch_exceptions=False, standalone_mode=False)


# ovgenpy configuration sinks

@pytest.fixture
def yaml_sink(mocker: 'pytest_mock.MockFixture') -> Callable:
    def _yaml_sink(command):
        dump = mocker.patch.object(yaml, 'dump')

        argv = shlex.split(command) + ['-w']
        call_main(argv)

        dump.assert_called_once()
        (cfg, stream), kwargs = dump.call_args

        assert isinstance(cfg, Config)
        return cfg, stream
    return _yaml_sink


@pytest.fixture
def player_sink(mocker) -> Callable:
    def _player_sink(command):
        Ovgen = mocker.patch.object(cli, 'Ovgen')

        argv = shlex.split(command) + ['-p']
        call_main(argv)

        Ovgen.assert_called_once()
        (cfg,), kwargs = Ovgen.call_args

        assert isinstance(cfg, Config)
        return cfg,
    return _player_sink


def test_sink_fixture(yaml_sink, player_sink):
    """ Ensure we can use yaml_sink and player_sink as a fixture directly """
    pass


@pytest.fixture(params=[yaml_sink, player_sink])
def any_sink(request, mocker):
    sink = request.param
    return sink(mocker)


# ovgenpy configuration sources

def test_no_files(any_sink):
    with pytest.raises(click.ClickException):
        any_sink('')


@pytest.mark.parametrize('wav_dir', '. tests'.split())
def test_file_dirs(any_sink, wav_dir):
    """ Ensure loading files from `dir` places `dir/*.wav` in config. """
    wavs = Path(wav_dir).glob('*.wav')
    wavs = sorted(str(x) for x in wavs)

    cfg = any_sink(wav_dir)[0]
    assert isinstance(cfg, Config)

    assert [chan.wav_path for chan in cfg.channels] == wavs


def test_write_dir(yaml_sink):
    """ Loading `--audio another/dir` should write YAML to current dir.
    Writing YAML to audio dir: causes relative paths (relative to pwd) to break. """

    audio_path = Path('tests/sine440.wav')
    arg_str = f'tests -a {audio_path}'

    cfg, outpath = yaml_sink(arg_str)   # type: Config, Path
    assert isinstance(outpath, Path)

    # Ensure YAML config written to current dir.
    assert outpath.parent == Path()
    assert outpath.name == str(outpath)
    assert str(outpath) == audio_path.with_suffix(YAML_NAME).name

    # Ensure config paths are valid.
    assert outpath.parent / cfg.master_audio == audio_path


@pytest.fixture
def Wave(mocker):
    """ Logs all calls, and returns a real Wave object. """
    Wave = mocker.spy(ovgenpy.channel, 'Wave')
    yield Wave


@pytest.mark.usefixtures('Popen')
def test_load_yaml_another_dir(yaml_sink, Popen, Wave):
    """ Loading `another/dir/YAML` should resolve `master_audio`, `channels[].wav_path`,
    and video `path` from `another/dir`. """

    with pushd('tests'):
        arg_str = 'sine440.wav -a sine440.wav -o foo.mp4'
        cfg, outpath = yaml_sink(arg_str)   # type: Config, Path

    cfg.begin_time = 100    # To skip all actual rendering
    ovgen = Ovgen(cfg, 'tests')
    ovgen.play()

    # Test `wave_path`
    args, kwargs = Wave.call_args
    cfg, wave_path = args
    assert wave_path == 'tests/sine440.wav'

    # Test output `master_audio` and video `path`
    args, kwargs = Popen.call_args
    argv = args[0]
    assert argv[-1] == 'tests/foo.mp4'
    assert '-i tests/sine440.wav' in ' '.join(argv)


# TODO integration test without --audio
