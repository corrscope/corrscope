import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytest
from click.testing import CliRunner

from ovgenpy import cli
from ovgenpy.config import yaml
from ovgenpy.ovgenpy import Config
from ovgenpy.util import curry

if TYPE_CHECKING:
    import pytest_mock


def call_main(args):
    return CliRunner().invoke(cli.main, args, catch_exceptions=False, standalone_mode=False)


# ovgenpy configuration sinks

@pytest.fixture
@curry
def yaml_sink(mocker: 'pytest_mock.MockFixture', command):
    dump = mocker.patch.object(yaml, 'dump')

    args = shlex.split(command) + ['-w']
    call_main(args)

    dump.assert_called_once()
    args, kwargs = dump.call_args

    cfg = args[0]     # yaml.dump(cfg, out)
    assert isinstance(cfg, Config)
    return cfg


@pytest.fixture
@curry
def player_sink(mocker, command):
    Ovgen = mocker.patch.object(cli, 'Ovgen')

    args = shlex.split(command) + ['-p']
    call_main(args)

    Ovgen.assert_called_once()
    args, kwargs = Ovgen.call_args

    cfg = args[0]   # Ovgen(cfg)
    assert isinstance(cfg, Config)
    return cfg


@pytest.fixture(params=[yaml_sink, player_sink])
def any_sink(request, mocker):
    sink = request.param
    return sink(mocker)


# ovgenpy configuration sources

def test_no_files(any_sink):
    with pytest.raises(click.ClickException):
        any_sink('')


@pytest.mark.parametrize('folder', '. wav-formats'.split())
def test_cwd(any_sink, folder, mocker):
    """ wav_prefix"""
    wavs = Path(folder).glob('*.wav')
    wavs = sorted(x.name for x in wavs)

    cfg = any_sink(folder)
    assert isinstance(cfg, Config)

    assert cfg.wav_prefix == folder
    assert [chan.wav_path for chan in cfg.channels] == wavs
