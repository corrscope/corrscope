import os
import shlex
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple, ContextManager, Callable

import click
import pytest
from click.testing import CliRunner

from ovgenpy import cli
from ovgenpy.config import yaml
from ovgenpy.ovgenpy import Config

if TYPE_CHECKING:
    import pytest_mock


def call_main(args):
    return CliRunner().invoke(cli.main, args, catch_exceptions=False, standalone_mode=False)


# ovgenpy configuration sinks

def write_yaml(command):
    args = shlex.split(command) + ['-w']
    return call_main(args)


def play(command):
    args = shlex.split(command) + ['-p']
    return call_main(args)


# ovgenpy configuration sources

@pytest.mark.parametrize('runner', [write_yaml, play])
def test_no_files(runner):
    with pytest.raises(click.ClickException):
        runner('')


@pytest.mark.parametrize('runner', [write_yaml, play])
def test_cwd(runner):
    runner('.')
