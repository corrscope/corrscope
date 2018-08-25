import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from ovgenpy.channel import ChannelConfig
from ovgenpy.outputs import RGB_DEPTH, \
    FFmpegOutput, FFmpegOutputConfig, FFplayOutput, FFplayOutputConfig
from ovgenpy.ovgenpy import default_config, Ovgen
from ovgenpy.renderer import RendererConfig, MatplotlibRenderer
from tests.test_renderer import WIDTH, HEIGHT, ALL_ZEROS

if TYPE_CHECKING:
    import pytest_mock

CFG = default_config(render=RendererConfig(WIDTH, HEIGHT))
STDOUT_CFG = FFmpegOutputConfig('-', '-f nut')


def test_render_output():
    """ Ensure rendering to output does not raise exceptions. """
    renderer = MatplotlibRenderer(CFG.render, CFG.layout, nplots=1)
    output_cfg = FFmpegOutputConfig('-', '-f nut')
    out = FFmpegOutput(CFG, output_cfg)

    renderer.render_frame([ALL_ZEROS])
    out.write_frame(renderer.get_frame())

    assert out.close() == 0


def test_output():
    out = FFmpegOutput(CFG, STDOUT_CFG)

    frame = bytes(WIDTH * HEIGHT * RGB_DEPTH)
    out.write_frame(frame)

    assert out.close() == 0


# Ensure ovgen terminates FFplay upon exceptions.


def test_terminate_ffplay(Popen):
    """ FFplayOutput unit test: Ensure ffmpeg and ffplay are terminated when Python
    exceptions occur.
    """

    ffplay_cfg = FFplayOutputConfig()
    try:
        output: FFplayOutput
        with ffplay_cfg(CFG) as output:
            raise DummyException

    except DummyException:
        for popen in output._pipeline:
            popen.terminate.assert_called()


def test_ovgen_terminate_ffplay(Popen, mocker: 'pytest_mock.MockFixture'):
    """ Integration test: Ensure ffmpeg and ffplay are terminated when Python exceptions
    occur. """

    cfg = default_config(
        channels=[ChannelConfig('tests/sine440.wav')],
        master_audio='tests/sine440.wav',
        outputs=[FFplayOutputConfig()]
    )
    ovgen = Ovgen(cfg)

    render_frame = mocker.patch.object(MatplotlibRenderer, 'render_frame')
    render_frame.side_effect = DummyException()
    with pytest.raises(DummyException):
        ovgen.play()

    assert len(ovgen.outputs) == 1
    output: FFplayOutput = ovgen.outputs[0]

    for popen in output._pipeline:
        popen.terminate.assert_called()


# TODO integration test without audio

# TODO integration test on ???


@pytest.fixture
def Popen(mocker: 'pytest_mock.MockFixture'):
    Popen = mocker.patch.object(subprocess, 'Popen', autospec=True).return_value

    Popen.stdin = open(os.devnull, "wb")
    Popen.stdout = open(os.devnull, "rb")
    Popen.wait.return_value = 0

    yield Popen


class DummyException(Exception):
    pass
