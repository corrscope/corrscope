from pathlib import Path
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
NULL_CFG = FFmpegOutputConfig(None, '-f null')


def test_render_output():
    """ Ensure rendering to output does not raise exceptions. """
    renderer = MatplotlibRenderer(CFG.render, CFG.layout, nplots=1)
    out: FFmpegOutput = NULL_CFG(CFG)

    renderer.render_frame([ALL_ZEROS])
    out.write_frame(renderer.get_frame())

    assert out.close() == 0


def test_output():
    out: FFmpegOutput = NULL_CFG(CFG)

    frame = bytes(WIDTH * HEIGHT * RGB_DEPTH)
    out.write_frame(frame)

    assert out.close() == 0
    # Ensure video is written to stdout, and not current directory.
    assert not Path('-').exists()


# Ensure ovgen closes pipe to output upon completion.
@pytest.mark.usefixtures('Popen')
def test_close_output(Popen):
    """ FFplayOutput unit test: Ensure ffmpeg and ffplay are terminated when Python
    exceptions occur.
    """

    ffplay_cfg = FFplayOutputConfig()
    output: FFplayOutput
    with ffplay_cfg(CFG) as output:
        pass

    output._pipeline[0].stdin.close.assert_called()
    for popen in output._pipeline:
        popen.wait.assert_called()  # Does wait() need to be called?


# Ensure ovgen terminates FFplay upon exceptions.
@pytest.mark.usefixtures('Popen')
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


@pytest.mark.usefixtures('Popen')
def test_ovgen_terminate_ffplay(Popen, mocker: 'pytest_mock.MockFixture'):
    """ Integration test: Ensure ovgenpy calls terminate() on ffmpeg and ffplay when
    Python exceptions occur. """

    cfg = default_config(
        channels=[ChannelConfig('tests/sine440.wav')],
        master_audio='tests/sine440.wav',
    )
    ovgen = Ovgen(cfg, '.', outputs=[FFplayOutputConfig()])

    render_frame = mocker.patch.object(MatplotlibRenderer, 'render_frame')
    render_frame.side_effect = DummyException()
    with pytest.raises(DummyException):
        ovgen.play()

    assert len(ovgen.outputs) == 1
    output: FFplayOutput = ovgen.outputs[0]

    for popen in output._pipeline:
        popen.terminate.assert_called()


@pytest.mark.skip('Launches ffmpeg and ffplay processes, creating a ffplay window')
def test_ovgen_terminate_works():
    """ Ensure that ffmpeg/ffplay terminate quickly after Python exceptions, when
    `popen.terminate()` is called. """

    cfg = default_config(
        channels=[ChannelConfig('tests/sine440.wav')],
        master_audio='tests/sine440.wav',
        outputs=[FFplayOutputConfig()],
        end_time=0.5,   # Reduce test duration
    )
    ovgen = Ovgen(cfg, '.')
    ovgen.raise_on_teardown = DummyException

    with pytest.raises(DummyException):
        # Raises `subprocess.TimeoutExpired` if popen.terminate() doesn't work.
        ovgen.play()


# TODO integration test without audio

# TODO integration test on ???


class DummyException(Exception):
    pass
