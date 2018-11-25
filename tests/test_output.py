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
NULL_OUTPUT = FFmpegOutputConfig(None, '-f null')


def test_render_output():
    """ Ensure rendering to output does not raise exceptions. """
    renderer = MatplotlibRenderer(CFG.render, CFG.layout, nplots=1, channel_cfgs=None)
    out: FFmpegOutput = NULL_OUTPUT(CFG)

    renderer.render_frame([ALL_ZEROS])
    out.write_frame(renderer.get_frame())

    assert out.close() == 0


def test_output():
    out: FFmpegOutput = NULL_OUTPUT(CFG)

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


def sine440_config():
    cfg = default_config(
        channels=[ChannelConfig('tests/sine440.wav')],
        master_audio='tests/sine440.wav',
        end_time=0.5,  # Reduce test duration
    )
    return cfg


@pytest.mark.usefixtures('Popen')
def test_ovgen_terminate_ffplay(Popen, mocker: 'pytest_mock.MockFixture'):
    """ Integration test: Ensure ovgenpy calls terminate() on ffmpeg and ffplay when
    Python exceptions occur. """

    cfg = sine440_config()
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

    cfg = sine440_config()
    ovgen = Ovgen(cfg, '.', outputs=[FFplayOutputConfig()])
    ovgen.raise_on_teardown = DummyException

    with pytest.raises(DummyException):
        # Raises `subprocess.TimeoutExpired` if popen.terminate() doesn't work.
        ovgen.play()


# TODO test to ensure ffplay is killed before it terminates

def test_ovgen_output_without_audio():
    """Ensure running ovgen with FFmpeg output, with master audio disabled,
    does not crash.
    """
    cfg = sine440_config()
    cfg.master_audio = None

    ovgen = Ovgen(cfg, '.', outputs=[NULL_OUTPUT])
    # Should not raise exception.
    ovgen.play()


def test_render_subfps_one():
    """ Ensure video gets rendered when render_subfps=1.
    This test fails if ceildiv is used to calculate `ahead`.
    """
    pass #TODO


def test_render_subfps_non_integer():
    """ Ensure we output non-integer subfps as fractions,
    and that ffmpeg doesn't crash.
    TODO does ffmpeg understand decimals??
    """


def test_render_subfps_ahead():
    """ Ensure that we render slightly ahead in time when subfps>1,
    to avoid frames lagging behind audio. """
# TODO test render subfps
# including integer and non-integer rations
#


class DummyException(Exception):
    pass
