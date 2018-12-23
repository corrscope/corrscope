import shutil
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from corrscope.channel import ChannelConfig
from corrscope.outputs import RGB_DEPTH, \
    FFmpegOutput, FFmpegOutputConfig, FFplayOutput, FFplayOutputConfig
from corrscope.corrscope import default_config, CorrScope, Arguments
from corrscope.renderer import RendererConfig, MatplotlibRenderer
from tests.test_renderer import WIDTH, HEIGHT, ALL_ZEROS

if TYPE_CHECKING:
    import pytest_mock


if not shutil.which('ffmpeg'):
    pytestmark = pytest.mark.skip('Missing ffmpeg, skipping output tests')


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


# Ensure CorrScope closes pipe to output upon completion.
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


# Ensure CorrScope terminates FFplay upon exceptions.
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
def test_corr_terminate_ffplay(Popen, mocker: 'pytest_mock.MockFixture'):
    """ Integration test: Ensure corrscope calls terminate() on ffmpeg and ffplay when
    Python exceptions occur. """

    cfg = sine440_config()
    corr = CorrScope(cfg, Arguments('.', [FFplayOutputConfig()]))

    render_frame = mocker.patch.object(MatplotlibRenderer, 'render_frame')
    render_frame.side_effect = DummyException()
    with pytest.raises(DummyException):
        corr.play()

    assert len(corr.outputs) == 1
    output: FFplayOutput = corr.outputs[0]

    for popen in output._pipeline:
        popen.terminate.assert_called()


@pytest.mark.skip('Launches ffmpeg and ffplay processes, creating a ffplay window')
def test_corr_terminate_works():
    """ Ensure that ffmpeg/ffplay terminate quickly after Python exceptions, when
    `popen.terminate()` is called. """

    cfg = sine440_config()
    corr = CorrScope(cfg, Arguments('.', [FFplayOutputConfig()]))
    corr.raise_on_teardown = DummyException

    with pytest.raises(DummyException):
        # Raises `subprocess.TimeoutExpired` if popen.terminate() doesn't work.
        corr.play()


# TODO test to ensure ffplay is killed before it terminates

def test_corr_output_without_audio():
    """Ensure running CorrScope with FFmpeg output, with master audio disabled,
    does not crash.
    """
    cfg = sine440_config()
    cfg.master_audio = None

    corr = CorrScope(cfg, Arguments('.', [NULL_OUTPUT]))
    # Should not raise exception.
    corr.play()


def test_render_subfps_one():
    """ Ensure video gets rendered when render_subfps=1.
    This test fails if ceildiv is used to calculate `ahead`.
    """
    from corrscope.outputs import IOutputConfig, Output, register_output

    # region DummyOutput
    class DummyOutputConfig(IOutputConfig):
        pass

    @register_output(DummyOutputConfig)
    class DummyOutput(Output):
        frames_written = 0

        @classmethod
        def write_frame(cls, frame: bytes) -> None:
            cls.frames_written += 1

    assert DummyOutput
    # endregion

    # Create CorrScope with render_subfps=1. Ensure multiple frames are outputted.
    cfg = sine440_config()
    cfg.render_subfps = 1

    corr = CorrScope(cfg, Arguments('.', [DummyOutputConfig()]))
    corr.play()
    assert DummyOutput.frames_written >= 2


def test_render_subfps_non_integer(mocker: 'pytest_mock.MockFixture'):
    """ Ensure we output non-integer subfps as fractions,
    and that ffmpeg doesn't crash.
    TODO does ffmpeg understand decimals??
    """

    cfg = sine440_config()
    cfg.fps = 60
    cfg.render_subfps = 7

    # By default, we output render_fps (ffmpeg -framerate) as a fraction.
    assert isinstance(cfg.render_fps, Fraction)
    assert cfg.render_fps != int(cfg.render_fps)
    assert Fraction(1) == int(1)

    corr = CorrScope(cfg, Arguments('.', [NULL_OUTPUT]))
    corr.play()

    # But it seems FFmpeg actually allows decimal -framerate (although a bad idea).
    # from corrscope.corrscope import Config
    # render_fps = mocker.patch.object(Config, 'render_fps',
    #                                  new_callable=mocker.PropertyMock)
    # render_fps.return_value = 60 / 7
    # assert isinstance(cfg.render_fps, float)
    # corr = CorrScope(cfg, '.', outputs=[NULL_OUTPUT])
    # corr.play()


# Possibility: add a test to ensure that we render slightly ahead in time
# when subfps>1, to avoid frames lagging behind audio.


class DummyException(Exception):
    pass
