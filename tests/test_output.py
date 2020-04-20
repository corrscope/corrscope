"""
- Test Output classes.
- Integration tests (see conftest.py).
"""
import errno
import shutil
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import attr
import pytest

from corrscope.channel import ChannelConfig
from corrscope.corrscope import template_config, Config, CorrScope, Arguments
from corrscope.outputs import (
    FFmpegOutput,
    FFmpegOutputConfig,
    FFplayOutput,
    FFplayOutputConfig,
    Stop,
)
from corrscope.renderer import RendererConfig, Renderer, RenderInput
from tests.test_renderer import RENDER_Y_ZEROS, WIDTH, HEIGHT

if TYPE_CHECKING:
    import pytest_mock


parametrize = pytest.mark.parametrize
BYTES_PER_PIXEL = Renderer.bytes_per_pixel


# Global setup
if not shutil.which("ffmpeg"):
    pytestmark = pytest.mark.xfail(
        reason="Missing ffmpeg, ignoring failed output tests",
        raises=FileNotFoundError,  # includes MissingFFmpegError
        strict=False,
    )


class DummyException(Exception):
    pass


NULL_FFMPEG_OUTPUT = FFmpegOutputConfig(None, "-f null")

render_cfg = RendererConfig(WIDTH, HEIGHT)
CFG = template_config(render=render_cfg)


def sine440_config():
    cfg = template_config(
        channels=[ChannelConfig("tests/sine440.wav")],
        master_audio="tests/sine440.wav",
        end_time=0.5,  # Reduce test duration
        render=render_cfg,
    )
    return cfg


## Begin tests
# Calls MatplotlibRenderer, FFmpegOutput, FFmpeg.
def test_render_output():
    """ Ensure rendering to output does not raise exceptions. """
    datas = [RENDER_Y_ZEROS]

    renderer = Renderer(CFG.render, CFG.layout, datas, None, None)
    out: FFmpegOutput = NULL_FFMPEG_OUTPUT(CFG)

    renderer.update_main_lines(RenderInput.wrap_datas(datas))
    out.write_frame(renderer.get_frame())

    assert out.close() == 0


# Calls FFmpegOutput and FFmpeg.
def test_output():
    out: FFmpegOutput = NULL_FFMPEG_OUTPUT(CFG)

    frame = bytes(WIDTH * HEIGHT * BYTES_PER_PIXEL)
    out.write_frame(frame)

    assert out.close() == 0
    # Ensure video is written to stdout, and not current directory.
    assert not Path("-").exists()


## Ensure CorrScope closes pipe to output upon completion.
# Calls FFplayOutput, mocks Popen.
@pytest.mark.usefixtures("Popen")
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


# Calls CorrScope, mocks FFmpegOutput.
def test_corrscope_main_uses_contextmanager(mocker: "pytest_mock.MockFixture"):
    """ Ensure CorrScope() main wraps output in context manager. """
    FFmpegOutput = mocker.patch.object(FFmpegOutputConfig, "cls")
    output = FFmpegOutput.return_value

    cfg = sine440_config()
    cfg.master_audio = None
    corr = CorrScope(cfg, Arguments(".", [NULL_FFMPEG_OUTPUT]))
    corr.play()

    FFmpegOutput.assert_called()
    output.__enter__.assert_called()
    output.__exit__.assert_called()


# Calls FFplayOutput, mocks Popen.
@pytest.mark.usefixtures("Popen")
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


# Integration: Calls CorrScope, mocks Popen.
@pytest.mark.usefixtures("Popen")
def test_corr_terminate_ffplay(Popen, mocker: "pytest_mock.MockFixture"):
    """ Integration test: Ensure corrscope calls terminate() on ffmpeg and ffplay when
    Python exceptions occur. """

    cfg = sine440_config()
    corr = CorrScope(cfg, Arguments(".", [FFplayOutputConfig()]))

    update_main_lines = mocker.patch.object(Renderer, "update_main_lines")
    update_main_lines.side_effect = DummyException()
    with pytest.raises(DummyException):
        corr.play()

    assert len(corr.outputs) == 1
    output: FFplayOutput = corr.outputs[0]

    for popen in output._pipeline:
        popen.terminate.assert_called()


# Integration: Calls CorrScope and FFplay.


@attr.dataclass(kw_only=True)
class TestMode:
    should_abort: bool = False
    should_raise: bool = False


@parametrize(
    "test", [TestMode(should_abort=True), TestMode(should_raise=True)], ids=str
)
def test_corr_terminate_works(test):
    """
    Ensure that output exits quickly after output.terminate() is called.

    What calls output.terminate() -> popen.terminate()?

    - Cancelling a GUI render sets is_aborted()=True.
    - corrscope may throw an exception.

    Either way, ffmpeg should be terminated so it stops writing audio.
    """

    import sys
    import subprocess
    from corrscope.outputs import IOutputConfig, register_output, PipeOutput

    class StayOpenOutputConfig(IOutputConfig):
        pass

    @register_output(StayOpenOutputConfig)
    class StayOpenOutput(PipeOutput):
        def __init__(self, corr_cfg: "Config", cfg: StayOpenOutputConfig):
            super().__init__(corr_cfg, cfg)

            sleep_process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                stdin=subprocess.PIPE,
            )
            self.open(sleep_process)

    def is_aborted() -> bool:
        if test.should_raise:
            raise DummyException
        return test.should_abort

    cfg = sine440_config()
    arg = Arguments(".", [StayOpenOutputConfig()], is_aborted=is_aborted)
    corr = CorrScope(cfg, arg)

    if test.should_raise:
        with pytest.raises(DummyException):
            # Raises `subprocess.TimeoutExpired` if popen.terminate() doesn't work.
            corr.play()

    else:
        # Raises `subprocess.TimeoutExpired` if popen.terminate() doesn't work.
        corr.play()


# Simulate user closing ffplay window.
# Why OSError? See comment at PipeOutput.write_frame().
# Calls FFplayOutput, mocks Popen.
@pytest.mark.usefixtures("Popen")
@pytest.mark.parametrize("errno_id", [errno.EPIPE, errno.EINVAL])
def test_closing_ffplay_stops_main(Popen, errno_id):
    """ Closing FFplay should make FFplayOutput.write_frame() return Stop
    to main loop. """

    # Create mocks.
    exc = OSError(errno_id, "Simulated ffplay-closed error")
    if errno_id == errno.EPIPE:
        assert type(exc) == BrokenPipeError

    Popen.set_exception(exc)
    assert Popen.side_effect

    # Launch corrscope
    with FFplayOutputConfig()(CFG) as output:
        # Writing to Popen instance raises exc.
        ret = output.write_frame(b"")

    # Ensure FFplayOutput catches OSError.
    # Also ensure it returns Stop after exception.
    assert ret is Stop, ret


## Integration tests (calls CorrScope and FFmpeg).
# Duplicate test test_no_audio() removed.
def test_corr_output_without_audio():
    """Ensure running CorrScope with FFmpeg output, with master audio disabled,
    does not crash.
    """
    cfg = sine440_config()
    cfg.master_audio = None

    corr = CorrScope(cfg, Arguments(".", [NULL_FFMPEG_OUTPUT]))
    # Should not raise exception.
    corr.play()


# Test framerate subsampling
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

    corr = CorrScope(cfg, Arguments(".", [DummyOutputConfig()]))
    corr.play()
    assert DummyOutput.frames_written >= 2


def test_render_subfps_non_integer(mocker: "pytest_mock.MockFixture"):
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

    corr = CorrScope(cfg, Arguments(".", [NULL_FFMPEG_OUTPUT]))
    corr.play()

    # But it seems FFmpeg actually allows decimal -framerate (although a bad idea).
    # from corrscope.corrscope import Config
    # render_fps = mocker.patch.object(Config, 'render_fps',
    #                                  new_callable=mocker.PropertyMock)
    # render_fps.return_value = 60 / 7
    # assert isinstance(cfg.render_fps, float)
    # corr = CorrScope(cfg, '.', outputs=[NULL_FFMPEG_OUTPUT])
    # corr.play()


# Possibility: add a test to ensure that we render slightly ahead in time
# when subfps>1, to avoid frames lagging behind audio.


## Tests for Output-dependent performance options
def cfg_192x108():
    """ Return config which reduces rendering workload when previewing. """
    cfg = sine440_config()

    # Skip frames.
    assert cfg.end_time == 0.5
    cfg.render_subfps = 2

    # Divide dimensions.
    cfg.render.width = 192
    cfg.render.height = 108
    cfg.render.res_divisor = 1.5

    return cfg


def previews_records(mocker):
    """Returns 2 lists of method MagicMock."""
    configs = (Config, RendererConfig)

    previews = [mocker.spy(cls, "before_preview") for cls in configs]
    records = [mocker.spy(cls, "before_record") for cls in configs]
    return previews, records


NO_FFMPEG = [[], [FFplayOutputConfig()]]


@pytest.mark.usefixtures("Popen")  # Prevents FFplayOutput from launching processes.
@pytest.mark.parametrize("outputs", NO_FFMPEG)
def test_preview_performance(Popen, mocker: "pytest_mock.MockFixture", outputs):
    """ Ensure performance optimizations enabled
    if all outputs are FFplay or others. """
    get_frame = mocker.spy(Renderer, "get_frame")
    previews, records = previews_records(mocker)

    cfg = cfg_192x108()
    corr = CorrScope(cfg, Arguments(".", outputs))

    # Run corrscope main loop.
    corr.play()

    # Check that only before_preview() called.
    for p in previews:
        p.assert_called()
    for r in records:
        r.assert_not_called()

    # Check renderer is 128x72
    assert corr.renderer.w == 128
    assert corr.renderer.h == 72

    # Ensure subfps is enabled (only odd frames are rendered, 1..29).
    # See CorrScope `should_render` variable.
    assert (
        get_frame.call_count == round(cfg.end_time * cfg.fps / cfg.render_subfps) == 15
    )


YES_FFMPEG = [l + [FFmpegOutputConfig(None)] for l in NO_FFMPEG]


@pytest.mark.usefixtures("Popen")
@pytest.mark.parametrize("outputs", YES_FFMPEG)
def test_record_performance(Popen, mocker: "pytest_mock.MockFixture", outputs):
    """ Ensure performance optimizations disabled
    if any FFmpegOutputConfig is found. """
    get_frame = mocker.spy(Renderer, "get_frame")
    previews, records = previews_records(mocker)

    cfg = cfg_192x108()
    corr = CorrScope(cfg, Arguments(".", outputs))
    corr.play()

    # Check that only before_record() called.
    for p in previews:
        p.assert_not_called()
    for r in records:
        r.assert_called()

    # Check renderer is 192x108
    assert corr.renderer.cfg.width == 192
    assert corr.renderer.cfg.height == 108

    # Ensure subfps is disabled.
    assert get_frame.call_count == round(cfg.end_time * cfg.fps) + 1 == 31
