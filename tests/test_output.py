"""
- Test Output classes.
- Integration tests (see conftest.py).
"""
import errno
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pytest

from corrscope import parallelism
from corrscope.channel import ChannelConfig
from corrscope.corrscope import default_config, Config, CorrScope, Arguments, RenderJob
from corrscope.outputs import (
    RGB_DEPTH,
    FFmpegOutput,
    FFmpegOutputConfig,
    FFplayOutput,
    FFplayOutputConfig,
    Stop,
)
from corrscope.renderer import RendererConfig, MatplotlibRenderer
from corrscope.util import pushd
from tests.test_renderer import RENDER_Y_ZEROS, WIDTH, HEIGHT


if TYPE_CHECKING:
    import pytest_mock
    from unittest.mock import MagicMock
    import py


# Global setup
if not shutil.which("ffmpeg"):
    missing_ffmpeg = True
    pytestmark = pytest.mark.xfail(
        reason="Missing ffmpeg, ignoring failed output tests",
        raises=FileNotFoundError,  # includes MissingFFmpegError
        strict=False,
    )
else:
    missing_ffmpeg = False


def exception_Popen(mocker: "pytest_mock.MockFixture", exc: Exception) -> "MagicMock":
    """Mock Popen to raise an exception."""
    real_Popen = subprocess.Popen

    def popen_factory(*args, **kwargs):
        popen = mocker.create_autospec(real_Popen)

        popen.stdin = mocker.mock_open()(os.devnull, "wb")
        popen.stdout = mocker.mock_open()(os.devnull, "rb")
        assert popen.stdin != popen.stdout

        popen.stdin.write.side_effect = exc
        popen.wait.return_value = 0
        return popen

    Popen = mocker.patch.object(subprocess, "Popen", autospec=True)
    Popen.side_effect = popen_factory
    return Popen


class DummyException(Exception):
    pass


NULL_FFMPEG_OUTPUT = FFmpegOutputConfig(None, "-f null")

render_cfg = RendererConfig(WIDTH, HEIGHT)
CFG = default_config(render=render_cfg)


sine440_wav = os.path.abspath("tests/sine440.wav")


def sine440_config():
    cfg = default_config(
        channels=[ChannelConfig(sine440_wav)],
        master_audio=sine440_wav,
        end_time=0.5,  # Reduce test duration
        render=render_cfg,
    )
    return cfg


## Begin tests
# Calls MatplotlibRenderer, FFmpegOutput, FFmpeg.
def test_render_output():
    """ Ensure rendering to output does not raise exceptions. """
    renderer = MatplotlibRenderer(CFG.render, CFG.layout, nplots=1, channel_cfgs=None)
    out: FFmpegOutput = NULL_FFMPEG_OUTPUT(CFG)

    renderer.render_frame([RENDER_Y_ZEROS])
    out.write_frame(renderer.get_frame())

    assert out.close() == 0


# Calls FFmpegOutput and FFmpeg.
def test_output():
    out: FFmpegOutput = NULL_FFMPEG_OUTPUT(CFG)

    frame = bytes(WIDTH * HEIGHT * RGB_DEPTH)
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
    corr = CorrScope(
        cfg, Arguments(".", [NULL_FFMPEG_OUTPUT], worker=parallelism.SerialWorker)
    )
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
    corr = CorrScope(
        cfg, Arguments(".", [FFplayOutputConfig()], worker=parallelism.SerialWorker)
    )

    render_frame = mocker.patch.object(MatplotlibRenderer, "render_frame")
    render_frame.side_effect = DummyException()
    with pytest.raises(DummyException):
        corr.play()

    # Ensure outputs are terminated.
    render_worker = corr.render_worker
    assert isinstance(render_worker, parallelism.SerialWorker)

    child_job = render_worker.child_job
    assert isinstance(child_job, RenderJob)

    assert len(child_job.outputs) == 1
    output: FFplayOutput = child_job.outputs[0]

    for popen in output._pipeline:
        popen.terminate.assert_called()


# Integration: Calls CorrScope and FFplay.
@pytest.mark.skip("Launches ffmpeg and ffplay processes, creating a ffplay window")
def test_corr_terminate_works():
    """ Ensure that ffmpeg/ffplay terminate quickly after Python exceptions, when
    `popen.terminate()` is called. """

    cfg = sine440_config()
    corr = CorrScope(
        cfg, Arguments(".", [FFplayOutputConfig()], worker=parallelism.SerialWorker)
    )
    corr.raise_on_teardown = DummyException

    with pytest.raises(DummyException):
        # Raises `subprocess.TimeoutExpired` if popen.terminate() doesn't work.
        corr.play()


# Simulate user closing ffplay window.
# Why OSError? See comment at PipeOutput.write_frame().
# Calls FFplayOutput, mocks Popen.
@pytest.mark.parametrize("errno_id", [errno.EPIPE, errno.EINVAL])
def test_closing_ffplay_stops_main(mocker: "pytest_mock.MockFixture", errno_id):
    """ Closing FFplay should make FFplayOutput.write_frame() return Stop
    to main loop. """

    # Create mocks.
    exc = OSError(errno_id, "Simulated ffplay-closed error")
    if errno_id == errno.EPIPE:
        assert type(exc) == BrokenPipeError

    # Yo Mock, I herd you like not working properly,
    # so I put a test in your test so I can test your mocks while I test my code.
    Popen = exception_Popen(mocker, exc)
    assert Popen is subprocess.Popen
    assert Popen.side_effect

    # Launch corrscope
    with FFplayOutputConfig()(CFG) as output:
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

    corr = CorrScope(
        cfg, Arguments(".", [NULL_FFMPEG_OUTPUT], worker=parallelism.SerialWorker)
    )
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

    corr = CorrScope(
        cfg, Arguments(".", [DummyOutputConfig()], worker=parallelism.SerialWorker)
    )
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

    corr = CorrScope(
        cfg, Arguments(".", [NULL_FFMPEG_OUTPUT], worker=parallelism.SerialWorker)
    )
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
    get_frame = mocker.spy(MatplotlibRenderer, "get_frame")
    previews, records = previews_records(mocker)

    cfg = cfg_192x108()
    corr = CorrScope(cfg, Arguments(".", outputs, worker=parallelism.SerialWorker))
    corr.play()

    # Check that only before_preview() called.
    for p in previews:
        p.assert_called()
    for r in records:
        r.assert_not_called()

    # Check renderer is 128x72. (CorrScope() mutates cfg)
    assert cfg.render.width == 128
    assert cfg.render.height == 72

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
    get_frame = mocker.spy(MatplotlibRenderer, "get_frame")
    previews, records = previews_records(mocker)

    cfg = cfg_192x108()
    corr = CorrScope(cfg, Arguments(".", outputs, worker=parallelism.SerialWorker))
    corr.play()

    # Check that only before_record() called.
    for p in previews:
        p.assert_not_called()
    for r in records:
        r.assert_called()

    # Check renderer is 192x108. (CorrScope() mutates cfg)
    assert cfg.render.width == 192
    assert cfg.render.height == 108

    # Ensure subfps is disabled.
    assert get_frame.call_count == round(cfg.end_time * cfg.fps) + 1 == 31


# Integration test: Output and ParallelWorker
@pytest.mark.timeout(3)
@pytest.mark.xfail(
    condition=missing_ffmpeg,
    reason="Missing ffmpeg, parent_send() will receive Error and exit(1)",
    raises=SystemExit,
)
@pytest.mark.parametrize("profile_name", [None, "test_output_parallel--profile"])
def test_output_parallel(profile_name: Optional[str], tmpdir: "py.path.local"):
    """ Ensure output doesn't deadlock/etc.
    Ideally I'd make assertions on the communication protocol,
    but spying on the Connection call arguments is hard.
    """
    with pushd(tmpdir):  # converted to str(tmpdir)
        cfg = sine440_config()
        arg = Arguments(
            ".",
            [NULL_FFMPEG_OUTPUT],
            profile_name=profile_name,
            worker=parallelism.ParallelWorker,
        )
        corr = CorrScope(cfg, arg)
        corr.play()


"""
FIXME add tests... I have no confidence that parallelism works properly:

- If parent raises exception, should send Error to child.
    - if child gets Error, it should terminate.
- If child raises exception, should send exc OR traceback to parent.
    - if parent receives exception/traceback, should raise it (show to cli/gui).

This needs 4 separate unit tests = (2 parent and 2 child) tests.
- since pytest can't mock objects in the child process.
"""
