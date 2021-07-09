import errno
import shlex
import subprocess
from abc import ABC, abstractmethod
from os.path import abspath
from typing import TYPE_CHECKING, Type, List, Union, Optional, ClassVar, Callable

from corrscope.config import DumpableAttrs
from corrscope.renderer import ByteBuffer, Renderer
from corrscope.settings.paths import MissingFFmpegError

if TYPE_CHECKING:
    from corrscope.corrscope import Config


FRAMES_TO_BUFFER = 2

FFMPEG_QUIET = "-nostats -hide_banner -loglevel error".split()


class IOutputConfig(DumpableAttrs):
    cls: "ClassVar[Type[Output]]"

    def __call__(self, corr_cfg: "Config") -> "Output":
        """Must be called in the .yaml file's directory.
        This is used to properly resolve corr_cfg.master_audio."""
        return self.cls(corr_cfg, cfg=self)


class _Stop:
    pass


Stop = _Stop()


class Output(ABC):
    def __init__(self, corr_cfg: "Config", cfg: IOutputConfig):
        self.corr_cfg = corr_cfg
        self.cfg = cfg

        rcfg = corr_cfg.render

        frame_bytes = (
            rcfg.divided_height * rcfg.divided_width * Renderer.bytes_per_pixel
        )
        self.bufsize = frame_bytes * FRAMES_TO_BUFFER

    def __enter__(self):
        return self

    @abstractmethod
    def write_frame(self, frame: ByteBuffer) -> Optional[_Stop]:
        """Output a Numpy ndarray."""

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def terminate(self, from_same_thread: bool = True) -> None:
        pass


# Glue logic


def register_output(
    config_t: Type[IOutputConfig],
) -> Callable[[Type[Output]], Type[Output]]:
    def inner(output_t: Type[Output]):
        config_t.cls = output_t
        return output_t

    return inner


# FFmpeg command line generation


class _FFmpegProcess:
    def __init__(self, templates: List[str], corr_cfg: "Config"):
        self.templates = templates
        self.corr_cfg = corr_cfg

        self.templates += ffmpeg_input_video(corr_cfg)  # video
        if corr_cfg.master_audio:
            # Load master audio and trim to timestamps.

            self.templates.append(f"-ss {corr_cfg.begin_time}")

            audio_path = shlex.quote(abspath(corr_cfg.master_audio))
            self.templates += ffmpeg_input_audio(audio_path)  # audio

            if corr_cfg.end_time is not None:
                dur = corr_cfg.end_time - corr_cfg.begin_time
                self.templates.append(f"-to {dur}")

    def add_output(self, cfg: "Union[FFmpegOutputConfig, FFplayOutputConfig]") -> None:
        self.templates.append(cfg.video_template)  # video
        if self.corr_cfg.master_audio:
            self.templates.append(cfg.audio_template)  # audio

    def popen(self, extra_args: List[str], bufsize: int, **kwargs) -> subprocess.Popen:
        """Raises FileNotFoundError if FFmpeg missing"""
        try:
            args = self._generate_args() + extra_args
            return subprocess.Popen(
                args, stdin=subprocess.PIPE, bufsize=bufsize, **kwargs
            )
        except FileNotFoundError as e:
            raise MissingFFmpegError() from e

    def _generate_args(self) -> List[str]:
        return [arg for template in self.templates for arg in shlex.split(template)]


def ffmpeg_input_video(cfg: "Config") -> List[str]:
    fps = cfg.render_fps
    width = cfg.render.divided_width
    height = cfg.render.divided_height

    return [
        f"-f rawvideo -pixel_format {Renderer.ffmpeg_pixel_format}",
        f"-video_size {width}x{height} -framerate {fps}",
        *FFMPEG_QUIET,
        "-i -",
    ]


def ffmpeg_input_audio(audio_path: str) -> List[str]:
    return ["-i", audio_path]


class PipeOutput(Output):
    def open(self, *pipeline: subprocess.Popen) -> None:
        """Called by __init__ with a Popen pipeline to ffmpeg/ffplay."""
        if len(pipeline) == 0:
            raise TypeError("must provide at least one Popen argument to popens")

        self._pipeline = pipeline
        self._stream = pipeline[0].stdin
        # Python documentation discourages accessing popen.stdin. It's wrong.
        # https://stackoverflow.com/a/9886747

    def __enter__(self) -> Output:
        return self

    def write_frame(self, frame: ByteBuffer) -> Optional[_Stop]:
        try:
            self._stream.write(frame)
            return None

        # Exception handling taken from Popen._stdin_write().
        # https://bugs.python.org/issue35754
        except BrokenPipeError:
            return Stop  # communicate() must ignore broken pipe errors.
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                # bpo-19612, bpo-30418: On Windows, stdin.write() fails
                # with EINVAL if the child process exited or if the child
                # process is still running but closed the pipe.
                return Stop
            else:
                raise

    def close(self, wait: bool = True) -> int:
        try:
            self._stream.close()
        # technically it should match the above exception handler,
        # but I personally don't care about exceptions when *closing* a pipe.
        except OSError:
            pass

        if not wait:
            return 0

        retval = 0
        for popen in self._pipeline:
            retval |= popen.wait()
        return retval  # final value

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            self.terminate()

    def terminate(self, from_same_thread: bool = True) -> None:
        # Calling self.close() is bad.
        # If exception occurred but ffplay continues running,
        # popen.wait() will prevent stack trace from showing up.
        if from_same_thread:
            self.close(wait=False)
        # If terminating from another thread,
        # possibly not thread-safe to call self._stream.close().

        # Terminate all processes.
        # If blocked on reading, must close pipe and terminate from front to back.
        # If blocked on writing, must terminate from back to front.
        # If we terminate everything, both cases should work.
        for popen in self._pipeline:
            popen.terminate()

        exc = None
        for popen in self._pipeline:
            # https://stackoverflow.com/a/49038779/2683842
            try:
                popen.wait(1)  # timeout=seconds
            except subprocess.TimeoutExpired as e:
                # gee thanks Python, https://stackoverflow.com/questions/45292479/
                exc = e
                popen.kill()

        if exc:
            raise exc


# FFmpegOutput


class FFmpegOutputConfig(IOutputConfig):
    # path=None writes to stdout.
    #
    # This parameter is not loaded from disk, but set when the user picks a render path
    # on the GUI or calls the CLI with `--render out.mp4`.
    #
    # It must be an absolute path. How is this ensured?
    #
    # - Paths supplied from the GUI are always absolute.
    # - Paths supplied from the CLI must be resolved before storing in
    #   FFmpegOutputConfig.
    #
    # Why are relative paths not allowed? Currently, to resolve `corr_cfg.master_audio`
    # relative to the config file, we change directories to the config dir, then call
    # `abspath(corr_cfg.master_audio)`. As a result, if we called `corr dir/cfg.yaml -r
    # out.mp4` and corrscope didn't resolve `out.mp4` before passing into
    # FFmpegOutputConfig, it would mistakenly write to `dir/out.mp4`.
    #
    # In the future, relative paths could be allowed by not switching directories to the
    # config dir, and finding another way to resolve `corr_cfg.master_audio` based on
    # the config dir.
    path: Optional[str]
    args: str = ""

    video_template: str = (
        "-c:v libx264 -crf 18 -preset superfast "
        "-pix_fmt yuv420p -vf scale=out_color_matrix=bt709 "
        "-color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 "
        "-movflags faststart"
    )
    audio_template: str = "-c:a aac -b:a 384k"


FFMPEG = "ffmpeg"


@register_output(FFmpegOutputConfig)
class FFmpegOutput(PipeOutput):
    def __init__(self, corr_cfg: "Config", cfg: FFmpegOutputConfig):
        super().__init__(corr_cfg, cfg)

        ffmpeg = _FFmpegProcess([FFMPEG, "-y"], corr_cfg)
        ffmpeg.add_output(cfg)
        ffmpeg.templates.append(cfg.args)

        if cfg.path is None:
            video_path = "-"  # Write to stdout
        else:
            video_path = abspath(cfg.path)

        self.open(ffmpeg.popen([video_path], self.bufsize))


# FFplayOutput


class FFplayOutputConfig(IOutputConfig):
    video_template: str = "-c:v copy"
    audio_template: str = "-c:a copy"


FFPLAY = "ffplay"


@register_output(FFplayOutputConfig)
class FFplayOutput(PipeOutput):
    def __init__(self, corr_cfg: "Config", cfg: FFplayOutputConfig):
        super().__init__(corr_cfg, cfg)

        ffmpeg = _FFmpegProcess([FFMPEG, *FFMPEG_QUIET], corr_cfg)
        ffmpeg.add_output(cfg)
        ffmpeg.templates.append("-f nut")

        p1 = ffmpeg.popen(["-"], self.bufsize, stdout=subprocess.PIPE)

        ffplay = shlex.split("ffplay -autoexit -") + FFMPEG_QUIET
        try:
            p2 = subprocess.Popen(ffplay, stdin=p1.stdout)
        except FileNotFoundError as e:
            raise MissingFFmpegError() from e

        p1.stdout.close()
        # assert p2.stdin is None   # True unless Popen is being mocked (test_output).

        self.open(p1, p2)
