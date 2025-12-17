import errno
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from os.path import abspath
from typing import TYPE_CHECKING, Type, List, Union, Optional, ClassVar, Callable

from corrscope.config import DumpableAttrs, CorrError
from corrscope.renderer import ByteBuffer, Renderer
from corrscope.settings.paths import MissingFFmpegError

if TYPE_CHECKING:
    from corrscope.corrscope import Config


FRAMES_TO_BUFFER = 2

FFMPEG_QUIET = "-nostats -hide_banner -loglevel error".split()


class IOutputConfig(DumpableAttrs):
    cls: "ClassVar[Type[Output]]"

    def __call__(self, corr_cfg: "Config", ffprobe_detect_mono: bool) -> "Output":
        """Must be called in the .yaml file's directory.
        This is used to properly resolve corr_cfg.master_audio."""
        return self.cls(corr_cfg, cfg=self, ffprobe_detect_mono=ffprobe_detect_mono)


class _Stop:
    pass


Stop = _Stop()


class Output(ABC):
    def __init__(
        self, corr_cfg: "Config", cfg: IOutputConfig, ffprobe_detect_mono: bool = True
    ):
        self.corr_cfg = corr_cfg
        self.cfg = cfg
        del ffprobe_detect_mono

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


def ffprobe_is_mono(path: str) -> bool:
    """Returns whether ffprobe thinks the input path is mono.
    If ffprobe is missing, raises MissingFFmpegError(ffprobe=True). (We currently don't have a way to spawn a nonfatal warning.)
    If ffprobe errors trying to read the file, raises CorrError."""

    # How does ffprobe behave?
    # - If passed an invalid file, it prints nothing and returns a nonzero error code.
    # - On a valid file, it prints a number and returns error code 0.
    #
    # On some files (eg. produced by yt-dlp?), ffprobe may return blank lines of output for non-audio streams.
    # - If you remove `-of compact=p=0:nk=1`, you will see multiple [STREAM]...[/STREAM] instead.
    # So strip whitespace before parsing the integer.
    r"""
    C:\Users\user>ffprobe -show_entries stream=channels -of compact=p=0:nk=1 -v 0 asdf
    C:\Users\user>echo %ERRORLEVEL%
    1
    C:\Users\user>ffprobe -show_entries stream=channels -of compact=p=0:nk=1 -v 0 D:\Music\eirinbae.opus
    2
    C:\Users\user>echo %ERRORLEVEL%
    0
    C:\Users\user>ffprobe -show_entries stream=channels -of compact=p=0:nk=1 -v 0 "C:\Users\user\Videos\[Vinesauce] Vinny - Kukkiizu Bassuru [o3-XlcMpxeU].webm"

    2
    """
    try:
        proc = subprocess.run(
            [
                *"ffprobe -show_entries stream=channels -of compact=p=0:nk=1 -v 0".split(),
                path,
            ],
            capture_output=True,
        )
    except FileNotFoundError as e:
        raise MissingFFmpegError(ffprobe=True) from e

    def error():
        return CorrError(f'Could not determine channel count for master audio "{path}"')

    if proc.returncode != 0:
        raise error()

    try:
        nchan = int(proc.stdout.strip())
    except ValueError:
        raise error()

    is_mono = nchan == 1
    return is_mono


class _FFmpegProcess:
    def __init__(
        self, templates: List[str], corr_cfg: "Config", ffprobe_detect_mono: bool
    ):
        self.templates = templates
        self.corr_cfg = corr_cfg

        # Test for ffmpeg's existence before calling ffprobe.
        if ffprobe_detect_mono and shutil.which("ffmpeg") is None:
            raise MissingFFmpegError

        self.templates += ffmpeg_input_video(corr_cfg)  # video
        if corr_cfg.master_audio:
            # Raise FileNotFoundError if missing.
            open(corr_cfg.master_audio, "rb").close()

            # Load master audio and trim to timestamps.

            self.templates.append(f"-ss {corr_cfg.begin_time}")

            if ffprobe_detect_mono:
                self.mono = ffprobe_is_mono(corr_cfg.master_audio)
            else:
                self.mono = False

            audio_path = shlex.quote(abspath(corr_cfg.master_audio))
            self.templates += ffmpeg_input_audio(audio_path, self.mono)  # audio

            if corr_cfg.end_time is not None:
                dur = corr_cfg.end_time - corr_cfg.begin_time
                self.templates.append(f"-to {dur}")

    def add_output(self, cfg: "Union[FFmpegOutputConfig, FFplayOutputConfig]") -> None:
        self.templates.append(cfg.video_template)  # video
        if self.corr_cfg.master_audio:
            audio_template = cfg.audio_template

            # When upmixing mono to stereo (in ffmpeg_input_audio()), we need to
            # reencode audio.
            if self.mono and audio_template.endswith(" copy"):
                audio_template = "-c:a pcm_s16le"
            self.templates.append(audio_template)  # audio

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


def ffmpeg_input_audio(audio_path: str, mono: bool) -> List[str]:
    out = ["-i", audio_path]

    # Preserve mono volume and upmix to stereo. If we don't do this explicitly, most
    # players will attenuate stereo playback by 3 dB during encoding/playback.
    if mono:
        out += ["-af", "pan=stereo|c0=c0|c1=c0"]
    return out


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
    audio_template: str = "-c:a libopus -b:a 256k"


FFMPEG = "ffmpeg"


@register_output(FFmpegOutputConfig)
class FFmpegOutput(PipeOutput):
    def __init__(
        self,
        corr_cfg: "Config",
        cfg: FFmpegOutputConfig,
        ffprobe_detect_mono: bool = True,
    ):
        super().__init__(corr_cfg, cfg)

        ffmpeg = _FFmpegProcess([FFMPEG, "-y"], corr_cfg, ffprobe_detect_mono)
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
    def __init__(
        self,
        corr_cfg: "Config",
        cfg: FFplayOutputConfig,
        ffprobe_detect_mono: bool = True,
    ):
        super().__init__(corr_cfg, cfg)

        # Test for ffplay's existence before calling ffprobe.
        if ffprobe_detect_mono and shutil.which("ffplay") is None:
            raise MissingFFmpegError

        ffmpeg = _FFmpegProcess([FFMPEG, *FFMPEG_QUIET], corr_cfg, ffprobe_detect_mono)
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
