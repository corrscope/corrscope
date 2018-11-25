# https://ffmpeg.org/ffplay.html
import shlex
import subprocess
from abc import ABC, abstractmethod
from os.path import abspath
from typing import TYPE_CHECKING, Type, List, Union, Optional

from ovgenpy.config import register_config

if TYPE_CHECKING:
    from ovgenpy.ovgenpy import Config


RGB_DEPTH = 3
PIXEL_FORMAT = 'rgb24'

FRAMES_TO_BUFFER = 2


class IOutputConfig:
    cls: 'Type[Output]'

    def __call__(self, ovgen_cfg: 'Config'):
        return self.cls(ovgen_cfg, cfg=self)


class Output(ABC):
    def __init__(self, ovgen_cfg: 'Config', cfg: IOutputConfig):
        self.ovgen_cfg = ovgen_cfg
        self.cfg = cfg

        rcfg = ovgen_cfg.render

        frame_bytes = rcfg.height * rcfg.width * RGB_DEPTH
        self.bufsize = frame_bytes * FRAMES_TO_BUFFER

    def __enter__(self):
        return self

    @abstractmethod
    def write_frame(self, frame: bytes) -> None:
        """ Output a Numpy ndarray. """

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Glue logic

def register_output(config_t: Type[IOutputConfig]):
    def inner(output_t: Type[Output]):
        config_t.cls = output_t
        return output_t

    return inner


# FFmpeg command line generation

class _FFmpegProcess:
    def __init__(self, templates: List[str], ovgen_cfg: 'Config'):
        self.templates = templates
        self.ovgen_cfg = ovgen_cfg

        self.templates += ffmpeg_input_video(ovgen_cfg)  # video
        if ovgen_cfg.master_audio:
            # Load master audio and trim to timestamps.

            self.templates.append(f'-ss {ovgen_cfg.begin_time}')

            audio_path = shlex.quote(abspath(ovgen_cfg.master_audio))
            self.templates += ffmpeg_input_audio(audio_path)    # audio

            if ovgen_cfg.end_time is not None:
                dur = ovgen_cfg.end_time - ovgen_cfg.begin_time
                self.templates.append(f'-to {dur}')

    def add_output(self, cfg: 'Union[FFmpegOutputConfig, FFplayOutputConfig]') -> None:
        self.templates.append(cfg.video_template)  # video
        if self.ovgen_cfg.master_audio:
            self.templates.append(cfg.audio_template)  # audio

    def popen(self, extra_args, bufsize, **kwargs) -> subprocess.Popen:
        return subprocess.Popen(self._generate_args() + extra_args,
                                stdin=subprocess.PIPE, bufsize=bufsize, **kwargs)

    def _generate_args(self) -> List[str]:
        return [arg
                for template in self.templates
                for arg in shlex.split(template)]


def ffmpeg_input_video(cfg: 'Config') -> List[str]:
    fps = cfg.render_fps
    width = cfg.render.width
    height = cfg.render.height

    return [f'-f rawvideo -pixel_format {PIXEL_FORMAT} -video_size {width}x{height}',
            f'-framerate {fps}',
            '-i -']


def ffmpeg_input_audio(audio_path: str) -> List[str]:
    return ['-i', audio_path]


class PipeOutput(Output):
    def open(self, *pipeline: subprocess.Popen):
        """ Called by __init__ with a Popen pipeline to ffmpeg/ffplay. """
        if len(pipeline) == 0:
            raise TypeError('must provide at least one Popen argument to popens')

        self._pipeline = pipeline
        self._stream = pipeline[0].stdin
        # Python documentation discourages accessing popen.stdin. It's wrong.
        # https://stackoverflow.com/a/9886747

    def __enter__(self):
        return self

    def write_frame(self, frame: bytes) -> None:
        self._stream.write(frame)

    def close(self) -> int:
        self._stream.close()

        retval = 0
        for popen in self._pipeline:
            retval |= popen.wait()
        return retval   # final value

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            # Calling self.close() is bad.
            # If exception occurred but ffplay continues running.
            # popen.wait() will prevent stack trace from showing up.
            self._stream.close()

            exc = None
            for popen in self._pipeline:
                popen.terminate()
                # https://stackoverflow.com/a/49038779/2683842
                try:
                    popen.wait(1)   # timeout=seconds
                except subprocess.TimeoutExpired as e:
                    # gee thanks Python, https://stackoverflow.com/questions/45292479/
                    exc = e
                    popen.kill()

            if exc:
                raise exc


# FFmpegOutput

@register_config
class FFmpegOutputConfig(IOutputConfig):
    # path=None writes to stdout.
    path: Optional[str]
    args: str = ''

    video_template: str = '-c:v libx264 -crf 18 -preset superfast -movflags faststart'
    audio_template: str = '-c:a aac -b:a 384k'


FFMPEG = 'ffmpeg'

@register_output(FFmpegOutputConfig)
class FFmpegOutput(PipeOutput):
    def __init__(self, ovgen_cfg: 'Config', cfg: FFmpegOutputConfig):
        super().__init__(ovgen_cfg, cfg)

        ffmpeg = _FFmpegProcess([FFMPEG, '-y'], ovgen_cfg)
        ffmpeg.add_output(cfg)
        ffmpeg.templates.append(cfg.args)

        if cfg.path is None:
            video_path = '-'    # Write to stdout
        else:
            video_path = abspath(cfg.path)

        self.open(ffmpeg.popen([video_path], self.bufsize))


# FFplayOutput

@register_config
class FFplayOutputConfig(IOutputConfig):
    video_template: str = '-c:v copy'
    audio_template: str = '-c:a copy'


FFPLAY = 'ffplay'

@register_output(FFplayOutputConfig)
class FFplayOutput(PipeOutput):
    def __init__(self, ovgen_cfg: 'Config', cfg: FFplayOutputConfig):
        super().__init__(ovgen_cfg, cfg)

        ffmpeg = _FFmpegProcess([FFMPEG], ovgen_cfg)
        ffmpeg.add_output(cfg)
        ffmpeg.templates.append('-f nut')

        p1 = ffmpeg.popen(['-'], self.bufsize, stdout=subprocess.PIPE)

        ffplay = shlex.split('ffplay -autoexit -')
        p2 = subprocess.Popen(ffplay, stdin=p1.stdout)

        p1.stdout.close()
        # assert p2.stdin is None   # True unless Popen is being mocked (test_output).

        self.open(p1, p2)


# ImageOutput

@register_config
class ImageOutputConfig(IOutputConfig):
    path_prefix: str


@register_output(ImageOutputConfig)
class ImageOutput(Output):
    pass
