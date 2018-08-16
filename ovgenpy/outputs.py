# https://ffmpeg.org/ffplay.html
import shlex
import subprocess
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, List, Union

from ovgenpy.config import register_config

if TYPE_CHECKING:
    import numpy as np
    from ovgenpy.ovgenpy import Config


RGB_DEPTH = 3


class IOutputConfig:
    cls: 'Type[Output]'

    def __call__(self, ovgen_cfg: 'Config'):
        return self.cls(ovgen_cfg, cfg=self)


class Output(ABC):
    def __init__(self, ovgen_cfg: 'Config', cfg: IOutputConfig):
        self.ovgen_cfg = ovgen_cfg
        self.cfg = cfg

    @abstractmethod
    def write_frame(self, frame: bytes) -> None:
        """ Output a Numpy ndarray. """


# Glue logic

def register_output(config_t: Type[IOutputConfig]):
    def inner(output_t: Type[Output]):
        config_t.cls = output_t
        return output_t

    return inner


# FFmpeg input format

class _FFmpegCommand:
    def __init__(self, templates: List[str], ovgen_cfg: 'Config'):
        self.templates = templates
        self.ovgen_cfg = ovgen_cfg

        self.templates += ffmpeg_input_video(ovgen_cfg)  # video
        if ovgen_cfg.master_audio:
            audio_path = shlex.quote(ovgen_cfg.master_audio)
            self.templates += ffmpeg_input_audio(audio_path)    # audio

    def add_output(self, cfg: 'Union[FFmpegOutputConfig, FFplayOutputConfig]') -> None:
        self.templates.append(cfg.video_template)  # video
        if self.ovgen_cfg.master_audio:
            self.templates.append(cfg.audio_template)  # audio

    def popen(self, process_args=None, **kwargs) -> subprocess.Popen:
        if process_args is None:
            process_args = []

        return subprocess.Popen(self._generate_args() + process_args,
                                stdin=subprocess.PIPE, **kwargs)

    def _generate_args(self) -> List[str]:
        return [arg
                for template in self.templates
                for arg in shlex.split(template)]


assert RGB_DEPTH == 3
def ffmpeg_input_video(cfg: 'Config') -> List[str]:
    fps = cfg.fps
    width = cfg.render.width
    height = cfg.render.height

    return [f'-f rawvideo -pixel_format rgb24 -video_size {width}x{height}',
            f'-framerate {fps}',
            '-i -']


def ffmpeg_input_audio(audio_path: str) -> List[str]:
    return ['-i', audio_path]


class ProcessOutput(Output):
    def open(self, popen: subprocess.Popen):
        self._popen = popen
        self._stream = self._popen.stdin
        # Python documentation discourages accessing popen.stdin. It's wrong.
        # https://stackoverflow.com/a/9886747

    def write_frame(self, frame: bytes) -> None:
        # frame.tobytes() avoids PyCharm complaining about type mismatch,
        # but results in slightly higher CPU consumption.
        self._stream.write(frame)

    def close(self):
        self._stream.close()
        self._popen.wait()


# FFmpegOutput
@register_config
class FFmpegOutputConfig(IOutputConfig):
    path: str

    # Do not use `-movflags faststart`, I get corrupted mp4 files (missing MOOV)
    video_template: str = '-c:v libx264 -crf 18 -preset superfast'
    audio_template: str = '-c:a aac -b:a 384k'


FFMPEG = 'ffmpeg'

@register_output(FFmpegOutputConfig)
class FFmpegOutput(ProcessOutput):
    def __init__(self, ovgen_cfg: 'Config', cfg: FFmpegOutputConfig):
        super().__init__(ovgen_cfg, cfg)

        ffmpeg = _FFmpegCommand([FFMPEG, '-y'], ovgen_cfg)
        ffmpeg.add_output(cfg)
        self.open(ffmpeg.popen([cfg.path]))


# FFplayOutput
@register_config
class FFplayOutputConfig(IOutputConfig):
    video_template: str = '-c:v copy'
    audio_template: str = '-c:a copy'


FFPLAY = 'ffplay'

@register_output(FFplayOutputConfig)
class FFplayOutput(ProcessOutput):
    def __init__(self, ovgen_cfg: 'Config', cfg: FFplayOutputConfig):
        super().__init__(ovgen_cfg, cfg)

        ffmpeg = _FFmpegCommand([FFMPEG], ovgen_cfg)
        ffmpeg.add_output(cfg)
        ffmpeg.templates.append('-f nut -')

        p1 = ffmpeg.popen(stdout=subprocess.PIPE)

        ffplay = shlex.split('ffplay -autoexit -')
        self.p2 = subprocess.Popen(ffplay, stdin=p1.stdout)

        p1.stdout.close()
        self.open(p1)

    def close(self):
        ProcessOutput.close(self)
        self.p2.wait()


# ImageOutput
@register_config
class ImageOutputConfig(IOutputConfig):
    path_prefix: str


@register_output(ImageOutputConfig)
class ImageOutput(Output):
    pass
