# https://ffmpeg.org/ffplay.html
import shlex
import subprocess
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional, TYPE_CHECKING, Type, Callable, TypeVar, List

from dataclasses import dataclass

if TYPE_CHECKING:
    from ovgenpy.ovgenpy import Config


IMAGE_FORMAT = 'png'


class OutputConfig:
    cls: 'Type[Output]'

    def __call__(self, ovgen_cfg: 'Config'):
        return self.cls(ovgen_cfg, cfg=self)


class Output(ABC):
    def __init__(self, ovgen_cfg: 'Config', cfg: OutputConfig):
        self.ovgen_cfg = ovgen_cfg
        self.cfg = cfg

    @abstractmethod
    def output_frame_png(self, frame: bytes) -> None:
        """ Output an encoded PNG file. TODO PNG compression overhead is bad """
        pass


# Glue logic

def register_output(config_t: Type[OutputConfig]):
    def inner(output_t: Type[Output]):
        config_t.cls = output_t
        return output_t

    return inner


# Output subclasses

## FFMPEG templates TODO rename to "...template..."
FFMPEG = 'ffmpeg'
FFPLAY = 'ffplay'


def ffmpeg_input_video(fps: int) -> List[str]:
    # Removed: '-c:v {IMAGE_FORMAT}' since it doesn't work
    return ['-f image2pipe -framerate', str(fps), '-i -']


def ffmpeg_input_audio(audio_path: str) -> List[str]:
    return ['-i', audio_path]


FFMPEG_OUTPUT_VIDEO_DEFAULT = '-c:v libx264 -crf 18 -bf 2 -flags +cgop -pix_fmt yuv420p -movflags faststart'
FFMPEG_OUTPUT_AUDIO_DEFAULT = '-c:a aac -b:a 384k'


def parse_templates(templates: List[str]) -> List[str]:
    return [arg
            for template in templates
            for arg in shlex.split(template)]


# @dataclass
# class FFmpegCommand:
#     audio: Optional[str] = None
#
#     def generate_command(self):


@dataclass
class FFmpegOutputConfig(OutputConfig):
    path: str
    video_template: str = FFMPEG_OUTPUT_VIDEO_DEFAULT
    audio_template: str = FFMPEG_OUTPUT_AUDIO_DEFAULT


@register_output(FFmpegOutputConfig)
class FFmpegOutput(Output):
    # TODO https://github.com/kkroening/ffmpeg-python

    def __init__(self, ovgen_cfg: 'Config', cfg: FFmpegOutputConfig):
        super().__init__(ovgen_cfg, cfg)

        # Input
        templates: List[str] = [FFMPEG]

        # TODO factor out "get_ffmpeg_input"... what if wrong abstraction?
        templates += ffmpeg_input_video(fps=ovgen_cfg.fps)  # video
        if ovgen_cfg.audio_path:
            templates += ffmpeg_input_audio(audio_path=ovgen_cfg.audio_path)    # audio

        # Output
        templates.append(cfg.video_template)  # video
        if ovgen_cfg.audio_path:
            templates.append(cfg.audio_template)  # audio

        templates.append(cfg.path)  # output filename

        # Split arguments by words
        args = parse_templates(templates)

        self._popen = subprocess.Popen(args, stdin=subprocess.PIPE)
        self._stream = self._popen.stdin

        # Python documentation discourages accessing popen.stdin. It's wrong.
        # https://stackoverflow.com/a/9886747

    def output_frame_png(self, frame: bytes) -> None:
        self._stream.write(frame)

    def close(self):
        self._stream.close()
        self._popen.wait()
    # {ffmpeg}
    #
    #     # input
    #     -f image2pipe -framerate {framerate} -c:v {IMAGE_FORMAT} -i {img}
    #     -i {audio}
    #
    #     # output
    #     -c:a aac -b:a 384k
    #     -c:v libx264 -crf 18 -bf 2 -flags +cgop -pix_fmt yuv420p -movflags faststart
    #     {outfile}


class FFplayOutputConfig(OutputConfig):
    pass

@register_output(FFplayOutputConfig)
class FFplayOutput(Output):
    pass


@dataclass
class ImageOutputConfig:
    path_prefix: str


@register_output(ImageOutputConfig)
class ImageOutput(Output):
    pass
