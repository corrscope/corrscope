# -*- coding: utf-8 -*-
import time
from contextlib import ExitStack, contextmanager
from enum import unique, IntEnum
from types import SimpleNamespace
from typing import Optional, List, Union, TYPE_CHECKING

from ovgenpy import outputs
from ovgenpy.channel import Channel, ChannelConfig
from ovgenpy.config import register_config, register_enum, Ignored
from ovgenpy.renderer import MatplotlibRenderer, RendererConfig, LayoutConfig
from ovgenpy.triggers import ITriggerConfig, CorrelationTriggerConfig, Trigger
from ovgenpy.utils import keyword_dataclasses as dc
from ovgenpy.utils.keyword_dataclasses import field
from ovgenpy.wave import Wave

if TYPE_CHECKING:
    from ovgenpy.triggers import CorrelationTrigger

# cgitb.enable(format='text')

PRINT_TIMESTAMP = True

@register_enum
@unique
class BenchmarkMode(IntEnum):
    NONE = 0
    TRIGGER = 1
    RENDER = 2
    OUTPUT = 3


@register_config(always_dump='begin_time')
class Config:
    master_audio: Optional[str]
    fps: int
    begin_time: float = 0

    subsampling: int

    width_ms: int
    channels: List[ChannelConfig] = field(default_factory=list)
    trigger: ITriggerConfig  # Can be overriden per Wave

    amplification: float
    layout: LayoutConfig
    render: RendererConfig

    outputs: List[outputs.IOutputConfig]

    show_internals: List[str] = field(default_factory=list)
    benchmark_mode: Union[str, BenchmarkMode] = BenchmarkMode.NONE

    # region Legacy Fields
    wav_prefix = Ignored
    # endregion

    @property
    def render_width_s(self) -> float:
        return self.width_ms / 1000

    def __post_init__(self):
        try:
            if not isinstance(self.benchmark_mode, BenchmarkMode):
                self.benchmark_mode = BenchmarkMode[self.benchmark_mode]
        except KeyError:
            raise ValueError(
                f'invalid benchmark_mode mode {self.benchmark_mode} not in '
                f'{[el.name for el in BenchmarkMode]}')


_FPS = 60  # f_s

def default_config(**kwargs):
    cfg = Config(
        master_audio='',
        fps=_FPS,
        # begin_time=0,

        channels=[],

        width_ms=25,
        subsampling=1,
        trigger=CorrelationTriggerConfig(),

        amplification=1,
        layout=LayoutConfig(ncols=1),
        render=RendererConfig(1280, 720),

        outputs=[
            # outputs.FFmpegOutputConfig(output),
            outputs.FFplayOutputConfig(),
        ]
    )
    return dc.replace(cfg, **kwargs)


class Ovgen:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.has_played = False

        if len(self.cfg.channels) == 0:
            raise ValueError('Config.channels is empty')

    waves: List[Wave]
    channels: List[Channel]
    outputs: List[outputs.Output]
    nchan: int

    def _load_channels(self):
        self.channels = [Channel(ccfg, self.cfg) for ccfg in self.cfg.channels]
        self.waves = [channel.wave for channel in self.channels]
        self.triggers = [channel.trigger for channel in self.channels]
        self.nchan = len(self.channels)

    @contextmanager
    def _load_outputs(self):
        with ExitStack() as stack:
            self.outputs = [
                stack.enter_context(output_cfg(self.cfg))
                for output_cfg in self.cfg.outputs
            ]
            yield

    def _load_renderer(self):
        renderer = MatplotlibRenderer(self.cfg.render, self.cfg.layout, self.nchan)
        renderer.set_colors(self.cfg.channels)
        return renderer

    def play(self):
        if self.has_played:
            raise ValueError('Cannot call Ovgen.play() more than once')
        self.has_played = True

        self._load_channels()
        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps

        begin_frame = round(fps * self.cfg.begin_time)

        end_frame = fps * self.waves[0].get_s()
        end_frame = int(end_frame) + 1

        renderer = self._load_renderer()

        # Display buffers, for debugging purposes.
        internals = self.cfg.show_internals
        extra_outputs = SimpleNamespace()
        if internals:
            from ovgenpy.outputs import FFplayOutputConfig
            from ovgenpy.utils.keyword_dataclasses import replace

            no_audio = replace(self.cfg, master_audio='')

            ovgen = self

            class RenderOutput:
                def __init__(self):
                    self.renderer = ovgen._load_renderer()
                    self.output = FFplayOutputConfig()(no_audio)

                def render_frame(self, datas):
                    self.renderer.render_frame(datas)
                    self.output.write_frame(self.renderer.get_frame())

        extra_outputs.window = None
        if 'window' in internals:
            for trigger in self.triggers:   # type: CorrelationTrigger
                trigger.save_window = True
            extra_outputs.window = RenderOutput()

        extra_outputs.buffer = None
        if 'buffer' in internals:
            extra_outputs.buffer = RenderOutput()

        if PRINT_TIMESTAMP:
            begin = time.perf_counter()

        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        # For each frame, render each wave
        with self._load_outputs():
            prev = -1
            for frame in range(begin_frame, end_frame):
                time_seconds = frame / fps

                rounded = int(time_seconds)
                if PRINT_TIMESTAMP and rounded != prev:
                    print(rounded)
                    prev = rounded

                datas = []
                # Get data from each wave
                for wave, channel in zip(self.waves, self.channels):
                    sample = round(wave.smp_s * time_seconds)

                    if not_benchmarking or benchmark_mode == BenchmarkMode.TRIGGER:
                        trigger_sample = channel.trigger.get_trigger(sample)
                    else:
                        trigger_sample = sample

                    datas.append(wave.get_around(
                        trigger_sample, channel.window_samp, channel.render_subsampling))

                # Display buffers, for debugging purposes.

                if extra_outputs.window:
                    triggers: List['CorrelationTrigger'] = self.triggers
                    extra_outputs.window.render_frame(
                        [trigger._prev_window for trigger in triggers])

                if extra_outputs.buffer:
                    triggers: List['CorrelationTrigger'] = self.triggers
                    extra_outputs.buffer.render_frame(
                        [trigger._buffer for trigger in triggers])

                if not_benchmarking or benchmark_mode >= BenchmarkMode.RENDER:
                    # Render frame
                    renderer.render_frame(datas)

                    if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
                        # Output frame
                        frame = renderer.get_frame()
                        for output in self.outputs:
                            output.write_frame(frame)

        if PRINT_TIMESTAMP:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = (end_frame - begin_frame) / dtime
            print(f'FPS = {render_fps}')
