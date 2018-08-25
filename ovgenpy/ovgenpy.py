# -*- coding: utf-8 -*-
import time
from contextlib import ExitStack, contextmanager
from enum import unique, IntEnum
from typing import Optional, List, Union

from ovgenpy import outputs
from ovgenpy.channel import Channel, ChannelConfig
from ovgenpy.config import register_config, register_enum
from ovgenpy.renderer import MatplotlibRenderer, RendererConfig, LayoutConfig
from ovgenpy.triggers import ITriggerConfig, CorrelationTriggerConfig
from ovgenpy.utils import keyword_dataclasses as dc
from ovgenpy.utils.keyword_dataclasses import field
from ovgenpy.wave import Wave

# cgitb.enable(format='text')

RENDER_PROFILING = True

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

    channels: List[ChannelConfig] = field(default_factory=list)

    width_ms: int
    subsampling: int
    trigger: ITriggerConfig  # Maybe overriden per Wave

    amplification: float
    layout: LayoutConfig
    render: RendererConfig

    outputs: List[outputs.IOutputConfig]

    benchmark_mode: Union[str, BenchmarkMode] = BenchmarkMode.NONE

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
        trigger=CorrelationTriggerConfig(
            trigger_strength=10,
            use_edge_trigger=True,

            responsiveness=0.1,
            falloff_width=0.5,
        ),

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
        render_width_s = self.cfg.render_width_s
        fps = self.cfg.fps

        begin_frame = round(fps * self.cfg.begin_time)

        end_frame = fps * self.waves[0].get_s()
        end_frame = int(end_frame) + 1

        renderer = self._load_renderer()

        if RENDER_PROFILING:
            begin = time.perf_counter()

        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        # For each frame, render each wave
        with self._load_outputs():
            prev = -1
            for frame in range(begin_frame, end_frame):
                time_seconds = frame / fps

                rounded = int(time_seconds)
                if RENDER_PROFILING and rounded != prev:
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
                        trigger_sample, channel.nsamp, channel.render_subsampling))

                if not_benchmarking or benchmark_mode >= BenchmarkMode.RENDER:
                    # Render frame
                    renderer.render_frame(datas)

                    if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
                        # Output frame
                        frame = renderer.get_frame()
                        for output in self.outputs:
                            output.write_frame(frame)

        if RENDER_PROFILING:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = (end_frame - begin_frame) / dtime
            print(f'FPS = {render_fps}')
