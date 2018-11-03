# -*- coding: utf-8 -*-
import time
from contextlib import ExitStack, contextmanager
from enum import unique, IntEnum
from types import SimpleNamespace
from typing import Optional, List, Union, TYPE_CHECKING

from ovgenpy import outputs as outputs_
from ovgenpy.channel import Channel, ChannelConfig
from ovgenpy.config import register_config, register_enum, Ignored
from ovgenpy.renderer import MatplotlibRenderer, RendererConfig, LayoutConfig
from ovgenpy.triggers import ITriggerConfig, CorrelationTriggerConfig, PerFrameCache
from ovgenpy.util import pushd, coalesce
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


@register_config(always_dump='begin_time end_time subsampling')
class Config:
    master_audio: Optional[str]
    fps: int
    begin_time: float = 0
    end_time: float = None

    width_ms: int
    subsampling: int = 1
    trigger_width: int = 1
    render_width: int = 1

    amplification: float

    trigger: ITriggerConfig  # Can be overriden per Wave

    # Can override trigger_width, render_width, trigger
    channels: List[ChannelConfig] = field(default_factory=list)

    layout: LayoutConfig
    render: RendererConfig

    player: outputs_.IOutputConfig = outputs_.FFplayOutputConfig()
    encoder: outputs_.IOutputConfig = outputs_.FFmpegOutputConfig(None)

    show_internals: List[str] = field(default_factory=list)
    benchmark_mode: Union[str, BenchmarkMode] = BenchmarkMode.NONE

    # region Legacy Fields
    outputs = Ignored
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
        amplification=1,

        width_ms=40,
        subsampling=2,
        trigger=CorrelationTriggerConfig(
            edge_strength=2,
            responsiveness=0.5,
            use_edge_trigger=False,
            # Removed due to speed hit.
            # post=LocalPostTriggerConfig(strength=0.1),
        ),
        channels=[],

        layout=LayoutConfig(ncols=2),
        render=RendererConfig(800, 480),
    )
    return dc.replace(cfg, **kwargs)


class Ovgen:
    def __init__(self, cfg: Config, cfg_dir: str,
                 outputs: List[outputs_.IOutputConfig]):
        self.cfg = cfg
        self.cfg_dir = cfg_dir
        self.has_played = False
        self.output_cfgs = outputs

        if len(self.cfg.channels) == 0:
            raise ValueError('Config.channels is empty')

    waves: List[Wave]
    channels: List[Channel]
    outputs: List[outputs_.Output]
    nchan: int

    def _load_channels(self):
        with pushd(self.cfg_dir):
            self.channels = [Channel(ccfg, self.cfg) for ccfg in self.cfg.channels]
            self.waves = [channel.wave for channel in self.channels]
            self.triggers = [channel.trigger for channel in self.channels]
            self.nchan = len(self.channels)

    @contextmanager
    def _load_outputs(self):
        with pushd(self.cfg_dir):
            with ExitStack() as stack:
                self.outputs = [
                    stack.enter_context(output_cfg(self.cfg))
                    for output_cfg in self.output_cfgs
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

        end_frame = fps * coalesce(self.cfg.end_time, self.waves[0].get_s())
        end_frame = int(end_frame) + 1

        renderer = self._load_renderer()

        # region show_internals
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
        # endregion

        if PRINT_TIMESTAMP:
            begin = time.perf_counter()

        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        with self._load_outputs():
            prev = -1
            # For each frame, render each wave
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
                        cache = PerFrameCache()
                        trigger_sample = channel.trigger.get_trigger(sample, cache)
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

            if self.raise_on_teardown:
                raise self.raise_on_teardown

        if PRINT_TIMESTAMP:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = (end_frame - begin_frame) / dtime
            print(f'FPS = {render_fps}')

    raise_on_teardown: Exception = None
