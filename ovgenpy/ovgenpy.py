# -*- coding: utf-8 -*-
import time
import warnings
from contextlib import ExitStack, contextmanager
from enum import unique, IntEnum
from fractions import Fraction
from types import SimpleNamespace
from typing import Optional, List, Union, TYPE_CHECKING, Callable

import attr

from ovgenpy import outputs as outputs_
from ovgenpy.channel import Channel, ChannelConfig
from ovgenpy.config import kw_config, register_enum, Ignored, OvgenError, OvgenWarning
from ovgenpy.renderer import MatplotlibRenderer, RendererConfig
from ovgenpy.layout import LayoutConfig
from ovgenpy.triggers import ITriggerConfig, CorrelationTriggerConfig, PerFrameCache
from ovgenpy.util import pushd, coalesce
from ovgenpy.wave import Wave

if TYPE_CHECKING:
    from ovgenpy.triggers import CorrelationTrigger


PRINT_TIMESTAMP = True

@register_enum
@unique
class BenchmarkMode(IntEnum):
    NONE = 0
    TRIGGER = 1
    RENDER = 2
    OUTPUT = 3


@kw_config(always_dump='render_subfps begin_time end_time subsampling')
class Config:
    """ Default values indicate optional attributes. """
    master_audio: Optional[str]
    begin_time: float = 0
    end_time: Optional[float] = None

    fps: int

    trigger_ms: Optional[int] = None
    render_ms: Optional[int] = None
    _width_ms: Optional[int] = None

    # trigger_subsampling and render_subsampling override subsampling.
    # Always non-None after __attrs_post_init__()
    trigger_subsampling: int = None
    render_subsampling: int = None
    _subsampling: int = 1

    render_subfps: int = 1
    # FFmpeg accepts FPS as a fraction only.
    render_fps = property(lambda self:
                          Fraction(self.fps, self.render_subfps))

    # TODO: Remove cfg._width (breaks compat)
    # ISSUE: baking into trigger_ms will stack with channel-specific ms
    trigger_width: int = 1
    render_width: int = 1

    amplification: float

    trigger: ITriggerConfig  # Can be overriden per Wave

    # Can override trigger_width, render_width, trigger
    channels: List[ChannelConfig]

    layout: LayoutConfig
    render: RendererConfig

    player: outputs_.IOutputConfig = outputs_.FFplayOutputConfig()
    encoder: outputs_.IOutputConfig = outputs_.FFmpegOutputConfig(None)

    show_internals: List[str] = attr.Factory(list)
    benchmark_mode: Union[str, BenchmarkMode] = BenchmarkMode.NONE

    # region Legacy Fields
    outputs = Ignored
    wav_prefix = Ignored
    # endregion

    def __attrs_post_init__(self):
        # Cast benchmark_mode to enum.
        try:
            if not isinstance(self.benchmark_mode, BenchmarkMode):
                self.benchmark_mode = BenchmarkMode[self.benchmark_mode]
        except KeyError:
            raise OvgenError(
                f'invalid benchmark_mode mode {self.benchmark_mode} not in '
                f'{[el.name for el in BenchmarkMode]}')

        # Compute trigger_subsampling and render_subsampling.
        subsampling = self._subsampling
        self.trigger_subsampling = coalesce(self.trigger_subsampling, subsampling)
        self.render_subsampling = coalesce(self.render_subsampling, subsampling)

        # Compute trigger_ms and render_ms.
        width_ms = self._width_ms
        try:
            self.trigger_ms = coalesce(self.trigger_ms, width_ms)
            self.render_ms = coalesce(self.render_ms, width_ms)
        except TypeError:
            raise OvgenError(
                'Must supply either width_ms or both (trigger_ms and render_ms)')

        deprecated = []
        if self.trigger_width != 1:
            deprecated.append('trigger_width')
        if self.render_width != 1:
            deprecated.append('render_width')
        if deprecated:
            warnings.warn(f"Options {deprecated} are deprecated and will be removed",
                          OvgenWarning)


_FPS = 60  # f_s

def default_config(**kwargs) -> Config:
    """ Default template values do NOT indicate optional attributes. """
    cfg = Config(
        render_subfps=1,
        master_audio='',
        fps=_FPS,
        amplification=1,

        trigger_ms=40,
        render_ms=40,
        trigger_subsampling=1,
        render_subsampling=2,
        trigger=CorrelationTriggerConfig(
            edge_strength=2,
            responsiveness=0.5,
            buffer_falloff=0.5,
            use_edge_trigger=False,
            # Removed due to speed hit.
            # post=LocalPostTriggerConfig(strength=0.1),
        ),
        channels=[],

        layout=LayoutConfig(ncols=2),
        render=RendererConfig(1280, 720),
    )
    return attr.evolve(cfg, **kwargs)


BeginFunc = Callable[[float, float], None]
ProgressFunc = Callable[[int], None]
IsAborted = Callable[[], bool]

@attr.dataclass
class Arguments:
    cfg_dir: str
    outputs: List[outputs_.IOutputConfig]

    on_begin: BeginFunc = lambda begin_time, end_time: None
    progress: ProgressFunc = print
    is_aborted: IsAborted = lambda: False
    on_end: Callable[[], None] = lambda: None

class Ovgen:
    def __init__(self, cfg: Config, arg: Arguments):
        self.cfg = cfg
        self.arg = arg
        self.has_played = False

        # TODO test progress and is_aborted
        # TODO benchmark_mode/not_benchmarking == code duplication.
        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
            self.output_cfgs = arg.outputs
        else:
            self.output_cfgs = []

        if len(self.cfg.channels) == 0:
            raise OvgenError('Config.channels is empty')

    waves: List[Wave]
    channels: List[Channel]
    outputs: List[outputs_.Output]
    nchan: int

    def _load_channels(self):
        with pushd(self.arg.cfg_dir):
            self.channels = [Channel(ccfg, self.cfg) for ccfg in self.cfg.channels]
            self.waves = [channel.wave for channel in self.channels]
            self.triggers = [channel.trigger for channel in self.channels]
            self.nchan = len(self.channels)

    @contextmanager
    def _load_outputs(self):
        with pushd(self.arg.cfg_dir):
            with ExitStack() as stack:
                self.outputs = [
                    stack.enter_context(output_cfg(self.cfg))
                    for output_cfg in self.output_cfgs
                ]
                yield

    def _load_renderer(self):
        renderer = MatplotlibRenderer(self.cfg.render, self.cfg.layout, self.nchan,
                                      self.cfg.channels)
        return renderer

    def play(self):
        if self.has_played:
            raise ValueError('Cannot call Ovgen.play() more than once')
        self.has_played = True

        self._load_channels()
        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps

        begin_frame = round(fps * self.cfg.begin_time)

        end_time = coalesce(self.cfg.end_time, self.waves[0].get_s())
        end_frame = fps * end_time
        end_frame = int(end_frame) + 1

        self.arg.on_begin(self.cfg.begin_time, end_time)

        renderer = self._load_renderer()

        # region show_internals
        # Display buffers, for debugging purposes.
        internals = self.cfg.show_internals
        extra_outputs = SimpleNamespace()
        if internals:
            from ovgenpy.outputs import FFplayOutputConfig
            import attr

            no_audio = attr.evolve(self.cfg, master_audio='')

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

            # When subsampling FPS, render frames from the future to alleviate lag.
            # subfps=1, ahead=0.
            # subfps=2, ahead=1.
            render_subfps = self.cfg.render_subfps
            ahead = render_subfps // 2

            # For each frame, render each wave
            for frame in range(begin_frame, end_frame):
                if self.arg.is_aborted():
                    # Used for FPS calculation
                    end_frame = frame

                    for output in self.outputs:
                        output.terminate()
                    break

                time_seconds = frame / fps
                should_render = (frame - begin_frame) % render_subfps == ahead

                rounded = int(time_seconds)
                if PRINT_TIMESTAMP and rounded != prev:
                    self.arg.progress(rounded)
                    prev = rounded

                render_datas = []
                # Get data from each wave
                for wave, channel in zip(self.waves, self.channels):
                    sample = round(wave.smp_s * time_seconds)

                    if not_benchmarking or benchmark_mode == BenchmarkMode.TRIGGER:
                        cache = PerFrameCache()
                        trigger_sample = channel.trigger.get_trigger(sample, cache)
                    else:
                        trigger_sample = sample
                    if should_render:
                        render_datas.append(wave.get_around(
                            trigger_sample, channel.render_samp, channel.render_stride))

                if not should_render:
                    continue

                # region Display buffers, for debugging purposes.
                if extra_outputs.window:
                    triggers: List['CorrelationTrigger'] = self.triggers
                    extra_outputs.window.render_frame(
                        [trigger._prev_window for trigger in triggers])

                if extra_outputs.buffer:
                    triggers: List['CorrelationTrigger'] = self.triggers
                    extra_outputs.buffer.render_frame(
                        [trigger._buffer for trigger in triggers])
                # endregion

                if not_benchmarking or benchmark_mode >= BenchmarkMode.RENDER:
                    # Render frame
                    renderer.render_frame(render_datas)
                    frame_data = renderer.get_frame()

                    if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
                        # Output frame
                        aborted = False
                        for output in self.outputs:
                            if output.write_frame(frame_data) is outputs_.Stop:
                                aborted = True
                                break
                        if aborted:
                            # Outputting frame happens after most computation finished.
                            end_frame = frame + 1
                            break

            if self.raise_on_teardown:
                raise self.raise_on_teardown

        if PRINT_TIMESTAMP:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = (end_frame - begin_frame) / dtime
            print(f'FPS = {render_fps}')

    raise_on_teardown: Optional[Exception] = None
