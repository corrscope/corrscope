# -*- coding: utf-8 -*-
import sys
import time
from multiprocessing import Process, Pipe
from contextlib import ExitStack, contextmanager
from enum import unique, IntEnum
from types import SimpleNamespace
from typing import Optional, List, Union, TYPE_CHECKING, Iterator, Tuple

from ovgenpy import outputs as outputs_
from ovgenpy.channel import Channel, ChannelConfig
from ovgenpy.config import register_config, register_enum, Ignored
from ovgenpy.renderer import MatplotlibRenderer, RendererConfig
from ovgenpy.layout import LayoutConfig
from ovgenpy.triggers import ITriggerConfig, CorrelationTriggerConfig, PerFrameCache
from ovgenpy.util import pushd, coalesce
from ovgenpy.utils import keyword_dataclasses as dc
from ovgenpy.utils.keyword_dataclasses import field, InitVar

if TYPE_CHECKING:
    from ovgenpy.wave import Wave
    from ovgenpy.triggers import CorrelationTrigger
    from multiprocessing.connection import Connection
    import numpy as np


PRINT_TIMESTAMP = True

@register_enum
@unique
class BenchmarkMode(IntEnum):
    NONE = 0
    TRIGGER = 1
    SEND_TO_WORKER = 2
    RENDER = 3
    OUTPUT = 4


@register_config(always_dump='begin_time end_time subsampling')
class Config:
    master_audio: Optional[str]
    fps: int
    begin_time: float = 0
    end_time: float = None

    width_ms: int

    # trigger_subsampling and render_subsampling override subsampling.
    trigger_subsampling: int = None
    render_subsampling: int = None
    subsampling: InitVar[int] = 1

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
    def width_s(self) -> float:
        return self.width_ms / 1000

    def __post_init__(self, subsampling):
        # Cast benchmark_mode to enum.
        try:
            if not isinstance(self.benchmark_mode, BenchmarkMode):
                self.benchmark_mode = BenchmarkMode[self.benchmark_mode]
        except KeyError:
            raise ValueError(
                f'invalid benchmark_mode mode {self.benchmark_mode} not in '
                f'{[el.name for el in BenchmarkMode]}')

        # Compute trigger_subsampling and render_subsampling.
        self.trigger_subsampling = coalesce(self.trigger_subsampling, subsampling)
        self.render_subsampling = coalesce(self.render_subsampling, subsampling)


_FPS = 60  # f_s

def default_config(**kwargs):
    cfg = Config(
        master_audio='',
        fps=_FPS,
        amplification=1,

        width_ms=40,
        trigger_subsampling=1,
        render_subsampling=2,
        trigger=CorrelationTriggerConfig(
            edge_strength=2,
            responsiveness=0.5,
            use_edge_trigger=False,
            # Removed due to speed hit.
            # post=LocalPostTriggerConfig(strength=0.1),
        ),
        channels=[],

        layout=LayoutConfig(ncols=2),
        render=RendererConfig(1280, 800),
    )
    return dc.replace(cfg, **kwargs)


class Ovgen:
    def __init__(self, cfg: Config, cfg_dir: str,
                 outputs: List[outputs_.IOutputConfig]):
        self.cfg = cfg
        self.cfg_dir = cfg_dir
        self.has_played = False

        # TODO benchmark_mode/not_benchmarking == code duplication.
        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
            self.output_cfgs = outputs
        else:
            self.output_cfgs = []

        if len(self.cfg.channels) == 0:
            raise ValueError('Config.channels is empty')

    waves: List['Wave']
    channels: List[Channel]
    outputs: List[outputs_.Output]
    nchan: int

    def _load_channels(self):
        with pushd(self.cfg_dir):
            self.channels = [Channel(ccfg, self.cfg) for ccfg in self.cfg.channels]
            self.waves = [channel.wave for channel in self.channels]
            self.triggers = [channel.trigger for channel in self.channels]
            self.nchan = len(self.channels)

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
            extra_outputs.window = RenderOutput()

        extra_outputs.buffer = None
        if 'buffer' in internals:
            extra_outputs.buffer = RenderOutput()
        # endregion

        if PRINT_TIMESTAMP:
            begin = time.perf_counter()

        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        parent, child = Pipe(duplex=True)
        parent_send = connection_host(parent)

        render_thread = Process(
            name='render_thread',
            target=self.render_worker,
            args=(child, benchmark_mode, not_benchmarking)
        )
        del child
        render_thread.start()

        prev = -1
        try:
            # For each frame, render each wave
            for frame in range(begin_frame, end_frame):
                time_seconds = frame / fps

                rounded = int(time_seconds)
                if PRINT_TIMESTAMP and rounded != prev:
                    print(rounded)
                    prev = rounded

                render_datas = []
                # Get data from each wave
                for wave, channel in zip(self.waves, self.channels):
                    sample = round(wave.smp_s * time_seconds)

                    if not_benchmarking or \
                            BenchmarkMode.TRIGGER <= benchmark_mode <= BenchmarkMode.SEND_TO_WORKER:
                        cache = PerFrameCache()
                        trigger_sample = channel.trigger.get_trigger(sample, cache)
                    else:
                        trigger_sample = sample

                    render_datas.append(wave.get_around(
                        trigger_sample, channel.render_samp, channel.render_stride))

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

                if not_benchmarking or benchmark_mode >= BenchmarkMode.SEND_TO_WORKER:
                    # Type should match QueueMessage.
                    parent_send(render_datas)
                    # Processed by self.render_worker().

        # Terminate render thread.
        finally:
            parent_send(None)

        if self.raise_on_teardown:
            raise self.raise_on_teardown

        if PRINT_TIMESTAMP:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = (end_frame - begin_frame) / dtime
            print(f'FPS = {render_fps}')

        # Print FPS before waiting for render thread to terminate.
        render_thread.join()

    raise_on_teardown: Exception = None

    #### Render/output thread
    def render_worker(self, conn: 'Connection',
                      benchmark_mode, not_benchmarking):
        """ Communicates with main process via `pipe`.
        Accepts QueueMessage, sends Optional[exception info].

        Accepts N QueueMessage followed by None.
        Replies with N Optional[exception info].

        The parent checks for replies N times (before sending N-1 QueueMessage + None)
        """
        import traceback
        # assert threading.current_thread() != threading.main_thread()

        renderer = self._load_renderer()
        with self._load_outputs():
            # foreach frame
            for datas in iter_conn(conn):  # type: Message
                try:
                    # Render frame
                    if not_benchmarking or benchmark_mode >= BenchmarkMode.RENDER:
                        renderer.render_frame(datas)
                        frame = renderer.get_frame()

                        if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
                            # Output frame
                            for output in self.outputs:
                                output.write_frame(frame)
                except BaseException as e:
                    conn.send(True)
                    raise
                else:
                    conn.send(False)

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
        renderer = MatplotlibRenderer(self.cfg.render, self.cfg.layout, self.nchan,
                                      self.cfg.channels)
        return renderer


# message[chan] = trigger_sample (created by trigger)
Message = List['np.ndarray']
ReplyMessage = bool  # Has exception occurred?


# Parent
def connection_host(conn: 'Connection'):
    """ Checks for exceptions, then sends a message to the child process. """
    not_first = False

    def send(obj) -> None:
        nonlocal not_first
        if not_first:
            is_child_exc = conn.recv()  # type: ReplyMessage
            if is_child_exc:
                exit(1)

        conn.send(obj)
        not_first = True

    return send


# Child
def iter_conn(conn: 'Connection') -> Iterator[Message]:
    """ Yields elements of a threading queue, stops on None. """
    while True:
        item = conn.recv()
        if item is None:
            break
        yield item
