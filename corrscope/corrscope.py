# -*- coding: utf-8 -*-
import time
from contextlib import ExitStack, contextmanager
from enum import unique, IntEnum
from fractions import Fraction
from multiprocessing import Process, Pipe
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Union, TYPE_CHECKING, Callable

import attr

from corrscope import outputs as outputs_
from corrscope.channel import Channel, ChannelConfig
from corrscope.config import kw_config, register_enum, CorrError
from corrscope.layout import LayoutConfig
from corrscope.parallelism import Message, connection_host, iter_conn
from corrscope.renderer import MatplotlibRenderer, RendererConfig
from corrscope.triggers import ITriggerConfig, CorrelationTriggerConfig, PerFrameCache
from corrscope.util import pushd, coalesce
from corrscope.wave import Wave

if TYPE_CHECKING:
    from corrscope.wave import Wave
    from corrscope.triggers import CorrelationTrigger
    from multiprocessing.connection import Connection

PRINT_TIMESTAMP = True


@register_enum
@unique
class BenchmarkMode(IntEnum):
    NONE = 0
    TRIGGER = 1
    SEND_TO_WORKER = 2
    RENDER = 3
    OUTPUT = 4


@kw_config(always_dump="render_subfps begin_time end_time subsampling")
class Config:
    """ Default values indicate optional attributes. """

    master_audio: Optional[str]
    begin_time: float = 0
    end_time: Optional[float] = None

    fps: int

    trigger_ms: int
    render_ms: int

    # Performance
    trigger_subsampling: int = 1
    render_subsampling: int = 1

    # Performance (skipped when recording to video)
    render_subfps: int = 1
    render_fps = property(lambda self: Fraction(self.fps, self.render_subfps))
    # FFmpeg accepts FPS as a fraction. (decimals may work, but are inaccurate.)

    def before_preview(self) -> None:
        """ Called *once* before preview. Decreases render fps/etc. """
        self.render.before_preview()

    def before_record(self) -> None:
        """ Called *once* before recording video. Force high-quality rendering. """
        self.render_subfps = 1
        self.render.before_record()

    # End Performance
    amplification: float

    trigger: ITriggerConfig  # Can be overriden per Wave

    # Multiplies by trigger_width, render_width. Can override trigger.
    channels: List[ChannelConfig]

    layout: LayoutConfig
    render: RendererConfig

    show_internals: List[str] = attr.Factory(list)
    benchmark_mode: Union[str, BenchmarkMode] = BenchmarkMode.NONE

    def __attrs_post_init__(self):
        # Cast benchmark_mode to enum.
        try:
            if not isinstance(self.benchmark_mode, BenchmarkMode):
                self.benchmark_mode = BenchmarkMode[self.benchmark_mode]
        except KeyError:
            raise CorrError(
                f"invalid benchmark_mode mode {self.benchmark_mode} not in "
                f"{[el.name for el in BenchmarkMode]}"
            )


_FPS = 60  # f_s


def default_config(**kwargs) -> Config:
    """ Default template values do NOT indicate optional attributes. """
    cfg = Config(
        render_subfps=1,
        master_audio="",
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


class CorrScope:
    def __init__(self, cfg: Config, arg: Arguments):
        """ cfg is mutated!
        Recording config is triggered if any FFmpegOutputConfig is found.
        Preview mode is triggered if all outputs are FFplay or others.
        """
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
            raise CorrError("Config.channels is empty")

        # Check for ffmpeg video recording, then mutate cfg.
        is_record = False
        for output in self.output_cfgs:
            if isinstance(output, outputs_.FFmpegOutputConfig):
                is_record = True
                break
        if is_record:
            self.cfg.before_record()
        else:
            self.cfg.before_preview()

    waves: List["Wave"]
    channels: List[Channel]
    outputs: List[outputs_.Output]
    nchan: int

    def _load_channels(self):
        with pushd(self.arg.cfg_dir):
            # Tell user if master audio path is invalid.
            # (Otherwise, only ffmpeg uses the value of master_audio)
            # Windows likes to raise OSError when path contains *, but we don't care.
            if self.cfg.master_audio and not Path(self.cfg.master_audio).exists():
                raise CorrError(
                    f'File not found: master_audio="{self.cfg.master_audio}"'
                )
            self.channels = [Channel(ccfg, self.cfg) for ccfg in self.cfg.channels]
            self.waves = [channel.wave for channel in self.channels]
            self.triggers = [channel.trigger for channel in self.channels]
            self.nchan = len(self.channels)

    def play(self):
        if self.has_played:
            raise ValueError("Cannot call CorrScope.play() more than once")
        self.has_played = True

        self._load_channels()
        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps

        begin_frame = round(fps * self.cfg.begin_time)

        end_time = coalesce(self.cfg.end_time, self.waves[0].get_s())
        end_frame = fps * end_time
        end_frame = int(end_frame) + 1

        self.arg.on_begin(self.cfg.begin_time, end_time)

        # region show_internals
        # Display buffers, for debugging purposes.
        internals = self.cfg.show_internals
        extra_outputs = SimpleNamespace()
        if internals:
            from corrscope.outputs import FFplayOutputConfig
            import attr

            no_audio = attr.evolve(self.cfg, master_audio="")

            corr = self

            class RenderOutput:
                def __init__(self):
                    self.renderer = corr._load_renderer()
                    self.output = FFplayOutputConfig()(no_audio)

                def render_frame(self, datas):
                    self.renderer.render_frame(datas)
                    self.output.write_frame(self.renderer.get_frame())

        extra_outputs.window = None
        if "window" in internals:
            extra_outputs.window = RenderOutput()

        extra_outputs.buffer = None
        if "buffer" in internals:
            extra_outputs.buffer = RenderOutput()
        # endregion

        if PRINT_TIMESTAMP:
            begin = time.perf_counter()

        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        parent, child = Pipe(duplex=True)
        parent_send = connection_host(parent)

        render_thread = Process(
            name="render_thread",
            target=self.render_worker,
            args=(child, benchmark_mode, not_benchmarking),
        )
        del child
        render_thread.start()

        prev = -1
        try:
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

                    if not_benchmarking or (
                        BenchmarkMode.TRIGGER
                        <= benchmark_mode
                        <= BenchmarkMode.SEND_TO_WORKER
                    ):
                        cache = PerFrameCache()
                        trigger_sample = channel.trigger.get_trigger(sample, cache)
                    else:
                        trigger_sample = sample
                    if should_render:
                        render_datas.append(
                            wave.get_around(
                                trigger_sample,
                                channel.render_samp,
                                channel.render_stride,
                            )
                        )

                if not should_render:
                    continue

                # region Display buffers, for debugging purposes.
                if extra_outputs.window:
                    triggers: List["CorrelationTrigger"] = self.triggers
                    extra_outputs.window.render_frame(
                        [trigger._prev_window for trigger in triggers]
                    )

                if extra_outputs.buffer:
                    triggers: List["CorrelationTrigger"] = self.triggers
                    extra_outputs.buffer.render_frame(
                        [trigger._buffer for trigger in triggers]
                    )
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
            print(f"FPS = {render_fps}")

        # Print FPS before waiting for render thread to terminate.
        render_thread.join()

    raise_on_teardown: Optional[Exception] = None

    #### Render/output thread
    def render_worker(self, conn: "Connection", benchmark_mode, not_benchmarking):
        """ Communicates with main process via `pipe`.
        Accepts QueueMessage, sends Optional[exception info].

        Accepts N QueueMessage followed by None.
        Replies with N Optional[exception info].

        The parent checks for replies N times (before sending N-1 QueueMessage + None)
        """
        # assert threading.current_thread() != threading.main_thread()

        renderer = self._load_renderer()
        with self._load_outputs():
            # foreach frame
            for datas in iter_conn(conn):  # type: Message
                should_break = False
                try:
                    # Render frame
                    if not_benchmarking or benchmark_mode >= BenchmarkMode.RENDER:
                        renderer.render_frame(datas)
                        frame_data = renderer.get_frame()

                        if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
                            # Output frame
                            for output in self.outputs:
                                if output.write_frame(frame_data) is outputs_.Stop:
                                    should_break = True
                except BaseException as e:
                    conn.send(True)
                    raise

                conn.send(should_break)
                if should_break:
                    break

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
        renderer = MatplotlibRenderer(
            self.cfg.render, self.cfg.layout, self.nchan, self.cfg.channels
        )
        return renderer
