# -*- coding: utf-8 -*-
import os.path
import threading
import time
from concurrent.futures import ProcessPoolExecutor, Future
from contextlib import ExitStack, contextmanager
from enum import unique
from fractions import Fraction
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Iterator, Optional, List, Callable, Tuple

import attr

from corrscope import outputs as outputs_
from corrscope.channel import Channel, ChannelConfig, DefaultLabel
from corrscope.config import KeywordAttrs, DumpEnumAsStr, CorrError, with_units
from corrscope.layout import LayoutConfig
from corrscope.outputs import FFmpegOutputConfig, IOutputConfig
from corrscope.renderer import (
    Renderer,
    RendererConfig,
    RendererFrontend,
    RenderInput,
    ByteBuffer,
)
from corrscope.triggers import (
    CorrelationTriggerConfig,
    PerFrameCache,
    SpectrumConfig,
)
from corrscope.util import pushd, coalesce
from corrscope.wave import Wave, Flatten, FlattenOrStr

PRINT_TIMESTAMP = True

NAMED_THREADS = False


def named_threads():
    global NAMED_THREADS
    if NAMED_THREADS:
        return
    NAMED_THREADS = True

    LIB = "libcap.so.2"
    try:
        import ctypes

        libcap = ctypes.CDLL(LIB)
    except OSError:
        print("Library {} not found. Unable to set thread name.".format(LIB))
    else:

        def _name_hack(self):
            # PR_SET_NAME = 15
            libcap.prctl(15, self.name.encode())
            threading.Thread._bootstrap_original(self)

        threading.Thread._bootstrap_original = threading.Thread._bootstrap
        threading.Thread._bootstrap = _name_hack


named_threads()

# Placing Enum before any other superclass results in errors.
# Placing DumpEnumAsStr before IntEnum or (int, Enum) results in errors on Python 3.6:
# - TypeError: object.__new__(BenchmarkMode) is not safe, use int.__new__()
# I don't know *why* this works. It's magic.
@unique
class BenchmarkMode(int, DumpEnumAsStr):
    NONE = 0
    TRIGGER = 1
    RENDER = 2
    OUTPUT = 3


class Config(
    KeywordAttrs,
    always_dump="""
    begin_time end_time
    render_subfps trigger_subsampling render_subsampling
    trigger_stereo render_stereo
    """,
):
    """Default values indicate optional attributes."""

    master_audio: Optional[str]
    begin_time: float = with_units("s", default=0)
    end_time: Optional[float] = None

    fps: int

    trigger_ms: int = with_units("ms")
    render_ms: int = with_units("ms")

    # Performance
    trigger_subsampling: int = 1
    render_subsampling: int = 1

    # Performance (skipped when recording to video)
    render_subfps: int = 2
    render_fps = property(lambda self: Fraction(self.fps, self.render_subfps))
    # FFmpeg accepts FPS as a fraction. (decimals may work, but are inaccurate.)

    # Both before_* functions should be idempotent, AKA calling twice does no harm.
    def before_preview(self) -> None:
        """Called *once* before preview. Does nothing."""
        self.render.before_preview()

    def before_record(self) -> None:
        """Called *once* before recording video. Force high-quality rendering."""
        self.render_subfps = 1
        self.trigger_subsampling = 1
        self.render_subsampling = 1
        self.render.before_record()

    # End Performance
    amplification: float

    # Stereo config
    trigger_stereo: FlattenOrStr = Flatten.SumAvg
    render_stereo: FlattenOrStr = Flatten.SumAvg

    trigger: CorrelationTriggerConfig  # Can be overriden per Wave

    # Multiplies by trigger_width, render_width. Can override trigger.
    channels: List[ChannelConfig]
    default_label: DefaultLabel = DefaultLabel.NoLabel

    layout: LayoutConfig
    render: RendererConfig
    ffmpeg_cli: FFmpegOutputConfig = attr.ib(factory=lambda: FFmpegOutputConfig(None))

    def get_ffmpeg_cfg(self, video_path: str) -> FFmpegOutputConfig:
        return attr.evolve(self.ffmpeg_cli, path=os.path.abspath(video_path))

    benchmark_mode: BenchmarkMode = attr.ib(
        BenchmarkMode.NONE, converter=BenchmarkMode.by_name
    )


_FPS = 60  # f_s


def template_config(**kwargs) -> Config:
    """Default template values do NOT indicate optional attributes."""
    cfg = Config(
        master_audio="",
        fps=_FPS,
        amplification=1,
        trigger_ms=40,
        render_ms=40,
        trigger_subsampling=1,
        render_subsampling=2,
        trigger=CorrelationTriggerConfig(
            mean_responsiveness=0.0,
            edge_strength=1.0,
            responsiveness=0.5,
            reset_below=0.3,
            pitch_tracking=SpectrumConfig(),
            # post_trigger=ZeroCrossingTriggerConfig(),
        ),
        channels=[],
        layout=LayoutConfig(orientation="v", ncols=1),
        render=RendererConfig(
            1280,
            720,
            res_divisor=4 / 3,
            grid_color="#55aaff",
            v_midline=True,
            h_midline=True,
        ),
    )
    return attr.evolve(cfg, **kwargs)


BeginFunc = Callable[[float, float], None]
ProgressFunc = Callable[[int], None]
IsAborted = Callable[[], bool]


@attr.dataclass
class Arguments:
    cfg_dir: str
    outputs: List[outputs_.IOutputConfig]
    parallel: bool = False

    on_begin: BeginFunc = lambda begin_time, end_time: None
    progress: ProgressFunc = lambda p: print(p, flush=True)
    is_aborted: IsAborted = lambda: False
    on_end: Callable[[], None] = lambda: None


class CorrScope:
    def __init__(self, cfg: Config, arg: Arguments):
        """cfg is mutated!
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
            self.output_cfgs = []  # type: List[IOutputConfig]

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

    trigger_waves: List[Wave]
    render_waves: List[Wave]
    channels: List[Channel]
    outputs: List[outputs_.Output]
    nchan: int

    def _load_channels(self) -> None:
        with pushd(self.arg.cfg_dir):
            # Tell user if master audio path is invalid.
            # (Otherwise, only ffmpeg uses the value of master_audio)
            # Windows likes to raise OSError when path contains *, but we don't care.
            if self.cfg.master_audio and not Path(self.cfg.master_audio).exists():
                raise CorrError(
                    f'File not found: master_audio="{self.cfg.master_audio}"'
                )
            self.channels = [
                Channel(ccfg, self.cfg, idx)
                for idx, ccfg in enumerate(self.cfg.channels)
            ]
            self.trigger_waves = [channel.trigger_wave for channel in self.channels]
            self.render_waves = [channel.render_wave for channel in self.channels]
            self.triggers = [channel.trigger for channel in self.channels]
            self.nchan = len(self.channels)

    @contextmanager
    def _load_outputs(self) -> Iterator[None]:
        with pushd(self.arg.cfg_dir):
            with ExitStack() as stack:
                self.outputs = [
                    stack.enter_context(output_cfg(self.cfg))
                    for output_cfg in self.output_cfgs
                ]
                yield

    def _load_renderer(self) -> RendererFrontend:
        dummy_datas = [channel.get_render_around(0) for channel in self.channels]
        renderer = Renderer(
            self.cfg.render,
            self.cfg.layout,
            dummy_datas,
            self.cfg.channels,
            self.channels,
        )
        return renderer

    def play(self) -> None:
        if self.has_played:
            raise ValueError("Cannot call CorrScope.play() more than once")
        self.has_played = True

        self._load_channels()
        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps

        begin_frame = round(fps * self.cfg.begin_time)

        end_time = coalesce(
            self.cfg.end_time, max(wave.get_s() for wave in self.render_waves)
        )
        end_frame = fps * end_time
        end_frame = int(end_frame) + 1

        @attr.dataclass
        class ThreadShared:
            # mutex? i hardly knew 'er!
            end_frame: int

        thread_shared = ThreadShared(end_frame)
        del end_frame

        self.arg.on_begin(self.cfg.begin_time, end_time)

        renderer = self._load_renderer()
        self.renderer = renderer  # only used for unit tests

        renderer.add_labels([channel.label for channel in self.channels])

        # For debugging only
        # for trigger in self.triggers:
        #     trigger.set_renderer(renderer)

        if PRINT_TIMESTAMP:
            begin = time.perf_counter()

        benchmark_mode = self.cfg.benchmark_mode
        not_benchmarking = not benchmark_mode

        # When subsampling FPS, render frames from the future to alleviate lag.
        # subfps=1, ahead=0.
        # subfps=2, ahead=1.
        render_subfps = self.cfg.render_subfps
        ahead = render_subfps // 2

        # Single-process
        def play_impl():
            end_frame = thread_shared.end_frame
            prev = -1

            # For each frame, render each wave
            for frame in range(begin_frame, end_frame):
                if self.arg.is_aborted():
                    # Used for FPS calculation
                    thread_shared.end_frame = frame

                    for output in self.outputs:
                        output.terminate()
                    break

                time_seconds = frame / fps
                should_render = (frame - begin_frame) % render_subfps == ahead

                rounded = int(time_seconds)
                if PRINT_TIMESTAMP and rounded != prev:
                    self.arg.progress(rounded)
                    prev = rounded

                render_inputs = []
                trigger_samples = []
                # Get render-data from each wave.
                for render_wave, channel in zip(self.render_waves, self.channels):
                    sample = round(render_wave.smp_s * time_seconds)

                    # Get trigger.
                    if not_benchmarking or benchmark_mode == BenchmarkMode.TRIGGER:
                        cache = PerFrameCache()

                        result = channel.trigger.get_trigger(sample, cache)
                        trigger_sample = result.result
                        freq_estimate = result.freq_estimate

                    else:
                        trigger_sample = sample
                        freq_estimate = 0

                    # Get render data.
                    if should_render:
                        trigger_samples.append(trigger_sample)
                        data = channel.get_render_around(trigger_sample)
                        render_inputs.append(RenderInput(data, freq_estimate))

                if not should_render:
                    continue

                if not_benchmarking or benchmark_mode >= BenchmarkMode.RENDER:
                    # Render frame
                    renderer.update_main_lines(render_inputs, trigger_samples)
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
                            thread_shared.end_frame = frame + 1
                            break

        # Multiprocess
        def play_parallel():
            ncores = len(os.sched_getaffinity(0))

            abort_from_thread = threading.Event()
            # self.arg.is_aborted() from GUI, abort_from_thread.is_set() from thread
            is_aborted = lambda: self.arg.is_aborted() or abort_from_thread.is_set()

            # Same size as ProcessPoolExecutor, so threads won't starve if they all
            # finish a job at the same time.
            render_to_output: "Queue[Tuple[int, Future[ByteBuffer]] | None]" = Queue(
                ncores
            )

            def worker_create_renderer(renderer: RendererFrontend):
                global WORKER_RENDERER
                # TODO del self.renderer and recreate Renderer if it can't be pickled?
                WORKER_RENDERER = renderer

            # TODO https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
            def render_thread():
                end_frame = thread_shared.end_frame
                prev = -1

                # TODO gather trigger points from triggering threads
                # For each frame, render each wave
                for frame in range(begin_frame, end_frame):
                    if is_aborted():
                        # Only count output-displayed frames, not rendered.
                        # # Used for FPS calculation
                        # thread_shared.end_frame = frame
                        break

                    time_seconds = frame / fps
                    should_render = (frame - begin_frame) % render_subfps == ahead

                    rounded = int(time_seconds)
                    if PRINT_TIMESTAMP and rounded != prev:
                        self.arg.progress(rounded)
                        prev = rounded

                    render_inputs = []
                    trigger_samples = []
                    # Get render-data from each wave.
                    for render_wave, channel in zip(self.render_waves, self.channels):
                        sample = round(render_wave.smp_s * time_seconds)

                        # Get trigger.
                        if not_benchmarking or benchmark_mode == BenchmarkMode.TRIGGER:
                            cache = PerFrameCache()

                            result = channel.trigger.get_trigger(sample, cache)
                            trigger_sample = result.result
                            freq_estimate = result.freq_estimate

                        else:
                            trigger_sample = sample
                            freq_estimate = 0

                        # Get render data.
                        if should_render:
                            trigger_samples.append(trigger_sample)
                            data = channel.get_render_around(trigger_sample)
                            render_inputs.append(RenderInput(data, freq_estimate))

                    if not should_render:
                        continue

                    # blocking
                    render_to_output.put(
                        (
                            frame,
                            pool.submit(
                                worker_render_frame, render_inputs, trigger_samples
                            ),
                        )
                    )

                render_to_output.put(None)
                print("exit render")

            global worker_render_frame  # hack to allow pickling function

            def worker_render_frame(
                render_inputs: List[RenderInput], trigger_samples: List[int]
            ) -> ByteBuffer:
                global WORKER_RENDERER
                renderer = WORKER_RENDERER
                renderer.update_main_lines(render_inputs, trigger_samples)
                frame_data = renderer.get_frame()
                return bytes(frame_data)

            def output_thread():
                while True:
                    if is_aborted():
                        for output in self.outputs:
                            output.terminate()
                        break

                    msg = render_to_output.get()  # blocking
                    if msg is None:
                        break
                    frame, render_future = msg
                    frame_data: ByteBuffer = render_future.result()

                    if not_benchmarking or benchmark_mode == BenchmarkMode.OUTPUT:
                        # Output frame
                        for output in self.outputs:
                            if output.write_frame(frame_data) is outputs_.Stop:
                                abort_from_thread.set()
                                break
                        if is_aborted():
                            # Outputting frame happens after most computation finished.
                            thread_shared.end_frame = frame + 1
                            break

                if is_aborted():
                    # If is_aborted() is True but render_thread() is blocked on
                    # render_to_output.put(), then we need to clear the queue so
                    # render_thread() can return from put(), then check is_aborted()
                    # = True and terminate.
                    while True:
                        try:
                            render_to_output.get(block=False)
                        except Empty:
                            break

                print("exit output")

            with ProcessPoolExecutor(
                ncores, initializer=worker_create_renderer, initargs=(renderer,)
            ) as pool:
                render_handle = Thread(target=render_thread, name="render_thread")
                output_handle = Thread(target=output_thread, name="output_thread")

                render_handle.start()
                output_handle.start()

                render_handle.join()
                output_handle.join()

        with self._load_outputs():
            if self.arg.parallel:
                play_parallel()
            else:
                play_impl()

        if PRINT_TIMESTAMP:
            # noinspection PyUnboundLocalVariable
            dtime_sec = time.perf_counter() - begin
            dframe = thread_shared.end_frame - begin_frame

            frame_per_sec = dframe / dtime_sec
            try:
                msec_per_frame = 1000 * dtime_sec / dframe
            except ZeroDivisionError:
                msec_per_frame = float("inf")

            print(f"{frame_per_sec:.1f} FPS, {msec_per_frame:.2f} ms/frame")
