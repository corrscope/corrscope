# -*- coding: utf-8 -*-
import os.path
import threading
import time
from concurrent.futures import ProcessPoolExecutor, Future
from contextlib import ExitStack, contextmanager
from enum import unique
from fractions import Fraction
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Iterator, Optional, List, Callable, Dict, Union, Any

import attr
import numpy as np

from corrscope import outputs as outputs_
from corrscope.channel import Channel, ChannelConfig, DefaultLabel
from corrscope.config import KeywordAttrs, DumpEnumAsStr, CorrError, with_units
from corrscope.layout import LayoutConfig
from corrscope.outputs import FFmpegOutputConfig, IOutputConfig
from corrscope.renderer import (
    Renderer,
    RendererConfig,
    RendererParams,
    RenderInput,
    StereoLevels,
)
from corrscope.settings.global_prefs import Parallelism
from corrscope.triggers import (
    CorrelationTriggerConfig,
    PerFrameCache,
    SpectrumConfig,
)
from corrscope.util import pushd, coalesce
from corrscope.wave import Wave, Flatten, FlattenOrStr

PRINT_TIMESTAMP = True


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
    end_time: Optional[float] = with_units("s", default=None)

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
        default_label=DefaultLabel.FileName,
        layout=LayoutConfig(orientation="h", stereo_orientation="v", ncols=0),
        render=RendererConfig(
            1920,
            1080,
            res_divisor=3 / 2,
            grid_color="#55aaff",
            v_midline=True,
            h_midline=True,
        ),
    )
    return attr.evolve(cfg, **kwargs)


class PropagatingThread(Thread):
    # Based off https://stackoverflow.com/a/31614591 and Thread source code.
    def run(self):
        self.exc = None
        try:
            if self._target is not None:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def join(self, timeout=None) -> Any:
        try:
            super(PropagatingThread, self).join(timeout)
            if self.exc:
                raise RuntimeError(f"exception from {self.name}") from self.exc

            return self.ret
        finally:
            # If join() raises, set `self = None` to avoid a reference cycle with the
            # backtrace, because concurrent.futures.Future.result() does it.
            self = None


BeginFunc = Callable[[float, float], None]
ProgressFunc = Callable[[int], None]
IsAborted = Callable[[], bool]


@attr.dataclass
class Arguments:
    cfg_dir: str
    outputs: List[outputs_.IOutputConfig]
    parallelism: Optional[Parallelism] = None

    on_begin: BeginFunc = lambda begin_time, end_time: None
    progress: ProgressFunc = lambda p: print(p, flush=True)
    is_aborted: IsAborted = lambda: False
    on_end: Callable[[], None] = lambda: None


def worker_create_renderer(renderer_params: RendererParams, shmem_names: List[str]):
    import appnope

    # Disable power saving for renderer processes.
    appnope.nope()

    global WORKER_RENDERER
    global SHMEMS

    WORKER_RENDERER = Renderer(renderer_params)
    SHMEMS = {
        name: SharedMemory(name) for name in shmem_names
    }  # type: Dict[str, SharedMemory]


prev = 0.0


def worker_render_frame(
    render_inputs: List[RenderInput],
    trigger_samples: List[int],
    shmem_name: str,
):
    global WORKER_RENDERER, SHMEMS, prev
    t = time.perf_counter() * 1000.0

    renderer = WORKER_RENDERER
    renderer.update_main_lines(render_inputs, trigger_samples)
    frame_data = renderer.get_frame()
    t1 = time.perf_counter() * 1000.0

    shmem = SHMEMS[shmem_name]
    shmem.buf[: len(frame_data)] = frame_data
    t2 = time.perf_counter() * 1000.0
    # print(f"idle = {t - prev}, dt1 = {t1 - t}, dt2 = {t2 - t1}")
    prev = t2


def calc_stereo_levels(data: np.ndarray) -> StereoLevels:
    def amplitude(chan_data: np.ndarray) -> float:
        sq = chan_data * chan_data
        mean = np.add.reduce(sq) / len(sq)
        root = np.sqrt(mean)
        return root

    return (amplitude(data.T[0]), amplitude(data.T[1]))


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

    outputs: List[outputs_.Output]
    nchan: int

    @contextmanager
    def _load_outputs(self) -> Iterator[None]:
        with pushd(self.arg.cfg_dir):
            with ExitStack() as stack:
                self.outputs = [
                    stack.enter_context(output_cfg(self.cfg))
                    for output_cfg in self.output_cfgs
                ]
                yield

    def _renderer_params(self) -> RendererParams:
        dummy_datas = []
        return RendererParams.from_obj(
            self.cfg.render,
            self.cfg.layout,
            dummy_datas,
            self.cfg.channels,
            None,
            self.arg.cfg_dir,
        )

    # def _load_renderer(self) -> Renderer:
    #     # only kept for unit tests I'm too lazy to rewrite.
    #     return Renderer(self._renderer_params())

    def play(self) -> None:
        if self.has_played:
            raise ValueError("Cannot call CorrScope.play() more than once")
        self.has_played = True

        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps

        begin_frame = round(fps * self.cfg.begin_time)

        end_time = self.cfg.end_time
        end_frame = fps * end_time
        end_frame = int(end_frame) + 1

        @attr.dataclass
        class ThreadShared:
            # mutex? i hardly knew 'er!
            end_frame: int

        thread_shared = ThreadShared(end_frame)
        del end_frame

        self.arg.on_begin(self.cfg.begin_time, end_time)

        renderer_params = self._renderer_params()
        renderer = Renderer(renderer_params)
        self.renderer = renderer  # only used for unit tests

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
            pt = 0.0

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

                if not should_render:
                    continue

                if not_benchmarking or benchmark_mode >= BenchmarkMode.RENDER:
                    # Render frame

                    t = time.perf_counter() * 1000.0
                    frame_data = renderer.get_frame()
                    t1 = time.perf_counter() * 1000.0
                    # print(f"idle = {t - pt}, dt1 = {t1 - t}")
                    pt = t1

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

        with self._load_outputs():
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
