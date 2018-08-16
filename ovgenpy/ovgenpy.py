# -*- coding: utf-8 -*-
import time
from typing import Optional, List

from ovgenpy import outputs
from ovgenpy.channel import Channel, ChannelConfig
from ovgenpy.config import register_config
from ovgenpy.renderer import MatplotlibRenderer, RendererConfig, LayoutConfig
from ovgenpy.triggers import ITriggerConfig, CorrelationTriggerConfig
from ovgenpy.utils import keyword_dataclasses as dc
from ovgenpy.utils.keyword_dataclasses import field
from ovgenpy.wave import Wave


# cgitb.enable(format='text')

RENDER_PROFILING = True


@register_config(always_dump='wave_prefix')
class Config:
    master_audio: Optional[str]
    fps: int

    wav_prefix: str = ''   # if wave/glob..., pwd. if folder, folder.
    channels: List[ChannelConfig] = field(default_factory=list)

    width_ms: int
    subsampling: int
    trigger: ITriggerConfig  # Maybe overriden per Wave

    amplification: float
    layout: LayoutConfig
    render: RendererConfig

    outputs: List[outputs.IOutputConfig]

    @property
    def render_width_s(self) -> float:
        return self.width_ms / 1000


_FPS = 60  # f_s

def default_config(**kwargs):
    cfg = Config(
        master_audio='',
        fps=_FPS,

        wav_prefix='',
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
        self._load_channels()  # self.waves =
        self._load_outputs()  # self.outputs =

    waves: List[Wave]
    channels: List[Channel]
    outputs: List[outputs.Output]
    nchan: int

    def _load_channels(self):
        self.channels = [Channel(ccfg, self.cfg) for ccfg in self.cfg.channels]
        self.waves = [channel.wave for channel in self.channels]
        self.nchan = len(self.channels)

    def _load_outputs(self):
        self.outputs = [output_cfg(self.cfg) for output_cfg in self.cfg.outputs]

    def play(self):
        # Calculate number of frames (TODO master file?)
        render_width_s = self.cfg.render_width_s
        fps = self.cfg.fps

        nframes = fps * self.waves[0].get_s()
        nframes = int(nframes) + 1

        renderer = MatplotlibRenderer(self.cfg.render, self.cfg.layout, self.nchan)

        if RENDER_PROFILING:
            begin = time.perf_counter()

        # For each frame, render each wave
        prev = -1
        for frame in range(nframes):
            time_seconds = frame / fps

            rounded = int(time_seconds)
            if rounded != prev:
                print(rounded)
                prev = rounded

            datas = []
            # Get data from each wave
            for wave, channel in zip(self.waves, self.channels):
                sample = round(wave.smp_s * time_seconds)
                region_len = round(wave.smp_s * render_width_s)

                trigger_sample = channel.trigger.get_trigger(sample)
                datas.append(wave.get_around(
                    trigger_sample, region_len, channel.render_subsampling))

            # Render frame
            renderer.render_frame(datas)

            # Output frame
            frame = renderer.get_frame()
            for output in self.outputs:
                output.write_frame(frame)

        if RENDER_PROFILING:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = nframes / dtime
            print(f'FPS = {render_fps}')
