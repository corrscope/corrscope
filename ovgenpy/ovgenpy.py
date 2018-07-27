# -*- coding: utf-8 -*-
import cgitb
import sys
import time
from pathlib import Path
from typing import Optional, List

import click

from ovgenpy import outputs
from ovgenpy.channel import Channel
from ovgenpy.config import register_config, yaml
from ovgenpy.renderer import MatplotlibRenderer, RendererConfig
from ovgenpy.triggers import ITriggerConfig, CorrelationTriggerConfig
from ovgenpy.utils.keyword_dataclasses import field
from ovgenpy.wave import _WaveConfig, Wave


# cgitb.enable(format='text')

RENDER_PROFILING = True


@register_config(always_dump='*')
class Config:
    wave_prefix: str = ''   # if wave/glob..., pwd. if folder, folder.
    channels: List[Channel] = field(default_factory=lambda: [])

    master_audio: Optional[str]
    fps: int

    time_visible_ms: int
    scan_ratio: float
    trigger: ITriggerConfig  # Maybe overriden per Wave

    amplification: float
    render: RendererConfig

    outputs: List[outputs.IOutputConfig]
    create_window: bool

    @property
    def time_visible_s(self) -> float:
        return self.time_visible_ms / 1000


Folder = click.Path(exists=True, file_okay=False)
File = click.Path(exists=True, dir_okay=False)

_FPS = 60  # f_s


def main():
    cfg = Config(
        wave_prefix='foo',
        channels=[],

        master_audio=None,
        fps=69,

        time_visible_ms=25,
        scan_ratio=1,
        trigger=CorrelationTriggerConfig(
            trigger_strength=1,
            use_edge_trigger=False,

            responsiveness=1,
            falloff_width=1,
        ),

        amplification=5,
        render=RendererConfig(  # todo
            1280, 720,
            ncols=1
        ),

        outputs=[
            # outputs.FFmpegOutputConfig(output),
            outputs.FFplayOutputConfig(),
        ],
        create_window=False
    )

    yaml.dump(cfg, sys.stdout)
    return

    ovgen = Ovgen(cfg)
    ovgen.write()


class Ovgen:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.waves: List[Wave] = []
        self.channels: List[Channel] = []
        self.outputs: List[outputs.Output] = []
        self.nchan: int = None

    def write(self):
        self._load_waves()  # self.waves =
        self._load_outputs()  # self.outputs =
        self._render()

    def _load_waves(self):
        wave_dir = Path(self.cfg.wave_folder)

        waves = sorted(wave_dir.glob('*.wav'))
        for idx, path in enumerate(waves):
            wcfg = _WaveConfig(
                amplification=self.cfg.amplification
            )

            wave = Wave(wcfg, str(path))
            self.waves.append(wave)

            trigger = self.cfg.trigger(
                wave=wave,
                scan_nsamp=round(
                    self.cfg.time_visible_s * self.cfg.scan_ratio * wave.smp_s),
                # I tried extracting variable, but got confused as a result
            )
            channel = Channel(None, wave, trigger)
            self.channels.append(channel)

        self.nchan = len(self.waves)

    def _load_outputs(self):
        self.outputs = []
        for output_cfg in self.cfg.outputs:
            output = output_cfg(self.cfg)
            self.outputs.append(output)

    def _render(self):
        # Calculate number of frames (TODO master file?)
        time_visible_s = self.cfg.time_visible_s
        fps = self.cfg.fps
        create_window = self.cfg.create_window

        nframes = fps * self.waves[0].get_s()
        nframes = int(nframes) + 1

        renderer = MatplotlibRenderer(self.cfg.render, self.nchan, create_window)

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
                region_len = round(wave.smp_s * time_visible_s)

                trigger_sample = channel.trigger.get_trigger(sample)
                datas.append(wave.get_around(trigger_sample, region_len))

            # Render frame
            renderer.render_frame(datas)

            # Output frame
            if self.outputs:
                frame = renderer.get_frame()
            for output in self.outputs:
                output.write_frame(frame)

        if RENDER_PROFILING:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = nframes / dtime
            print(f'FPS = {render_fps}')
