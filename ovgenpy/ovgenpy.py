# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import NamedTuple, Optional, List

import click

from ovgenpy.renderer import MatplotlibRenderer, RendererConfig
from ovgenpy.triggers import TriggerConfig, CorrelationTrigger
from ovgenpy.wave import WaveConfig, Wave


RENDER_PROFILING = True


class Config(NamedTuple):
    wave_dir: str
    master_wave: Optional[str]
    fps: int
    time_visible_ms: int

    trigger: TriggerConfig  # Maybe overriden per Wave
    render: RendererConfig


Folder = click.Path(exists=True, file_okay=False)
File = click.Path(exists=True, dir_okay=False)

_FPS = 60  # f_s


@click.command()
@click.argument('wave_dir', type=Folder)
@click.option('--master-wave', type=File, default=None)
@click.option('--fps', default=_FPS)
def main(wave_dir: str, master_wave: Optional[str], fps: int):
    cfg = Config(
        wave_dir=wave_dir,
        master_wave=master_wave,
        fps=fps,
        time_visible_ms=25,

        trigger=CorrelationTrigger.Config(
            trigger_strength=10,
            use_edge_trigger=True,

            responsiveness=1,
            falloff_width=.5,
        ),
        render=RendererConfig(     # todo
            1280, 720,
            rows_first=False,
            ncols=1
        )
    )

    ovgen = Ovgen(cfg)
    ovgen.write()


COLOR_CHANNELS = 3


class Ovgen:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.waves: List[Wave] = []
        self.nwaves: int = None

    def write(self):
        self._load_waves()  # self.waves =
        self._render()

    def _load_waves(self):
        wave_dir = Path(self.cfg.wave_dir)

        for idx, path in enumerate(wave_dir.glob('*.wav')):
            wcfg = WaveConfig(
                wave_path=str(path)
            )

            wave = Wave(wcfg, str(path))
            trigger = self.cfg.trigger(
                wave=wave,
                scan_nsamp=wave.smp_s // self.cfg.fps,  # TODO multiply by a thing
            )
            wave.set_trigger(trigger)
            self.waves.append(wave)

        self.nwaves = len(self.waves)

    def _render(self):
        # Calculate number of frames (TODO master file?)
        time_visible_ms = self.cfg.time_visible_ms
        fps = self.cfg.fps

        nframes = fps * self.waves[0].get_s()
        nframes = int(nframes) + 1

        renderer = MatplotlibRenderer(self.cfg.render, self.nwaves)

        if RENDER_PROFILING:
            begin = time.perf_counter()

        # For each frame, render each wave
        for frame in range(nframes):
            time_seconds = frame / fps

            datas = []
            # Get data from each wave
            for wave in self.waves:
                sample = round(wave.smp_s * time_seconds)
                region_len = round(wave.smp_s * time_visible_ms / 1000)

                trigger_sample = wave.trigger.get_trigger(sample)
                print(f'- {trigger_sample}')

                datas.append(wave.get_around(trigger_sample, region_len))

            print(frame)
            renderer.render_frame(datas)

        if RENDER_PROFILING:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = nframes / dtime
            print(f'FPS = {render_fps}')
