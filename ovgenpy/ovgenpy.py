# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import NamedTuple, Optional, List

import click

from ovgenpy.renderer import MatplotlibRenderer, RendererConfig
from ovgenpy.triggers import TriggerConfig
from ovgenpy.wave import WaveConfig, Wave


RENDER_PROFILING = True


class Config(NamedTuple):
    wave_dir: str
    # TODO: if wave_dir is present, it should overwrite List[WaveConfig].
    # wave_dir will be commented out when writing to file.

    master_wave: Optional[str]

    fps: int

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
        trigger=TriggerConfig(     # todo
            name='CorrelationTrigger',
            kwargs=dict(    # TODO: CorrelationTriggerConfig with default values
                window_width=0.5,
                trigger_strength=0.1
            )
        ),
        render=RendererConfig(     # todo
            1280, 720,
            samples_visible=1000,
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

    def write(self):
        self.load_waves()  # self.waves =
        self.render()

    def load_waves(self):
        wave_dir = Path(self.cfg.wave_dir)

        for idx, path in enumerate(wave_dir.glob('*.wav')):
            wcfg = WaveConfig(
                wave_path=str(path)
            )

            wave = Wave(wcfg, str(path))
            wave.set_trigger(self.cfg.trigger.generate_trigger(
                wave=wave,
                scan_nsamp=wave.smp_s // self.cfg.fps,  # TODO multiply by a thing
            ))
            self.waves.append(wave)

    def render(self):
        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps
        nframes = fps * self.waves[0].get_s()
        nframes = int(nframes) + 1

        renderer = MatplotlibRenderer(self.cfg.render, self.waves)

        if RENDER_PROFILING:
            begin = time.perf_counter()

        # For each frame, render each wave
        for frame in range(nframes):
            time_seconds = frame / fps

            center_smps = []
            for wave in self.waves:
                sample = round(wave.smp_s * time_seconds)
                trigger_sample = wave.trigger.get_trigger(sample)
                center_smps.append(trigger_sample)

            print(frame)
            renderer.render_frame(center_smps)

        if RENDER_PROFILING:
            dtime = time.perf_counter() - begin
            render_fps = nframes / dtime
            print(f'FPS = {render_fps}')
