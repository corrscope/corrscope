# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import NamedTuple, Optional, List

import click
from ovgenpy import outputs

from ovgenpy.renderer import MatplotlibRenderer, RendererConfig
from ovgenpy.triggers import TriggerConfig, CorrelationTrigger
from ovgenpy.wave import WaveConfig, Wave


RENDER_PROFILING = True


class Config(NamedTuple):
    wave_dir: str
    audio_path: Optional[str]
    fps: int
    time_visible_ms: int
    scan_ratio: float

    trigger: TriggerConfig  # Maybe overriden per Wave
    render: RendererConfig
    outputs: List[outputs.OutputConfig]
    create_window: bool

    @property
    def time_visible_s(self) -> float:
        return self.time_visible_ms / 1000


Folder = click.Path(exists=True, file_okay=False)
File = click.Path(exists=True, dir_okay=False)

_FPS = 60  # f_s


@click.command()
@click.argument('wave_dir', type=Folder)
@click.option('--audio_path', type=File, default=None)
@click.option('--fps', default=_FPS)
@click.option('--output', default='output.mp4')
def main(wave_dir: str, audio_path: Optional[str], fps: int, output: str):
    cfg = Config(
        wave_dir=wave_dir,
        audio_path=audio_path,
        fps=fps,
        time_visible_ms=25,
        scan_ratio=1,

        trigger=CorrelationTrigger.Config(
            trigger_strength=10,
            use_edge_trigger=True,

            responsiveness=0.1,
            falloff_width=.5,
        ),
        render=RendererConfig(     # todo
            1280, 720,
            ncols=1
        ),
        outputs=[
            outputs.FFmpegOutputConfig(output)
        ],
        create_window=True
    )

    ovgen = Ovgen(cfg)
    ovgen.write()


COLOR_CHANNELS = 3


class Ovgen:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.waves: List[Wave] = []
        self.nwaves: int = None
        self.outputs: List[outputs.Output] = []

    def write(self):
        self._load_waves()  # self.waves =
        self._load_outputs()  # self.outputs =
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
                scan_nsamp=round(
                    self.cfg.time_visible_s * self.cfg.scan_ratio * wave.smp_s),
                # I tried extracting variable, but got confused as a result
            )
            wave.set_trigger(trigger)
            self.waves.append(wave)

        self.nwaves = len(self.waves)

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

        renderer = MatplotlibRenderer(self.cfg.render, self.nwaves, create_window)

        if RENDER_PROFILING:
            begin = time.perf_counter()

        # For each frame, render each wave
        for frame in range(nframes):
            time_seconds = frame / fps

            datas = []
            # Get data from each wave
            for wave in self.waves:
                sample = round(wave.smp_s * time_seconds)
                region_len = round(wave.smp_s * time_visible_s)

                trigger_sample = wave.trigger.get_trigger(sample)
                print(f'- {trigger_sample}')

                datas.append(wave.get_around(trigger_sample, region_len))

            # Render frame
            print(frame)
            renderer.render_frame(datas)

            # Output frame
            frame = renderer.get_frame()

            # TODO write to file
            # how to write ndarray to ffmpeg?
            # idea: imageio.mimwrite(stdout, ... wait it's blocking = bad
            # idea: -f rawvideo, pass cfg.render.options... to ffmpeg_input_video()

        if RENDER_PROFILING:
            # noinspection PyUnboundLocalVariable
            dtime = time.perf_counter() - begin
            render_fps = nframes / dtime
            print(f'FPS = {render_fps}')
