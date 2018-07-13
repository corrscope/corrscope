import weakref
from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple

import click
import numpy as np
from scipy.io import wavfile

from ovgenpy.util import ceildiv


class Config(NamedTuple):
    wave_dir: str  # TODO remove, a function will expand wildcards and create List[WaveConfig]
    master_wave: Optional[str]

    fps: int
    # TODO algorithm and twiddle knobs

    render: 'RendererCfg'


class RendererCfg(NamedTuple):
    width: int
    height: int

    samples_visible: int

    rows_first: bool

    nrows: Optional[int] = None
    ncols: Optional[int] = None


Folder = click.Path(exists=True, file_okay=False)
File = click.Path(exists=True, dir_okay=False)

FPS = 60  # fps


@click.command()
@click.argument('wave_dir', type=Folder)
@click.option('--master-wave', type=File, default=None)
@click.option('--fps', default=FPS)
def main(wave_dir: str, master_wave: Optional[str], fps: int):
    cfg = Config(
        wave_dir=wave_dir,
        master_wave=master_wave,
        fps=fps,
        render=RendererCfg(  # todo
            640, 360,
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
            self.waves.append(wave)

    def render(self):
        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps
        nframes = fps * self.waves[0].get_s()
        nframes = int(nframes) + 1

        renderer = MatplotlibRenderer(self.cfg.render, self.waves)

        # For each frame, render each wave
        for frame in range(nframes):
            time_seconds = frame / fps

            center_smps = []
            for wave in self.waves:
                sample = round(wave.smp_s * time_seconds)
                trigger_sample = wave.trigger.get_trigger(sample)
                center_smps.append(trigger_sample)

            renderer.render_frame(center_smps)


class WaveConfig(NamedTuple):
    wave_path: str
    # TODO color


class Wave:
    def __init__(self, wcfg: WaveConfig, wave_path: str):
        self.cfg = wcfg
        self.smp_s, self.data = wavfile.read(wave_path)

        # FIXME cfg
        frames = 1
        self.trigger = Trigger(self, self.smp_s // FPS * frames, 0.1)

    def get_smp(self) -> int:
        return len(self.data)

    def get_s(self) -> float:
        """
        :return: time (seconds)
        """
        return self.get_smp() / self.smp_s


class Trigger:
    def __init__(self, wave: Wave, scan_nsamp: int, align_amount: float):
        """
        Correlation-based trigger which looks at a window of `scan_nsamp` samples.

        it's complicated

        :param wave: Wave file
        :param scan_nsamp: Number of samples used to align adjacent frames
        :param align_amount: Amount of centering to apply to each frame, within [0, 1]
        """

        # probably unnecessary
        self.wave = weakref.proxy(wave)
        self.scan_nsamp = scan_nsamp
        self.align_amount = align_amount

    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        return offset  # todo


class MatplotlibRenderer:
    def __init__(self, cfg: RendererCfg, waves: List[Wave]):
        self.cfg = cfg
        self.waves = waves

        self.dims: Tuple[int, int] = (0, 0)
        self.calc_layout()

        """
        If __init__ reads cfg, cfg cannot be hotswapped.
        
        Reasons to hotswap cfg: RendererCfg:
        - GUI preview size
        - Changing layout
        - Changing #smp drawn (samples_visible)
        (see RendererCfg)
            
            Original OVGen does not support hotswapping.
            It disables changing options during rendering.
        
        Reasons to hotswap trigger algorithms:
        - changing scan_nsamp (cannot be hotswapped, since correlation buffer is incompatible)
        So don't.
        """

    def calc_layout(self) -> None:
        """
        Inputs: self.cfg, self.waves
        Outputs: self.dims
        """
        cfg = self.cfg

        if cfg.rows_first:
            major = cfg.nrows
            if major is None:
                raise ValueError('invalid cfg: rows_first is True and nrows is None')
        else:
            major = cfg.ncols
            if major is None:
                raise ValueError('invalid cfg: rows_first is False and ncols is None')

        minor = ceildiv(len(self.waves), major)
        self.dims = (major, minor)

    def _get_coords(self, idx: int):
        major, minor = np.unravel_index(idx, self.dims)
        if self.cfg.rows_first:
            row, col = major, minor
        else:
            col, row = major, minor

        return Coords(row, col)

    def render_frame(self, center_smps: List[int]) -> None:
        nwaves = len(self.waves)
        ncenters = len(center_smps)
        if nwaves != ncenters:
            raise ValueError(
                f'incorrect number of wave offsets: {nwaves} waves but {ncenters} offsets')

        for wave, center_smp in zip(self.waves, center_smps):  # TODO
            print(wave)
            print(center_smp)
            print()


class Coords(NamedTuple):
    row: int
    col: int
