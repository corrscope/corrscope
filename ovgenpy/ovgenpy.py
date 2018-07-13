# -*- coding: utf-8 -*-

import weakref
from abc import ABC, abstractmethod
from itertools import count
from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple, Dict, Any
import time

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.io import wavfile

from ovgenpy.util import ceildiv


class Config(NamedTuple):
    wave_dir: str
    # TODO: if wave_dir is present, it should overwrite List[WaveConfig].
    # wave_dir will be commented out when writing to file.

    master_wave: Optional[str]

    fps: int

    trigger: 'TriggerConfig'  # Maybe overriden per Wave
    render: 'RendererConfig'


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
            kwargs=dict(
                align_amount=0.1    # TODO: default param?
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
    PROFILING = True

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
                scan_nsamp=wave.smp_s // self.cfg.fps,
            ))
            self.waves.append(wave)

    def render(self):
        # Calculate number of frames (TODO master file?)
        fps = self.cfg.fps
        nframes = fps * self.waves[0].get_s()
        nframes = int(nframes) + 1

        renderer = MatplotlibRenderer(self.cfg.render, self.waves)

        if self.PROFILING:
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

        if self.PROFILING:
            dtime = time.perf_counter() - begin
            render_fps = nframes / dtime
            print(f'FPS = {render_fps}')


class WaveConfig(NamedTuple):
    wave_path: str
    # TODO color

    # TODO wave-specific trigger options?


FLOAT = np.double
class Wave:
    def __init__(self, wcfg: WaveConfig, wave_path: str):
        self.cfg = wcfg
        # TODO mmap
        self.smp_s, self.data = wavfile.read(wave_path)     # type: int, np.ndarray
        self.nsamp = len(self.data)
        self.trigger: Trigger = None

        # Calculate scaling factor.
        # TODO extract function, unit tests... switch to pysoundfile and drop logic
        dtype = self.data.dtype

        max_val = np.iinfo(dtype).max + 1
        assert max_val & (max_val - 1) == 0     # power of 2

        if np.issubdtype(dtype, np.uint):
            self.offset = -max_val // 2
            self.max_val = max_val // 2
        else:
            self.offset = 0
            self.max_val = max_val

    def __getitem__(self, index):
        """ Convert self.data[item] to a FLOAT within range [-1, 1). """
        data = self.data[index].astype(FLOAT)
        data += self.offset
        data /= self.max_val
        return data

    def get_around(self, sample: int, region_nsamp: int):
        end = sample + region_nsamp // 2
        begin = end - region_nsamp

        if 0 <= begin and end <= self.nsamp:
            return self[begin:end]

        delta_begin = 0
        if begin < 0:
            delta_begin = 0 - begin
            assert delta_begin > 0      # TODO really not necessary unless I distrust myself
            assert begin + delta_begin == 0
            # begin += delta_begin

        delta_end = 0
        if end > self.nsamp:
            delta_end = self.nsamp - end
            assert delta_end < 0
            assert end + delta_end == self.nsamp
            # end += delta_end

        out = np.zeros(region_nsamp, dtype=FLOAT)

        # out[0 : region_nsamp]. == self[begin: end]
        # out[Δbegin : region_nsamp+Δend] == self[begin + Δbegin: end + Δend]
        out[delta_begin : region_nsamp+delta_end] = self[begin+delta_begin : end+delta_end]
        return out






    def set_trigger(self, trigger: 'Trigger'):
        self.trigger = trigger

    def get_smp(self) -> int:
        return len(self.data)

    def get_s(self) -> float:
        """
        :return: time (seconds)
        """
        return self.get_smp() / self.smp_s


class TriggerConfig(NamedTuple):
    name: str
    # scan_nsamp: int
    args: List = []
    kwargs: Dict[str, Any] = {}

    def generate_trigger(self, wave: Wave, scan_nsamp: int) -> 'Trigger':
        return TRIGGERS[self.name](wave, scan_nsamp, *self.args, **self.kwargs)


TRIGGERS: Dict[str, type] = {}

def register_trigger(trigger_class: type):
    TRIGGERS[trigger_class.__name__] = trigger_class
    return trigger_class


class Trigger(ABC):
    @abstractmethod
    def get_trigger(self, offset: int) -> int:
        """
        :param offset: sample index
        :return: new sample index, corresponding to rising edge
        """
        ...


@register_trigger
class CorrelationTrigger(Trigger):
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


class RendererConfig(NamedTuple):
    width: int
    height: int

    samples_visible: int

    rows_first: bool

    nrows: Optional[int] = None
    ncols: Optional[int] = None

    # TODO backend: FigureCanvasBase = FigureCanvasAgg


class MatplotlibRenderer:
    """
    TODO disable antialiasing
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

    DPI = 96

    def __init__(self, cfg: RendererConfig, waves: List[Wave]):
        self.cfg = cfg
        self.waves = waves
        self.fig: Figure = None

        # Setup layout

        self.nrows = 0
        self.ncols = 0
        # Flat array of nrows*ncols elements, ordered by cfg.rows_first.
        self.axes: List[Axes] = None
        self.lines: List[Line2D] = None

        self.set_layout()   # mutates self

    def set_layout(self) -> None:
        """
        Inputs: self.cfg, self.waves, self.fig
        Outputs: self.nrows, self.ncols, self.axes

        Creates a flat array of Matplotlib Axes, with the new layout.
        """

        self.nrows, self.ncols = self.calc_layout()

        # Create Axes
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
        if self.fig:
            plt.close(self.fig)     # FIXME

        axes2d: np.ndarray[Axes]
        self.fig, axes2d = plt.subplots(
            self.nrows, self.ncols,
            squeeze=False,
            # Remove gaps between Axes
            gridspec_kw=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        )

        # remove Axis from Axes
        for ax in axes2d.flatten():
            ax.set_axis_off()

        # if column major:
        if not self.cfg.rows_first:
            axes2d = axes2d.T

        nwave = len(self.waves)
        self.axes: List[Axes] = axes2d.flatten().tolist()[:nwave]

        # Create oscilloscope line objects
        self.lines = []
        for ax in self.axes:
            # Setup axes limits
            ax.set_xlim(0, self.cfg.samples_visible)
            ax.set_ylim(-1, 1)

            line = ax.plot([0] * self.cfg.samples_visible)[0]
            self.lines.append(line)

        # Setup figure geometry
        self.fig.set_dpi(self.DPI)
        self.fig.set_size_inches(
            self.cfg.width / self.DPI,
            self.cfg.height / self.DPI
        )
        plt.show(block=False)

    def calc_layout(self) -> Tuple[int, int]:
        """
        Inputs: self.cfg, self.waves
        :return: (nrows, ncols)
        """
        cfg = self.cfg
        nwaves = len(self.waves)

        if cfg.rows_first:
            nrows = cfg.nrows
            if nrows is None:
                raise ValueError('invalid cfg: rows_first is True and nrows is None')
            ncols = ceildiv(nwaves, nrows)
        else:
            ncols = cfg.ncols
            if ncols is None:
                raise ValueError('invalid cfg: rows_first is False and ncols is None')
            nrows = ceildiv(nwaves, ncols)

        return nrows, ncols

    def render_frame(self, center_smps: List[int]) -> None:
        nwaves = len(self.waves)
        ncenters = len(center_smps)
        if nwaves != ncenters:
            raise ValueError(
                f'incorrect wave offsets: {nwaves} waves but {ncenters} offsets')

        for idx, wave, center_smp in zip(count(), self.waves, center_smps):
            # Draw waveform data
            line = self.lines[idx]
            data = wave.get_around(center_smp, self.cfg.samples_visible)
            line.set_ydata(data)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
