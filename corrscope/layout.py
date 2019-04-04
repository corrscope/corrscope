import collections
import enum
from enum import auto
from typing import Optional, TypeVar, Callable, List, Iterable

import attr
import numpy as np

from corrscope.config import DumpableAttrs, CorrError, DumpEnumAsStr
from corrscope.util import ceildiv


class Orientation(str, DumpEnumAsStr):
    h = "h"
    v = "v"


class StereoOrientation(str, DumpEnumAsStr):
    h = "h"
    v = "v"
    overlay = "overlay"


assert Orientation.h == StereoOrientation.h
H = Orientation.h
V = Orientation.v
OVERLAY = StereoOrientation.overlay


class LayoutConfig(DumpableAttrs, always_dump="orientation stereo_orientation"):
    orientation: Orientation = attr.ib(default="h", converter=Orientation)
    nrows: Optional[int] = None
    ncols: Optional[int] = None

    stereo_orientation: StereoOrientation = attr.ib(
        default="h", converter=StereoOrientation
    )

    def __attrs_post_init__(self) -> None:
        if not self.nrows:
            self.nrows = None
        if not self.ncols:
            self.ncols = None

        if self.nrows and self.ncols:
            raise CorrError("cannot manually assign both nrows and ncols")

        if not self.nrows and not self.ncols:
            self.ncols = 1


class Edges(enum.Flag):
    NONE = 0
    Top = auto()
    Left = auto()
    Bottom = auto()
    Right = auto()

    @staticmethod
    def at(nrows: int, ncols: int, row: int, col: int):
        if not nrows > 0:
            raise ValueError(f"invalid nrows={nrows}, must be positive")
        if not ncols > 0:
            raise ValueError(f"invalid ncols={ncols}, must be positive")
        if not 0 <= row < nrows:
            raise ValueError(f"invalid row={row} not in [0 .. nrows={nrows})")
        if not 0 <= col < ncols:
            raise ValueError(f"invalid col={col} not in [0 .. ncols={ncols})")

        ret = Edges.NONE
        if row == 0:
            ret |= Edges.Top
        if row + 1 == nrows:
            ret |= Edges.Bottom
        if col == 0:
            ret |= Edges.Left
        if col + 1 == ncols:
            ret |= Edges.Right
        return ret


def attr_idx_property(key: str, idx: int) -> property:
    @property
    def inner(self: "RegionSpec"):
        return getattr(self, key)[idx]

    return inner


@attr.dataclass
class RegionSpec:
    """
    - Origin is located at top-left.
    - Row 0 = top.
        - Row nrows-1 = bottom.
    - Col 0 = left.
        - Col ncols-1 = right.
    """

    size: np.ndarray
    pos: np.ndarray

    nrow = attr_idx_property("size", 0)
    ncol = attr_idx_property("size", 1)
    row = attr_idx_property("pos", 0)
    col = attr_idx_property("pos", 1)

    screen_edges: "Edges"
    wave_edges: "Edges"


Region = TypeVar("Region")
RegionFactory = Callable[[RegionSpec], Region]  # f(row, column) -> Region


class RendererLayout:
    def __init__(self, cfg: LayoutConfig, wave_nchans: List[int]):
        self.cfg = cfg
        self.nwaves = len(wave_nchans)
        self.wave_nchans = wave_nchans

        self.orientation = cfg.orientation
        self.stereo_orientation = cfg.stereo_orientation

        # Setup layout
        self._calc_layout()

    # Shape of wave slots
    wave_nrow: int
    wave_ncol: int

    def _calc_layout(self) -> None:
        """
        Inputs: self.cfg, self.stereo_nchan
        Outputs: self.wave_nrow, ncol
        """
        cfg = self.cfg

        if cfg.nrows:
            nrows = cfg.nrows
            if nrows is None:
                raise ValueError("impossible cfg: nrows is None and true")
            ncols = ceildiv(self.nwaves, nrows)
        else:
            if cfg.ncols is None:
                raise ValueError(
                    "invalid LayoutConfig: nrows,ncols is None "
                    "(__attrs_post_init__ not called?)"
                )
            ncols = cfg.ncols
            nrows = ceildiv(self.nwaves, ncols)

        self.wave_nrow = nrows
        self.wave_ncol = ncols

    def arrange(
        self, region_factory: RegionFactory[Region], **kwargs
    ) -> List[List[Region]]:
        """
        (row, column) are fed into region_factory in a row-major order [row][col].
        Stereo channel pairs are extracted.
        The results are possibly reshaped into column-major order [col][row].

        **kwargs -> region_factory(**kwargs).

        :return arr[wave][channel] = Region
        """

        wave_spaces = self.wave_nrow * self.wave_ncol
        inds = np.arange(wave_spaces)

        # Compute location of each wave.
        if self.orientation == V:
            # column major
            cols, rows = np.unravel_index(inds, (self.wave_ncol, self.wave_nrow))
        else:
            # row major
            rows, cols = np.unravel_index(inds, (self.wave_nrow, self.wave_ncol))

        # Generate plot for each wave.chan. Leave unused slots empty.
        region_wave_chan: List[List[Region]] = []

        # The order of (rows, cols) has no effect.
        for stereo_nchan, wave_row, wave_col in zip(self.wave_nchans, rows, cols):
            # Wave = within Screen.
            # Chan = within Wave, generate a plot.
            # All arrays are [y, x] == [row, col].

            # Wave dim/pos (within screen)
            waves_per_screen = arr(self.wave_nrow, self.wave_ncol)
            wave_screen_pos = arr(wave_row, wave_col)
            del wave_row, wave_col

            # Channel dim/pos (within wave)
            chans_per_wave = arr(1, 1)  # Mutated based on orientation
            chan_wave_pos = arr(0, 0)  # Mutated in for-chan loop.

            # Distance between chans
            dchan = arr(0, 0)  # Mutated based on orientation

            if self.stereo_orientation == V:
                chans_per_wave[0] = stereo_nchan
                dchan[0] = 1
            elif self.stereo_orientation == H:
                chans_per_wave[1] = stereo_nchan
                dchan[1] = 1
            else:
                assert self.stereo_orientation == OVERLAY

            # Channel dim/pos (within screen)
            chans_per_screen = chans_per_wave * waves_per_screen
            chan_screen_pos = chans_per_wave * wave_screen_pos

            # Generate plots for each channel
            region_chan: List[Region] = []
            region_wave_chan.append(region_chan)
            region = None

            for chan in range(stereo_nchan):
                assert (chan_wave_pos < chans_per_wave).all()
                assert (chan_screen_pos < chans_per_screen).all()

                # Generate plot (channel position in screen)
                if region is None or dchan.any():
                    screen_edges = Edges.at(*chans_per_screen, *chan_screen_pos)
                    wave_edges = Edges.at(*chans_per_wave, *chan_wave_pos)

                    # Removing .copy() causes bugs if region_factory() holds
                    # mutable references.
                    chan_spec = RegionSpec(
                        chans_per_screen.copy(),
                        chan_screen_pos.copy(),
                        screen_edges,
                        wave_edges,
                    )
                    region = region_factory(chan_spec, **kwargs)
                region_chan.append(region)

                # Move to next channel position
                chan_screen_pos += dchan
                chan_wave_pos += dchan

        assert len(region_wave_chan) == self.nwaves
        return region_wave_chan


def arr(*args):
    return np.array(args)


T = TypeVar("T")


def unique_by_id(items: Iterable[T]) -> List[T]:
    seen = collections.OrderedDict()

    for item in items:
        if id(item) not in seen:
            seen[id(item)] = item

    return list(seen.values())
