from typing import Optional, TypeVar, Callable, List

import numpy as np

from corrscope.config import register_config, CorrError
from corrscope.util import ceildiv


@register_config(always_dump='orientation')
class LayoutConfig:
    orientation: str = 'h'
    nrows: Optional[int] = None
    ncols: Optional[int] = None

    def __attrs_post_init__(self):
        if not self.nrows:
            self.nrows = None
        if not self.ncols:
            self.ncols = None

        if self.nrows and self.ncols:
            raise CorrError('cannot manually assign both nrows and ncols')

        if not self.nrows and not self.ncols:
            self.ncols = 1


Region = TypeVar('Region')
RegionFactory = Callable[[int, int], Region]   # f(row, column) -> Region


class RendererLayout:
    VALID_ORIENTATIONS = ['h', 'v']

    def __init__(self, cfg: LayoutConfig, nplots: int):
        self.cfg = cfg
        self.nplots = nplots

        # Setup layout
        self.nrows, self.ncols = self._calc_layout()

        self.orientation = cfg.orientation
        if self.orientation not in self.VALID_ORIENTATIONS:
            raise CorrError(f'Invalid orientation {self.orientation} not in '
                             f'{self.VALID_ORIENTATIONS}')

    def _calc_layout(self):
        """
        Inputs: self.cfg, self.waves
        :return: (nrows, ncols)
        """
        cfg = self.cfg

        if cfg.nrows:
            nrows = cfg.nrows
            if nrows is None:
                raise ValueError('impossible cfg: nrows is None and true')
            ncols = ceildiv(self.nplots, nrows)
        else:
            ncols = cfg.ncols
            if ncols is None:
                raise ValueError('invalid LayoutConfig: nrows,ncols is None '
                                 '(__attrs_post_init__ not called?)')
            nrows = ceildiv(self.nplots, ncols)

        return nrows, ncols

    def arrange(self, region_factory: RegionFactory) -> List[Region]:
        """ Generates an array of regions.

        index, row, column are fed into region_factory in a row-major order [row][col].
        The results are possibly reshaped into column-major order [col][row].
        """
        nspaces = self.nrows * self.ncols
        inds = np.arange(nspaces)
        rows, cols = np.unravel_index(inds, (self.nrows, self.ncols))

        row_col = list(zip(rows, cols))
        regions = np.empty(len(row_col), dtype=object)          # type: np.ndarray[Region]
        regions[:] = [region_factory(*rc) for rc in row_col]

        regions2d = regions.reshape((self.nrows, self.ncols))   # type: np.ndarray[Region]

        # if column major:
        if self.orientation == 'v':
            regions2d = regions2d.T

        return regions2d.flatten()[:self.nplots].tolist()
