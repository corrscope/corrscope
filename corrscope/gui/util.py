from pathlib import Path
from typing import *
from typing import Iterable, Tuple

import matplotlib.colors
import more_itertools
from PyQt5.QtCore import QMutex
from PyQt5.QtWidgets import QWidget, QFileDialog

from corrscope.config import CorrError


def color2hex(color):
    try:
        return matplotlib.colors.to_hex(color, keep_alpha=False)
    except ValueError:
        raise CorrError(f'invalid color {color}')
    except Exception as e:
        raise CorrError(
            f'doubly invalid color {color}, raises {e} (report bug!)')


T = TypeVar('T')


class Locked(Generic[T]):
    """ Based off https://stackoverflow.com/a/37606669 """

    def __init__(self, obj: T):
        super().__init__()
        self.obj = obj
        self.lock = QMutex(QMutex.Recursive)
        self.skip_exit = False

    def __enter__(self) -> T:
        self.lock.lock()
        return self.obj

    def unlock(self):
        # FIXME does it work? i was not thinking clearly when i wrote this
        if not self.skip_exit:
            self.skip_exit = True
            self.lock.unlock()

    def __exit__(self, *args, **kwargs):
        if self.skip_exit:
            self.skip_exit = False
        else:
            self.lock.unlock()

    def set(self, value: T) -> T:
        with self:
            self.obj = value
        return value


def get_save_with_ext(
        parent: QWidget, caption: str, dir_or_file: str,
        filters: List[str], default_suffix: str
) -> Optional[Path]:
    """ On KDE, getSaveFileName does not append extension. This is a workaround. """
    name, sel_filter = QFileDialog.getSaveFileName(
        parent, caption, dir_or_file, ';;'.join(filters)
    )

    if name == '':
        return None

    path = Path(name)
    if sel_filter == filters[0] and path.suffix == '':
        path = path.with_suffix(default_suffix)
    return path


def find_ranges(iterable: Iterable[T]) -> Iterable[Tuple[T, int]]:
    """Extracts consecutive runs from a list of items.

    :param iterable: List of items.
    :return: Iterable of (first elem, length).
    """
    for group in more_itertools.consecutive_groups(iterable):
        group = list(group)
        yield group[0], len(group)
