from pathlib import Path
from typing import *

import matplotlib.colors
from PyQt5.QtCore import QMutex
from PyQt5.QtWidgets import QWidget, QFileDialog


def color2hex(color):
    return matplotlib.colors.to_hex(color, keep_alpha=False)


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

