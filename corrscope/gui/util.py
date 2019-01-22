import html
from typing import TypeVar, Iterable, Generic, Tuple

import matplotlib.colors
import more_itertools
from PyQt5.QtCore import QMutex
from PyQt5.QtWidgets import QErrorMessage

from corrscope.config import CorrError


def color2hex(color):
    try:
        return matplotlib.colors.to_hex(color, keep_alpha=False)
    except ValueError:
        raise CorrError(f"invalid color {color}")
    except Exception as e:
        raise CorrError(f"doubly invalid color {color}, raises {e} (report bug!)")


T = TypeVar("T")


class Locked(Generic[T]):
    """ Based off https://stackoverflow.com/a/37606669 """

    def __init__(self, obj: T):
        super().__init__()
        self.obj = obj
        self.lock = QMutex(QMutex.Recursive)

    def __enter__(self) -> T:
        self.lock.lock()
        return self.obj

    def __exit__(self, *args, **kwargs):
        self.lock.unlock()

    def set(self, value: T) -> T:
        with self:
            self.obj = value
        return value

    def get(self) -> T:
        with self:
            return self.obj


class TracebackDialog(QErrorMessage):
    w = 640
    h = 360
    template = """\
    <style>
    body {
        white-space: pre-wrap;
    }
    </style>
    <body>%s</body>"""

    def __init__(self, parent=None):
        QErrorMessage.__init__(self, parent)
        self.resize(self.w, self.h)

    def showMessage(self, message, type=None):
        message = self.template % (html.escape(message))
        QErrorMessage.showMessage(self, message, type)


def find_ranges(iterable: Iterable[T]) -> Iterable[Tuple[T, int]]:
    """Extracts consecutive runs from a list of items.

    :param iterable: List of items.
    :return: Iterable of (first elem, length).
    """
    for group in more_itertools.consecutive_groups(iterable):
        group = list(group)
        yield group[0], len(group)
