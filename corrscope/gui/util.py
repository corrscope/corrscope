import html
from itertools import groupby
from operator import itemgetter
from typing import TypeVar, Iterable, Generic, Tuple, Any, Optional

import matplotlib.colors
from qtpy.QtCore import QMutex, QMutexLocker
from qtpy.QtWidgets import QErrorMessage, QWidget

from corrscope.config import CorrError


def color2hex(color: Any) -> str:
    if color is None:
        return ""

    try:
        return matplotlib.colors.to_hex(color, keep_alpha=False)
    except ValueError:
        raise CorrError(f"invalid color {color}")
    except Exception as e:
        raise CorrError(f"doubly invalid color {color}, raises {e} (report bug!)")


T = TypeVar("T")


class Locked(Generic[T]):
    """Based off https://stackoverflow.com/a/37606669"""

    def __init__(self, obj: T):
        super().__init__()
        self.obj = obj
        self.lock = QMutex()

    def set(self, value: T) -> T:
        # We don't actually need a mutex since Python holds the GIL during reads and
        # writes. But keep it anyway.
        with QMutexLocker(self.lock):
            self.obj = value
        return value

    def get(self) -> T:
        with QMutexLocker(self.lock):
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

    def __init__(self, parent: Optional[QWidget] = None):
        QErrorMessage.__init__(self, parent)
        self.resize(self.w, self.h)

    def showMessage(self, message: str, type: Any = None) -> None:
        message = self.template % (html.escape(message))
        QErrorMessage.showMessage(self, message, type)


def find_ranges(iterable: Iterable[T]) -> Iterable[Tuple[T, int]]:
    """Extracts consecutive runs from a list of items.

    :param iterable: List of items.
    :return: Iterable of (first elem, length).
    """
    for group in consecutive_groups(iterable):
        group = list(group)
        yield group[0], len(group)


# Taken from more-itertools 4.3.0
def consecutive_groups(iterable, ordering=lambda x: x):
    """Yield groups of consecutive items using :func:`itertools.groupby`.
    The *ordering* function determines whether two items are adjacent by
    returning their position.

    By default, the ordering function is the identity function. This is
    suitable for finding runs of numbers:

        >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
        >>> for group in consecutive_groups(iterable):
        ...     print(list(group))
        [1]
        [10, 11, 12]
        [20]
        [30, 31, 32, 33]
        [40]

    For finding runs of adjacent letters, try using the :meth:`index` method
    of a string of letters:

        >>> from string import ascii_lowercase
        >>> iterable = 'abcdfgilmnop'
        >>> ordering = ascii_lowercase.index
        >>> for group in consecutive_groups(iterable, ordering):
        ...     print(list(group))
        ['a', 'b', 'c', 'd']
        ['f', 'g']
        ['i']
        ['l', 'm', 'n', 'o', 'p']

    """
    for k, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)
