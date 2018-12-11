from typing import *
from PyQt5.QtCore import QMutex
import matplotlib.colors


def color2hex(color):
    return matplotlib.colors.to_hex(color, keep_alpha=False)


T = TypeVar('T')


class Locked(Generic[T]):
    """ Based off https://stackoverflow.com/a/37606669 """
    def __init__(self, obj: T):
        super().__init__()
        self.__obj = obj
        self.lock = QMutex(QMutex.Recursive)

    def __enter__(self) -> T:
        self.lock.lock()
        return self.__obj

    def __exit__(self, *args, **kwargs):
        self.lock.unlock()

    def set(self, value: T) -> T:
        with self:
            self.__obj = value
        return value
