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
        self.skip_exit = False

    def __enter__(self) -> T:
        self.lock.lock()
        return self.__obj

    def unlock(self):
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
            self.__obj = value
        return value
