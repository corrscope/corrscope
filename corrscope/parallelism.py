from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import List, Iterator, Callable, Union

import numpy as np

__all__ = [
    "Message",
    "ReplyMessage",
    "Worker",
    "ParallelWorker",
    # "SerialWorker",
    "iter_conn",
]


Message = Union[List[np.ndarray], None]
ReplyMessage = bool  # Has exception occurred?


class Worker(ABC):
    @abstractmethod
    def __init__(self, child_func: Callable[[Connection], None]):
        pass

    def __enter__(self):
        pass

    @abstractmethod
    def parent_send(self, obj: Message) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# noinspection PyAttributeOutsideInit
class ParallelWorker(Worker):
    _parent_conn: Connection

    def __init__(self, child_func: Callable[[Connection], None], name: str = None):
        """ To pass parameters to child_func, wrap it in a closure.
        This allows for type-checking. """
        super().__init__(child_func)

        _child_conn: Connection
        self._parent_conn, _child_conn = Pipe(duplex=True)

        self._child_thread = Process(name=name, target=child_func, args=[_child_conn])
        self._child_thread.start()

    def __enter__(self):
        """ Ensure this class cannot be called without a with statement. """
        self._not_first = False
        self._child_dead = False
        return self

    def parent_send(self, obj: Message) -> None:
        """ Checks for exceptions, then sends a message to the child process. """

        # If child process is dead, do not `finally` send None.
        try:
            if self._child_dead:
                return
        except AttributeError:
            raise ValueError(
                "Must use `with` clause (__enter__) before calling parent_send"
            )

        if self._not_first:
            is_child_exc = self._parent_conn.recv()  # type: ReplyMessage
            if is_child_exc:
                self._child_dead = True
                exit(1)

        self._parent_conn.send(obj)
        self._not_first = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent_send(None)
        self._parent_conn.close()
        self._child_thread.join()


# class SerialWorker(Worker):
#     pass


# Parent


# Child
def iter_conn(conn: "Connection") -> Iterator[Message]:
    """ Yields elements of a threading queue, stops on None. """
    while True:
        item = conn.recv()
        if item is None:
            break
        yield item
