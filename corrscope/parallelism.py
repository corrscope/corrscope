from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import List, Iterator, Optional

import numpy as np

__all__ = [
    "Message",
    "MaybeMessage",
    "ReplyMessage",
    # "Worker",
    "ParallelWorker",
    # "SerialWorker",
]


Message = List[np.ndarray]
MaybeMessage = Optional[Message]
ReplyMessage = bool  # Has exception occurred?


class Job(ABC):
    """ All methods (except __init__) *must* be called from thread. """

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def foreach(self, msg: Message) -> ReplyMessage:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# class Worker(ABC):
#     @abstractmethod
#     def __init__(self, child_job: "Job"):
#         # self.child_job = child_job
#         pass
#
#     def __enter__(self):
#         pass
#
#     @abstractmethod
#     def parent_send(self, msg: Message) -> None:
#         pass
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         pass


# noinspection PyAttributeOutsideInit
class ParallelWorker:
    _parent_conn: Connection

    def __init__(self, child_job: Job, name: str = None):
        self._parent_conn, child_conn = Pipe(duplex=True)

        self._child_thread = Process(
            name=name, target=self._child_thread_func, args=[child_job, child_conn]
        )

    @staticmethod
    def _child_thread_func(child_job: Job, child_conn: Connection):
        """ Must be called from thread. """
        with child_job:
            for msg in iter_conn(child_conn):
                try:
                    reply = child_job.foreach(msg)
                except BaseException as e:
                    child_conn.send(True)
                    raise
                else:
                    child_conn.send(reply)

    def __enter__(self):
        """ Ensure this class cannot be called without a with statement. """
        self._child_thread.start()

        self._not_first = False
        self._child_dead = False
        return self

    def parent_send(self, msg: Message) -> None:
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

        self._parent_conn.send(msg)
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
