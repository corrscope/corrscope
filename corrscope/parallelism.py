"""
- Parent sends messages to child.
- Child replies False to every message except final None.
- If child crashes, it sends True and terminates.

# Normal operation

frame 0:
- parent to child
- child reads
... child to parent

frame 1:
- parent reads
- parent to child
- child reads
... child to parent

# End of song

- parent reads
- parent to child "do not reply"
- child reads, and quits

# Child exception

- parent reads
- parent to child
- child reads
... child to parent "error", and quits

frame:
- parent reads and quits
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import *

import numpy as np

__all__ = ["Message", "ReplyIsAborted", "Worker", "ParallelWorker", "SerialWorker"]


class Error(Enum):
    Error = auto()


Message = TypeVar("Message")
MessageOrAbort = Union[Message, Error, None]


ReplyIsAborted = bool  # Is aborted?
ReplyOrError = Union[ReplyIsAborted, Error]


class Job(Generic[Message]):
    """ All methods (except __init__) *must* be called from thread. """

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def foreach(self, msg: Message) -> ReplyIsAborted:
        """
        False: not aborted
        True: aborted without error (ffplay closed).
        raise: error.

        do *not* return Error.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Worker(Generic[Message]):
    @abstractmethod
    def __init__(self, child_job: Job[Message], profile_name: Optional[str]) -> None:
        pass

    def __enter__(self) -> "Worker[Message]":
        return self

    @abstractmethod
    def parent_send(self, msg: MessageOrAbort) -> ReplyIsAborted:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# noinspection PyAttributeOutsideInit
class ParallelWorker(Worker[Message]):
    _parent_conn: Connection

    def __init__(self, child_job: Job[Message], profile_name: Optional[str]) -> None:
        super().__init__(child_job, profile_name)
        self._parent_conn, child_conn = Pipe(duplex=True)

        command = lambda: self._child_thread_func(child_job, child_conn)
        if profile_name:
            import cProfile

            g = globals()
            l = locals()
            target = lambda: cProfile.runctx("command()", g, l, profile_name)

        else:
            target = command

        self._child_thread = Process(target=target)

    def __enter__(self) -> "ParallelWorker[Message]":
        """ Ensure this class cannot be called without a with statement. """
        self._child_thread.start()

        self._not_first = False
        self._child_dead = False
        return self

    def parent_send(self, msg: MessageOrAbort) -> ReplyIsAborted:
        """ Checks for exceptions, then sends a message to the child process. """
        try:
            if self._child_dead:
                raise ValueError("cannot send to dead worker")
        except AttributeError:
            raise ValueError(
                "Must use `with` clause (__enter__) before calling parent_send"
            )

        # Receive reply from child.
        if self._not_first:
            is_aborted_or_error = self._parent_conn.recv()  # type: ReplyOrError
            if is_aborted_or_error is not False:
                self._child_dead = True

            # https://github.com/python/mypy/issues/1803
            if isinstance(is_aborted_or_error, Error):
                exit(1)  # Stack trace is printed by child process.
            elif is_aborted_or_error:
                return True
            else:
                pass

        # Send parent message.
        self._parent_conn.send(msg)
        if msg is None or isinstance(msg, Error):
            self._child_dead = True

        self._not_first = True

        return False

    @staticmethod
    def _child_thread_func(child_job: Job, child_conn: Connection):
        """ Must be called from thread. """
        with child_job:
            while True:
                # Receive parent message.
                msg: MessageOrAbort = child_conn.recv()
                if msg is None:
                    break  # See module docstring

                elif isinstance(msg, Error):
                    raise SystemExit

                # Reply to parent (and optionally terminate).
                try:
                    reply = child_job.foreach(msg)
                except BaseException:
                    child_conn.send(Error.Error)
                    raise
                else:
                    child_conn.send(reply)
                    if reply:
                        break

    def __exit__(self, exc_type, exc_val: Optional[BaseException], exc_tb):
        if not self._child_dead:
            if exc_val:
                self.parent_send(Error.Error)
            else:
                self.parent_send(None)

        # parent_send(BaseException or None) sets _child_dead = True.

        self._parent_conn.close()
        self._child_thread.join(1)
        if self._child_thread.exitcode is None:
            self._child_thread.terminate()
            raise ValueError("renderer thread failed to terminate after 1 second")


class SerialWorker(Worker[Message]):
    def __init__(self, child_job: Job[Message], profile_name: Optional[str]) -> None:
        super().__init__(child_job, profile_name)
        self.child_job = child_job

    def __enter__(self) -> "SerialWorker[Message]":
        self.child_job.__enter__()
        return self

    def parent_send(self, msg: MessageOrAbort) -> ReplyIsAborted:
        if msg is None:
            return False
        return self.child_job.foreach(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.child_job.__exit__(exc_type, exc_val, exc_tb)
