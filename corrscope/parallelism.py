from typing import List, Iterator, TYPE_CHECKING, Callable, Union

import numpy as np

if TYPE_CHECKING:
    from multiprocessing.connection import Connection


Message = Union[List[np.ndarray], None]
ReplyMessage = bool  # Has exception occurred?


# Parent
def connection_host(conn: "Connection") -> Callable[[Message], None]:
    """ Checks for exceptions, then sends a message to the child process. """
    not_first = False
    # If child process is dead, do not `finally` send None.
    dead = False

    def send(obj: Message) -> None:
        nonlocal not_first, dead
        if dead:
            return

        if not_first:
            is_child_exc = conn.recv()  # type: ReplyMessage
            if is_child_exc:
                dead = True
                exit(1)

        conn.send(obj)
        not_first = True

    return send


# Child
def iter_conn(conn: "Connection") -> Iterator[Message]:
    """ Yields elements of a threading queue, stops on None. """
    while True:
        item = conn.recv()
        if item is None:
            break
        yield item
