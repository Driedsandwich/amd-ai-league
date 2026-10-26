"""Nonblocking lifetime locks shared by the CLI, supervisor, and server."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path


@contextmanager
def process_lock(sdir: str, name: str, inherited_fd=None):
    Path(sdir).mkdir(parents=True, exist_ok=True)
    lock = (open(Path(sdir) / f"{name}.lock", "a+b") if inherited_fd is None
            else os.fdopen(inherited_fd, "a+b"))
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        os.set_inheritable(lock.fileno(), False)
        yield lock
    finally:
        # Close only: LOCK_UN would also unlock a descriptor passed to the child.
        lock.close()
