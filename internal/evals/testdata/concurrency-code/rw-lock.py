import threading
from contextlib import contextmanager


class RWLock:
    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0          # threads holding the read lock
        self._writing = False      # a writer holds the lock
        self._writers_waiting = 0

    @contextmanager
    def read_lock(self):
        with self._cond:
            while self._writing or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writing or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writing = True
        try:
            yield
        finally:
            with self._cond:
                self._writing = False
                self._cond.notify_all()
