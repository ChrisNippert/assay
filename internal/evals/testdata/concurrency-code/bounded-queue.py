import threading
import time
from collections import deque


class QueueClosed(Exception):
    pass


class BoundedQueue:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._items = deque()
        self._closed = False
        lock = threading.Lock()
        self._not_empty = threading.Condition(lock)
        self._not_full = threading.Condition(lock)

    def __len__(self):
        with self._not_empty:
            return len(self._items)

    def _wait(self, cond, deadline):
        if deadline is None:
            cond.wait()
            return
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("timed out")
        cond.wait(left)

    def put(self, item, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_full:
            while True:
                if self._closed:
                    raise QueueClosed()
                if len(self._items) < self._cap:
                    break
                self._wait(self._not_full, deadline)
            self._items.append(item)
            self._not_empty.notify()

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while not self._items:
                if self._closed:
                    raise QueueClosed()
                self._wait(self._not_empty, deadline)
            item = self._items.popleft()
            self._not_full.notify()
            return item

    def close(self):
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()
