import threading
from collections import OrderedDict


class _Call:
    def __init__(self):
        self.done = threading.Event()
        self.value = None
        self.exc = None


class LRUCache:
    def __init__(self, capacity):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be an int >= 1")
        self._cap = capacity
        self._data = OrderedDict()
        self._inflight = {}
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return len(self._data)

    def keys(self):
        with self._lock:
            return list(self._data)

    def get(self, key, default=None):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return default

    def _store(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._cap:
            self._data.popitem(last=False)

    def put(self, key, value):
        with self._lock:
            self._store(key, value)

    def get_or_compute(self, key, fn):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            call = self._inflight.get(key)
            owner = call is None
            if owner:
                call = self._inflight[key] = _Call()
        if not owner:
            call.done.wait()
            if call.exc is not None:
                raise call.exc
            return call.value
        try:
            value = fn(key)
        except BaseException as e:
            with self._lock:
                del self._inflight[key]
            call.exc = e
            call.done.set()
            raise
        with self._lock:
            self._store(key, value)
            del self._inflight[key]
        call.value = value
        call.done.set()
        return value
