import threading
import time


class InvalidStateError(Exception):
    pass


class Future:
    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._done = False
        self._value = None
        self._exc = None
        self._callbacks = []

    def _complete(self, value, exc):
        with self._cond:
            if self._done:
                raise InvalidStateError("future is already complete")
            self._value, self._exc, self._done = value, exc, True
            callbacks, self._callbacks = self._callbacks, []
            self._cond.notify_all()
        for fn in callbacks:  # outside the lock
            self._run(fn)

    def set_result(self, value):
        self._complete(value, None)

    def set_exception(self, exc):
        self._complete(None, exc)

    def _run(self, fn):
        try:
            fn(self)
        except Exception:
            pass

    def done(self):
        with self._cond:
            return self._done

    def _wait(self, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._done:
                if deadline is None:
                    self._cond.wait()
                else:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError("future not complete")
                    self._cond.wait(left)

    def result(self, timeout=None):
        self._wait(timeout)
        if self._exc is not None:
            raise self._exc
        return self._value

    def exception(self, timeout=None):
        self._wait(timeout)
        return self._exc

    def add_done_callback(self, fn):
        with self._cond:
            if not self._done:
                self._callbacks.append(fn)
                return
        self._run(fn)
