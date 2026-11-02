import threading
import time


class TokenBucket:
    def __init__(self, rate, capacity, clock=time.monotonic):
        for v in (rate, capacity):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                raise ValueError("rate and capacity must be numbers > 0")
        self._rate = float(rate)
        self._cap = float(capacity)
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens = self._cap
        self._last = clock()

    def _check(self, n):
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not 0 < n <= self._cap:
            raise ValueError("n must satisfy 0 < n <= capacity")

    def _refill(self):
        now = self._clock()
        if now > self._last:
            self._tokens = min(self._cap, self._tokens + (now - self._last) * self._rate)
            self._last = now

    def try_acquire(self, n=1):
        self._check(n)
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def available(self):
        with self._lock:
            self._refill()
            return float(self._tokens)

    def wait_time(self, n=1):
        self._check(n)
        with self._lock:
            self._refill()
            missing = n - self._tokens
            return 0.0 if missing <= 0 else missing / self._rate
