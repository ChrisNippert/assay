import threading


class CyclicBarrier:
    def __init__(self, parties, action=None):
        if not isinstance(parties, int) or parties < 1:
            raise ValueError("parties must be an int >= 1")
        self._parties = parties
        self._action = action
        self._cond = threading.Condition(threading.Lock())
        self._count = 0
        self._generation = 0

    @property
    def parties(self):
        return self._parties

    def wait(self):
        with self._cond:
            gen = self._generation
            index = self._count
            self._count += 1
            if self._count == self._parties:
                # last to arrive: run the action while everyone else is still held
                if self._action is not None:
                    self._action()
                self._count = 0
                self._generation += 1
                self._cond.notify_all()
                return index
            while gen == self._generation:
                self._cond.wait()
            return index
