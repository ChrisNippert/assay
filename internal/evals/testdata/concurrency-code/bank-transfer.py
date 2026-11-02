import threading


class InsufficientFunds(Exception):
    pass


class _Account:
    def __init__(self, amount):
        self.lock = threading.Lock()
        self.amount = amount

    def get(self):
        return self.amount

    def add(self, n):
        self.amount += n


class Bank:
    def __init__(self, balances):
        self._accounts = {name: _Account(int(v)) for name, v in balances.items()}
        self._order = sorted(self._accounts)

    def balance(self, name):
        acct = self._accounts[name]
        with acct.lock:
            return acct.get()

    def total(self):
        # take every lock in the one global order, so no transfer is half done
        accts = [self._accounts[n] for n in self._order]
        for a in accts:
            a.lock.acquire()
        try:
            return sum(a.get() for a in accts)
        finally:
            for a in reversed(accts):
                a.lock.release()

    def transfer(self, src, dst, amount):
        if type(amount) is not int or amount <= 0:
            raise ValueError("amount must be a positive int")
        a, b = self._accounts[src], self._accounts[dst]
        if a is b:
            with a.lock:
                if a.get() < amount:
                    raise InsufficientFunds(src)
            return
        first, second = (a, b) if src < dst else (b, a)
        with first.lock:
            with second.lock:
                if a.get() < amount:
                    raise InsufficientFunds(src)
                a.add(-amount)
                b.add(amount)
