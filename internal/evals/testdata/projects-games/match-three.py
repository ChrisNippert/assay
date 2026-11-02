class Match3:
    def __init__(self, rows, refill):
        self.g = [list(r) for r in rows]
        self.h = len(self.g)
        self.w = len(self.g[0]) if self.g else 0
        self.refill = refill
        self.refill_pos = 0
        self.score = 0
        self.moves = 0

    def board(self):
        return ["".join(r) for r in self.g]

    def _runs(self):
        """Maximal runs of 3+ equal letters: list of lists of cells."""
        g = self.g
        runs = []
        for r in range(self.h):
            c = 0
            while c < self.w:
                e = c
                while e + 1 < self.w and g[r][e + 1] == g[r][c]:
                    e += 1
                if e - c + 1 >= 3:
                    runs.append([(r, k) for k in range(c, e + 1)])
                c = e + 1
        for c in range(self.w):
            r = 0
            while r < self.h:
                e = r
                while e + 1 < self.h and g[e + 1][c] == g[r][c]:
                    e += 1
                if e - r + 1 >= 3:
                    runs.append([(k, c) for k in range(r, e + 1)])
                r = e + 1
        return runs

    def _try(self, r1, c1, r2, c2):
        if not (0 <= r1 < self.h and 0 <= r2 < self.h and 0 <= c1 < self.w and 0 <= c2 < self.w):
            return False
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
        g = self.g
        g[r1][c1], g[r2][c2] = g[r2][c2], g[r1][c1]
        ok = bool(self._runs())
        g[r1][c1], g[r2][c2] = g[r2][c2], g[r1][c1]
        return ok

    def swap(self, r1, c1, r2, c2):
        if not self._try(r1, c1, r2, c2):
            return -1
        g = self.g
        g[r1][c1], g[r2][c2] = g[r2][c2], g[r1][c1]
        total = 0
        k = 1
        while True:
            runs = self._runs()
            if not runs:
                break
            cells = set()
            bonus = 0
            for run in runs:
                cells.update(run)
                bonus += 20 * (len(run) - 3)
            total += k * (10 * len(cells) + bonus)
            for r, c in cells:
                g[r][c] = None
            for c in range(self.w):
                col = [g[r][c] for r in range(self.h) if g[r][c] is not None]
                missing = self.h - len(col)
                new = []
                for _ in range(missing):
                    new.append(self.refill[self.refill_pos % len(self.refill)])
                    self.refill_pos += 1
                # new[0] fills the lowest empty cell (row missing-1), new[-1] row 0
                col = new[::-1] + col
                for r in range(self.h):
                    g[r][c] = col[r]
            k += 1
        self.moves += 1
        self.score += total
        return total

    def find_move(self):
        for r in range(self.h):
            for c in range(self.w):
                if self._try(r, c, r, c + 1):
                    return (r, c, r, c + 1)
                if self._try(r, c, r + 1, c):
                    return (r, c, r + 1, c)
        return None

    def has_moves(self):
        return self.find_move() is not None
