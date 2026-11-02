class Sandbox:
    def __init__(self, rows):
        self.g = [list(r) for r in rows]
        self.h = len(self.g)
        self.w = len(self.g[0]) if self.g else 0
        self.generation = 0

    def rows(self):
        return ["".join(r) for r in self.g]

    def count(self, ch):
        return sum(r.count(ch) for r in self.g)

    def _empty(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w and self.g[r][c] == "."

    def step(self):
        g = self.g
        pref = 1 if self.generation % 2 == 0 else -1
        cols = range(self.w) if pref == 1 else range(self.w - 1, -1, -1)
        moved = [[False] * self.w for _ in range(self.h)]
        for r in range(self.h - 1, -1, -1):
            for c in cols:
                p = g[r][c]
                if p not in "sw" or moved[r][c]:
                    continue
                if p == "s":
                    if r + 1 < self.h and g[r + 1][c] in ".w":
                        below = g[r + 1][c]
                        g[r + 1][c] = "s"
                        g[r][c] = below
                        moved[r + 1][c] = True
                        moved[r][c] = True
                        continue
                    for dx in (pref, -pref):
                        if self._empty(r + 1, c + dx):
                            g[r][c] = "."
                            g[r + 1][c + dx] = "s"
                            moved[r + 1][c + dx] = True
                            break
                else:
                    if self._empty(r + 1, c):
                        g[r][c] = "."
                        g[r + 1][c] = "w"
                        moved[r + 1][c] = True
                        continue
                    done = False
                    for dx in (pref, -pref):
                        if self._empty(r + 1, c + dx):
                            g[r][c] = "."
                            g[r + 1][c + dx] = "w"
                            moved[r + 1][c + dx] = True
                            done = True
                            break
                    if done:
                        continue
                    for dx in (pref, -pref):
                        if self._empty(r, c + dx):
                            g[r][c] = "."
                            g[r][c + dx] = "w"
                            moved[r][c + dx] = True
                            break
        self.generation += 1
