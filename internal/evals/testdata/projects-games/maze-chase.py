from collections import deque

_D = {"U": (-1, 0), "L": (0, -1), "D": (1, 0), "R": (0, 1)}
_OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}
_ORDER = "ULDR"


class MazeGame:
    def __init__(self, maze):
        rows = maze.splitlines()
        self.walls, self.pellets, self.powers = set(), set(), set()
        self.start_player = None
        self.starts = []
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == ".":
                    self.pellets.add((r, c))
                elif ch == "o":
                    self.powers.add((r, c))
                elif ch == "P":
                    self.start_player = (r, c)
                elif ch == "C":
                    self.starts.append((r, c))
        self.h = len(rows)
        self.w = max((len(x) for x in rows), default=0)
        self.player = self.start_player
        self.chasers = list(self.starts)
        self.heading = [None] * len(self.starts)
        self.score = 0
        self.lives = 3
        self.tick = 0
        self.frightened = 0
        self.combo = 0
        self.state = "playing"

    @property
    def pellets_left(self):
        return len(self.pellets) + len(self.powers)

    def _wall(self, p):
        r, c = p
        return r < 0 or c < 0 or r >= self.h or c >= self.w or p in self.walls

    def _dists(self):
        d = {self.player: 0}
        q = deque([self.player])
        while q:
            p = q.popleft()
            for dr, dc in _D.values():
                n = (p[0] + dr, p[1] + dc)
                if n not in d and not self._wall(n):
                    d[n] = d[p] + 1
                    q.append(n)
        return d

    def _collide(self):
        for i in range(len(self.chasers)):
            if self.chasers[i] != self.player:
                continue
            if self.frightened > 0:
                self.score += 200 * 2 ** self.combo
                self.combo += 1
                self.chasers[i] = self.starts[i]
                self.heading[i] = None
            else:
                self.lives -= 1
                if self.lives == 0:
                    self.state = "lost"
                else:
                    self.player = self.start_player
                    self.chasers = list(self.starts)
                    self.heading = [None] * len(self.starts)
                    self.frightened = 0
                return True
        return False

    def _move_chasers(self):
        dist = self._dists()
        for i, (r, c) in enumerate(self.chasers):
            cands = [d for d in _ORDER if not self._wall((r + _D[d][0], c + _D[d][1]))]
            h = self.heading[i]
            if h:
                nonrev = [d for d in cands if d != _OPP[h]]
                if nonrev:
                    cands = nonrev
            if not cands:
                continue
            best = None
            for d in cands:
                v = dist.get((r + _D[d][0], c + _D[d][1]), 10 ** 9)
                if best is None or (v > best[0] if self.frightened > 0 else v < best[0]):
                    best = (v, d)
            d = best[1]
            self.chasers[i] = (r + _D[d][0], c + _D[d][1])
            self.heading[i] = d

    def step(self, direction=None):
        if self.state != "playing":
            return self.state
        self.tick += 1
        if direction in _D:
            n = (self.player[0] + _D[direction][0], self.player[1] + _D[direction][1])
            if not self._wall(n):
                self.player = n
        p = self.player
        if p in self.pellets:
            self.pellets.discard(p)
            self.score += 10
        elif p in self.powers:
            self.powers.discard(p)
            self.score += 50
            self.frightened = 8
            self.combo = 0
        if self.pellets_left == 0:
            self.state = "won"
            return self.state
        if self._collide():
            return self.state
        self._move_chasers()
        if self._collide():
            return self.state
        if self.frightened > 0:
            self.frightened -= 1
        return self.state
