_DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    out = []
    while True:
        out.append((x, y))
        if (x, y) == (x1, y1):
            return out
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


class Dungeon:
    HP = {"g": 6, "O": 15}

    def __init__(self, rows, rolls):
        self.rows = list(rows)
        self.h = len(self.rows)
        self.w = max(len(r) for r in self.rows)
        self.rolls = list(rolls)
        self.rolls_used = 0
        self.wall = set()
        self.items = {}
        self.mons = []  # [kind, x, y, hp]
        self.pos = (0, 0)
        for y, line in enumerate(self.rows):
            for x, ch in enumerate(line):
                if ch == "#":
                    self.wall.add((x, y))
                elif ch == "@":
                    self.pos = (x, y)
                elif ch in "gO":
                    self.mons.append([ch, x, y, self.HP[ch]])
                elif ch in "!>":
                    self.items[(x, y)] = ch
        self.hp = 20
        self.max_hp = 20
        self.state = "playing"
        self.turn = 0
        self.kills = 0

    def _roll(self):
        v = self.rolls[self.rolls_used % len(self.rolls)]
        self.rolls_used += 1
        return v

    def _blocked(self, x, y):
        return x < 0 or y < 0 or x >= self.w or y >= self.h or (x, y) in self.wall

    def monsters(self):
        return [(m[0], m[1], m[2], m[3]) for m in self.mons]

    def fov(self):
        px, py = self.pos
        out = []
        for y in range(max(0, py - 8), min(self.h, py + 9)):
            for x in range(max(0, px - 8), min(self.w, px + 9)):
                if (x - px) ** 2 + (y - py) ** 2 > 64:
                    continue
                line = bresenham(px, py, x, y)
                if all(not self._blocked(cx, cy) for cx, cy in line[1:-1]):
                    out.append((x, y))
        return sorted(out)

    def act(self, action=None):
        if self.state != "playing":
            return False
        if action in _DIRS:
            dx, dy = _DIRS[action]
            tx, ty = self.pos[0] + dx, self.pos[1] + dy
            if self._blocked(tx, ty):
                return False
            target = next((m for m in self.mons if m[1] == tx and m[2] == ty), None)
            if target is not None:
                target[3] -= self._roll() + 1
                if target[3] <= 0:
                    self.mons.remove(target)
                    self.kills += 1
            else:
                self.pos = (tx, ty)
                it = self.items.get(self.pos)
                if it == "!":
                    self.hp = min(self.max_hp, self.hp + 8)
                    del self.items[self.pos]
                elif it == ">":
                    self.state = "won"
        self.turn += 1
        if self.state == "won":
            return True
        vis = set(self.fov())
        px, py = self.pos
        for m in list(self.mons):
            if (m[1], m[2]) not in vis:
                continue
            dist = abs(m[1] - px) + abs(m[2] - py)
            if dist == 1:
                r = self._roll()
                if m[0] == "g" and r >= 3:
                    self.hp -= r - 2
                elif m[0] == "O" and r >= 2:
                    self.hp -= r + 1
                if self.hp <= 0:
                    self.state = "dead"
                    break
                continue
            best = None
            for d in "NESW":
                nx, ny = m[1] + _DIRS[d][0], m[2] + _DIRS[d][1]
                if self._blocked(nx, ny) or (nx, ny) == self.pos:
                    continue
                if any(o is not m and o[1] == nx and o[2] == ny for o in self.mons):
                    continue
                nd = abs(nx - px) + abs(ny - py)
                if nd < dist and (best is None or nd < best[0]):
                    best = (nd, nx, ny)
            if best:
                m[1], m[2] = best[1], best[2]
        return True
