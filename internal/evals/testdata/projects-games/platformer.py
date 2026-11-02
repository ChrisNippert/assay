TILE = 16
W = 12
H = 14


class Platformer:
    def __init__(self, level):
        self.rows = list(level)
        self.rh = len(self.rows)
        self.rw = max(len(r) for r in self.rows)
        self.x = self.y = 0
        for r, line in enumerate(self.rows):
            c = line.find("@")
            if c >= 0:
                self.x = c * TILE + 2
                self.y = (r + 1) * TILE - H
        self.vx = 0
        self.vy = 0
        self.dead = False
        self.won = False
        self.frames = 0
        self.on_ground = self._hits(self.x, self.y + 1)

    def _tile(self, c, r):
        if c < 0 or c >= self.rw or r < 0:
            return "#"
        if r >= self.rh:
            return "."
        line = self.rows[r]
        return line[c] if c < len(line) else "."

    def _tiles(self, x, y):
        for r in range(y // TILE, (y + H - 1) // TILE + 1):
            for c in range(x // TILE, (x + W - 1) // TILE + 1):
                yield c, r

    def _hits(self, x, y):
        return any(self._tile(c, r) == "#" for c, r in self._tiles(x, y))

    def update(self, left=False, right=False, jump=False):
        if self.dead or self.won:
            return
        self.frames += 1
        self.vx = -3 if left and not right else 3 if right and not left else 0
        if jump and self.on_ground:
            self.vy = -12
        elif not jump and self.vy < -4:
            self.vy = -4
        self.vy = min(self.vy + 1, 40)
        sx = 1 if self.vx > 0 else -1
        for _ in range(abs(self.vx)):
            if self._hits(self.x + sx, self.y):
                break
            self.x += sx
        sy = 1 if self.vy > 0 else -1
        for _ in range(abs(self.vy)):
            if self._hits(self.x, self.y + sy):
                self.vy = 0
                break
            self.y += sy
        self.on_ground = self._hits(self.x, self.y + 1)
        if any(self._tile(c, r) == "G" for c, r in self._tiles(self.x, self.y)):
            self.won = True
        elif self.y >= self.rh * TILE:
            self.dead = True
