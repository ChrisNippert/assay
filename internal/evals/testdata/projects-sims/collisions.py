import math


def simulate(disks, width, height, t):
    x = [float(d[0]) for d in disks]
    y = [float(d[1]) for d in disks]
    vx = [float(d[2]) for d in disks]
    vy = [float(d[3]) for d in disks]
    r = [float(d[4]) for d in disks]
    m = [float(d[5]) for d in disks]
    n = len(disks)
    left = float(t)
    for _ in range(1000000):
        best, ev = left, None
        for i in range(n):
            if vx[i] > 0:
                tt = max(0.0, (width - r[i] - x[i]) / vx[i])
                if tt < best:
                    best, ev = tt, ("wx", i)
            elif vx[i] < 0:
                tt = max(0.0, (r[i] - x[i]) / vx[i])
                if tt < best:
                    best, ev = tt, ("wx", i)
            if vy[i] > 0:
                tt = max(0.0, (height - r[i] - y[i]) / vy[i])
                if tt < best:
                    best, ev = tt, ("wy", i)
            elif vy[i] < 0:
                tt = max(0.0, (r[i] - y[i]) / vy[i])
                if tt < best:
                    best, ev = tt, ("wy", i)
            for j in range(i + 1, n):
                dx, dy = x[j] - x[i], y[j] - y[i]
                dvx, dvy = vx[j] - vx[i], vy[j] - vy[i]
                b = dx * dvx + dy * dvy
                if b >= 0:
                    continue
                a = dvx * dvx + dvy * dvy
                R = r[i] + r[j]
                c = dx * dx + dy * dy - R * R
                disc = b * b - a * c
                if disc < 0:
                    continue
                tt = max(0.0, (-b - math.sqrt(disc)) / a)
                if tt < best:
                    best, ev = tt, ("pair", i, j)
        for i in range(n):
            x[i] += vx[i] * best
            y[i] += vy[i] * best
        left -= best
        if ev is None:
            break
        if ev[0] == "wx":
            vx[ev[1]] = -vx[ev[1]]
        elif ev[0] == "wy":
            vy[ev[1]] = -vy[ev[1]]
        else:
            i, j = ev[1], ev[2]
            dx, dy = x[j] - x[i], y[j] - y[i]
            d = math.hypot(dx, dy)
            nx, ny = dx / d, dy / d
            rel = (vx[i] - vx[j]) * nx + (vy[i] - vy[j]) * ny
            k = 2 * rel / (m[i] + m[j])
            vx[i] -= k * m[j] * nx
            vy[i] -= k * m[j] * ny
            vx[j] += k * m[i] * nx
            vy[j] += k * m[i] * ny
    return [(x[i], y[i], vx[i], vy[i], r[i], m[i]) for i in range(n)]
