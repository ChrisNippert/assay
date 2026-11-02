import math

B_CRIT = 3 * math.sqrt(3)


def captured(b):
    return abs(b) <= B_CRIT


def deflection(b):
    b = abs(b)
    if b <= B_CRIT:
        return None
    f = lambda u: 1 / b**2 - u * u + 2 * u**3
    lo, hi = 0.0, 1 / 3
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    u0 = (lo + hi) / 2
    # u = u0 (1 - t^2) removes the inverse-square-root singularity
    n = 4000
    total = 0.0
    for k in range(n):
        t = (k + 0.5) / n
        u = u0 * (1 - t * t)
        total += 2 * u0 * t / math.sqrt(f(u))
    return 2 * total / n - math.pi


def shadow(n, fov_deg, r_obs):
    T = math.tan(math.radians(fov_deg) / 2)
    k = math.sqrt(1 - 2 / r_obs)
    grid = []
    for i in range(n):
        y = (1 - 2 * (i + 0.5) / n) * T
        row = []
        for j in range(n):
            x = (2 * (j + 0.5) / n - 1) * T
            s = math.sqrt((x * x + y * y) / (1 + x * x + y * y))  # sin(alpha)
            row.append(r_obs * s / k <= B_CRIT)
        grid.append(row)
    return grid
