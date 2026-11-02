import math


def solve_kepler(M, e):
    M = math.fmod(M, 2 * math.pi)
    if M < 0:
        M += 2 * math.pi
    E = M + e * math.sin(M) if e < 0.8 else math.pi
    for _ in range(100):
        f = E - e * math.sin(E) - M
        E -= f / (1 - e * math.cos(E))
        if abs(f) < 1e-15:
            break
    # polish with bisection-safe check
    E = math.fmod(E, 2 * math.pi)
    if E < 0:
        E += 2 * math.pi
    return E


def period(a, mu):
    return 2 * math.pi * math.sqrt(a**3 / mu)


def state_at(a, e, mu, t):
    n = math.sqrt(mu / a**3)
    E = solve_kepler(n * t, e)
    r = a * (1 - e * math.cos(E))
    x = a * (math.cos(E) - e)
    y = a * math.sqrt(1 - e * e) * math.sin(E)
    k = math.sqrt(mu * a) / r
    return (x, y, -k * math.sin(E), k * math.sqrt(1 - e * e) * math.cos(E))


def time_since_periapsis(a, e, mu, nu):
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    M = E - e * math.sin(E)
    M = math.fmod(M, 2 * math.pi)
    if M < 0:
        M += 2 * math.pi
    return M / math.sqrt(mu / a**3)
