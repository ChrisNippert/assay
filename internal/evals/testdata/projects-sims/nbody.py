import math


def _acc(m, pos, G):
    n = len(m)
    a = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = [pos[j][k] - pos[i][k] for k in range(3)]
            r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            f = G / (r2 * math.sqrt(r2))
            for k in range(3):
                a[i][k] += f * m[j] * d[k]
                a[j][k] -= f * m[i] * d[k]
    return a


def simulate(masses, positions, velocities, dt, steps, G=1.0):
    pos = [[float(c) for c in p] for p in positions]
    vel = [[float(c) for c in v] for v in velocities]
    a = _acc(masses, pos, G)
    for _ in range(steps):
        for i in range(len(pos)):
            for k in range(3):
                vel[i][k] += 0.5 * dt * a[i][k]
                pos[i][k] += dt * vel[i][k]
        a = _acc(masses, pos, G)
        for i in range(len(pos)):
            for k in range(3):
                vel[i][k] += 0.5 * dt * a[i][k]
    return pos, vel


def energy(masses, positions, velocities, G=1.0):
    e = 0.0
    n = len(masses)
    for i in range(n):
        e += 0.5 * masses[i] * sum(c * c for c in velocities[i])
        for j in range(i + 1, n):
            r = math.dist(positions[i], positions[j])
            e -= G * masses[i] * masses[j] / r
    return e


def momentum(masses, velocities):
    return [sum(m * v[k] for m, v in zip(masses, velocities)) for k in range(3)]
