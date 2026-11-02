import math


def _deriv(s, m1, m2, l1, l2, g):
    t1, t2, w1, w2 = s
    d = t1 - t2
    c, sn = math.cos(d), math.sin(d)
    den = 2 * m1 + m2 - m2 * math.cos(2 * d)
    a1 = (-g * (2 * m1 + m2) * math.sin(t1) - m2 * g * math.sin(t1 - 2 * t2)
          - 2 * sn * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * c)) / (l1 * den)
    a2 = (2 * sn * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * math.cos(t1)
                    + w2 * w2 * l2 * m2 * c)) / (l2 * den)
    return (w1, w2, a1, a2)


def simulate(theta1, theta2, omega1, omega2, t, dt=0.001, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    s = (theta1, theta2, omega1, omega2)
    n = max(1, int(math.ceil(t / dt - 1e-9))) if t > 0 else 0
    h = t / n if n else 0.0
    p = (m1, m2, l1, l2, g)
    for _ in range(n):
        k1 = _deriv(s, *p)
        k2 = _deriv([x + 0.5 * h * k for x, k in zip(s, k1)], *p)
        k3 = _deriv([x + 0.5 * h * k for x, k in zip(s, k2)], *p)
        k4 = _deriv([x + h * k for x, k in zip(s, k3)], *p)
        s = tuple(x + h / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(s, k1, k2, k3, k4))
    return s


def energy(theta1, theta2, omega1, omega2, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    ke = 0.5 * m1 * (l1 * omega1) ** 2 + 0.5 * m2 * (
        (l1 * omega1) ** 2 + (l2 * omega2) ** 2 + 2 * l1 * l2 * omega1 * omega2 * math.cos(theta1 - theta2))
    pe = -(m1 + m2) * g * l1 * math.cos(theta1) - m2 * g * l2 * math.cos(theta2)
    return ke + pe
