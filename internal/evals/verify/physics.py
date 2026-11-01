"""Independent checker for the generated "physics" set.

Usage: python3 -I verify/physics.py questions.json

Each question's numbers are read back out of the prompt text and the answer
is recomputed by a different route than the Go code (numerical integration,
sympy equation solving, ray-transfer matrices, rapidity addition, ...).
Exits 1 and prints every mismatch where |python - answer| > tolerance/5.
"""
import json
import math
import re
import sys

import mpmath as mp
import sympy as sp

mp.mp.dps = 20
G = 9.81
MU = 3.986e14
RE = 6.371e6


def F(p, rx, i=1):
    rx = rx.replace(r"[-\d.]+", r"-?\d+(?:\.\d+)?").replace(r"[\d.]+", r"\d+(?:\.\d+)?")
    m = re.search(rx, p)
    if not m:
        raise ValueError("no match for %r" % rx)
    return float(m.group(i).replace("−", "-"))


def rk4(f, y, t0, t1, n):
    h = (t1 - t0) / n
    t = t0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, [a + h / 2 * b for a, b in zip(y, k1)])
        k3 = f(t + h / 2, [a + h / 2 * b for a, b in zip(y, k2)])
        k4 = f(t + h, [a + h * b for a, b in zip(y, k3)])
        y = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(y, k1, k2, k3, k4)]
        t += h
    return y


def proj(p):
    h = F(p, r"from a point ([\d.]+) m above")
    v = F(p, r"speed of ([\d.]+) m/s")
    th = math.radians(F(p, r"(\d+)° above the horizontal"))
    H = F(p, r"surface is ([\d.]+) m above")
    # energy conservation gives the landing speed; vx is conserved
    vx = v * math.cos(th)
    vf = math.sqrt(v * v + 2 * G * (h - H))
    vyf = -math.sqrt(vf * vf - vx * vx)
    t = (v * math.sin(th) - vyf) / G
    return vx * t


def pulley(p):
    m1 = F(p, r"mass ([\d.]+) kg rests")
    th = math.radians(F(p, r"angle of (\d+)°"))
    mu = F(p, r"friction between block and incline is ([\d.]+)")
    m2 = F(p, r"hanging mass of ([\d.]+) kg")
    a, T, N = sp.symbols("a T N")
    sol = sp.solve([
        sp.Eq(T - m1 * G * sp.sin(th) - mu * N, m1 * a),
        sp.Eq(N, m1 * G * sp.cos(th)),
        sp.Eq(m2 * G - T, m2 * a)], [a, T, N])
    return float(sol[T])


def damped(p):
    m = F(p, r"A ([\d.]+) kg mass")
    k = F(p, r"constant ([\d.]+) N/m")
    b = F(p, r"b = ([\d.]+) kg/s")
    x0 = F(p, r"pulled ([\d.]+) cm") / 100
    t = F(p, r"at t = ([\d.]+) s\?")
    f = mp.odefun(lambda t, y: [y[1], -(b * y[1] + k * y[0]) / m], 0, [x0, 0])
    x, v = f(t)
    return float(0.5 * m * v * v + 0.5 * k * x * x) * 1000


def roll(p):
    I = {"uniform solid sphere": "2/5", "uniform solid cylinder": "1/2",
         "thin-walled hollow cylinder": "1", "thin spherical shell": "2/3"}
    name = re.match(r"A (.*?) is released", p).group(1)
    beta = sp.Rational(I[name])
    L = F(p, r"length ([\d.]+) m")
    th = math.radians(F(p, r"inclined at (\d+)°"))
    H = F(p, r"ledge ([\d.]+) m above")
    # Newton + torque: m a = m g sin - f ; f R = beta m R^2 (a/R)
    a, f = sp.symbols("a f")
    sol = sp.solve([sp.Eq(a, G * sp.sin(th) - f), sp.Eq(f, beta * a)], [a, f])
    acc = float(sol[a])
    v = math.sqrt(2 * acc * L)
    # flight: find the time when height reaches -H
    y = lambda t: -v * math.sin(th) * t - 0.5 * G * t * t + H
    t = float(mp.findroot(y, 0.5))
    return v * math.cos(th) * t


def collide(p):
    m1, v1, m2, v2 = (F(p, r"a ([\d.]+) kg block moving right at ([\d.]+) m/s", 1),
                      F(p, r"a ([\d.]+) kg block moving right at ([\d.]+) m/s", 2),
                      F(p, r"with a ([\d.]+) kg block moving left at ([\d.]+) m/s", 1),
                      F(p, r"with a ([\d.]+) kg block moving left at ([\d.]+) m/s", 2))
    s = F(p, r"patch ([\d.]+) m long \(coefficient of kinetic friction ([\d.]+)\)", 1)
    mu1 = F(p, r"patch ([\d.]+) m long \(coefficient of kinetic friction ([\d.]+)\)", 2)
    th = math.radians(F(p, r"at (\d+)° above the horizontal"))
    mu2 = F(p, r"incline rising.*?friction ([\d.]+)\)")
    M = m1 + m2
    vc = sp.Symbol("vc")
    vc = float(sp.solve(sp.Eq(m1 * v1 - m2 * v2, M * vc), vc)[0])
    # time-stepped motion: patch, then incline until rest
    x, v, dt = 0.0, vc, 1e-5
    while x < s:
        v -= mu1 * G * dt
        x += v * dt
    d = 0.0
    while v > 0:
        v -= G * (math.sin(th) + mu2 * math.cos(th)) * dt
        d += v * dt
    return d


def hohmann(p):
    h1 = F(p, r"altitude of ([\d.]+) km\. It fires")
    h2 = F(p, r"apoapsis\) is at an altitude of ([\d.]+) km")
    r1, r2 = RE + h1 * 1e3, RE + h2 * 1e3
    vp = math.sqrt(MU * (2 / r1 - 2 / (r1 + r2)))  # vis-viva
    def acc(t, s):
        x, y, vx, vy = s
        r3 = (x * x + y * y) ** 1.5
        return [vx, vy, -MU * x / r3, -MU * y / r3]
    s, t, dt = [r1, 0.0, 0.0, vp], 0.0, 4.0
    while True:
        n = rk4(acc, s, t, t + dt, 1)
        if n[1] < 0:  # crossed the far side of the orbit
            return (t + dt * s[1] / (s[1] - n[1])) / 60
        s, t = n, t + dt


def shadow(p):
    h = F(p, r"altitude of ([\d.]+) km, in a plane")
    r = RE + h * 1e3
    w = math.sqrt(MU / r ** 3)
    T = 2 * math.pi / w
    dt, n, k = 0.05, 0, 0
    N = int(T / dt)
    for i in range(N):
        a = w * (i + 0.5) * dt
        if r * math.cos(a) < 0 and abs(r * math.sin(a)) < RE:
            k += 1
    return k * dt / 60


def banked(p):
    th = math.radians(F(p, r"banked at (\d+)°"))
    R = F(p, r"radius ([\d.]+) m\.")
    mu = F(p, r"static friction between tyres and road is ([\d.]+)")
    N, v = sp.symbols("N v", positive=True)
    f = mu * N
    # centripetal: horizontal toward the centre; vertical balance
    sol = sp.solve([sp.Eq(N * sp.sin(th) + f * sp.cos(th), v ** 2 / R),
                    sp.Eq(N * sp.cos(th) - f * sp.sin(th), G)], [N, v], dict=True)
    return float(sol[0][v]) * 3.6


def bridge(p):
    E = F(p, r"EMF ([\d.]+) V")
    ri = F(p, r"internal resistance ([\d.]+) Ω")
    R = [F(p, r"R%d = ([\d.]+) Ω" % i) for i in range(1, 6)]
    # mesh currents: i1 (E, ri, R1, R3 loop through C), i2 (R2, R4 loop), i5 via R5
    i1, i2, i3 = sp.symbols("i1 i2 i3")
    # i1: battery->R1->R3 ; i2: battery->R2->R4 ; i3 circulates R1,R5,R2 (A->C->D->A)
    eqs = [
        sp.Eq(E, ri * (i1 + i2) + R[0] * (i1 - i3) + R[2] * i1),
        sp.Eq(E, ri * (i1 + i2) + R[1] * (i2 + i3) + R[3] * i2),
        sp.Eq(0, R[0] * (i3 - i1) + R[4] * i3 + R[1] * (i3 + i2)),
    ]
    s = sp.solve(eqs, [i1, i2, i3])
    return abs(float(s[i3])) * 1000


def rc(p):
    R2 = F(p, r"R2 = ([\d.]+) kΩ")
    C = F(p, r"C = ([\d.]+) μF")
    V = F(p, r"ideal ([\d.]+) V source")
    R1 = F(p, r"R1 = ([\d.]+) kΩ")
    t1 = F(p, r"At t = ([\d.]+) ms the switch")
    t2 = F(p, r"\(that is, ([\d.]+) ms after")
    # units: kΩ, μF, ms -> time constants in ms
    f1 = mp.odefun(lambda t, y: [((V - y[0]) / R1 - y[0] / R2) / C], 0, [0])
    v1 = f1(t1)[0]
    f2 = mp.odefun(lambda t, y: [-y[0] / R2 / C], 0, [v1])
    return float(f2(t2)[0])


def rlc(p):
    R = F(p, r"R = ([\d.]+) Ω")
    L = F(p, r"L = ([\d.]+) mH") * 1e-3
    C = F(p, r"C = ([\d.]+) μF") * 1e-6
    V = F(p, r"amplitude ([\d.]+) V")
    w = 2 * math.pi * F(p, r"f = ([\d.]+) Hz")
    A, B = sp.symbols("A B")
    # q = A cos wt + B sin wt in L q'' + R q' + q/C = V cos wt
    s = sp.solve([sp.Eq(-L * w * w * A + R * w * B + A / C, V),
                  sp.Eq(-L * w * w * B - R * w * A + B / C, 0)], [A, B])
    return math.hypot(float(s[A]), float(s[B])) / C


def efield(p):
    k = 8.99e9
    ch = re.findall(r"q\d = ([+-][\d.]+) nC at \(x, y\) = \(([-\d.]+), ([-\d.]+)\) cm", p)
    m = re.search(r"at the point \(x, y\) = \(([-\d.]+), ([-\d.]+)\) cm", p)
    P = mp.mpc(float(m.group(1)), float(m.group(2))) / 100
    tot = mp.mpc(0)
    for q, x, y in ch:
        d = P - mp.mpc(float(x), float(y)) / 100
        tot += k * float(q) * 1e-9 * d / abs(d) ** 3
    return float(abs(tot)) / 1000


def gascycle(p):
    n = F(p, r"n = ([\d.]+) mol")
    mono = "monatomic" in p
    T = F(p, r"temperature is ([\d.]+) K")
    V1 = F(p, r"volume is ([\d.]+) L") * 1e-3
    ratio = F(p, r"to a volume ([\d.]+) times larger")
    R = 8.314462618
    cv = 1.5 * R if mono else 2.5 * R
    V2 = ratio * V1
    P1 = n * R * T / V1
    P2 = n * R * T / V2
    TC = P2 * V1 / (n * R)
    WAB = mp.quad(lambda V: n * R * T / V, [V1, V2])
    WBC = P2 * (V1 - V2)
    QAB = WAB  # isothermal
    QBC = n * (cv + R) * (TC - T)
    QCA = n * cv * (T - TC)
    W = WAB + WBC
    assert abs(W - (QAB + QBC + QCA)) < 1e-6 * abs(W)  # first law around the cycle
    qin = sum(q for q in (QAB, QBC, QCA) if q > 0)
    return float(100 * W / qin)


def lenses(p):
    f1 = F(p, r"Lens 1 has focal length ([-\d.]+) cm")
    f2 = F(p, r"lens 2 has focal length ([-\d.]+) cm")
    d = F(p, r"sits ([\d.]+) cm to the right")
    s = F(p, r"axis ([\d.]+) cm to the left of lens 1")
    z = sp.Symbol("z")
    T = lambda L: sp.Matrix([[1, L], [0, 1]])
    Lz = lambda f: sp.Matrix([[1, 0], [-sp.Rational(1) / f, 1]])
    M = T(z) * Lz(sp.nsimplify(f2)) * T(sp.nsimplify(d)) * Lz(sp.nsimplify(f1)) * T(sp.nsimplify(s))
    return float(sp.solve(M[0, 1], z)[0])  # B = 0 is the imaging condition


def doppler(p):
    f0 = F(p, r"tone of ([\d.]+) Hz")
    vs = F(p, r"moves to the right at ([\d.]+) m/s")
    vo = F(p, r"The observer moves at ([\d.]+) m/s")
    vo = vo if "away from the source" in p else -vo
    w = F(p, r"wind of ([\d.]+) m/s")
    w = w if "towards the observer (to the right)" in p else -w
    c = 343.0 + w  # sound speed over the ground, rightwards
    x0 = 500.0
    def arrive(te):  # observer position at t: x0 + vo t; pulse: vs te + c (t - te)
        return (x0 - vs * te + c * te) / (c - vo)
    return 1.0 / (arrive(1 / f0) - arrive(0.0))


def twins(p):
    D = F(p, r"A star is ([\d.]+) light-years")
    b1 = F(p, r"leaves Earth at ([\d.]+) c")
    u = F(p, r"towards Earth at ([\d.]+) c")
    ph1, phu = mp.atanh(b1), mp.atanh(u)
    ph2 = phu - ph1  # rapidity of B toward Earth, in Earth's frame
    T = D / mp.tanh(ph1) + D / mp.tanh(ph2)
    tau = D / mp.sinh(ph1) + D / mp.sinh(ph2)  # proper time = D/(gamma beta)
    return float(T - tau)


def calorimeter(p):
    mc = F(p, r"cup of mass ([\d.]+) g") / 1000
    mw = F(p, r"contains ([\d.]+) g") / 1000
    Tw = F(p, r"both at ([\d.]+) °C")
    mi = F(p, r"Then ([\d.]+) g of ice") / 1000
    Ti = F(p, r"ice at −([\d.]+) °C")
    Lf = 334e3
    # enthalpy relative to liquid water at 0 °C
    H = (mw * 4186 + mc * 900) * Tw - mi * (Lf + 2100 * Ti)
    if "equilibrium temperature" in p:
        assert H > 0
        return H / ((mw + mi) * 4186 + mc * 900)
    assert H < 0
    return -H / Lf * 1000  # ice mass in g


def sinker(p):
    a = F(p, r"side ([\d.]+) cm") / 100
    rw = F(p, r"density ([\d.]+) kg/m³ floats")
    rf = F(p, r"liquid of density ([\d.]+) kg/m³")
    m = sp.Symbol("m", positive=True)
    # forces on cube+sinker: weight = buoyancy on cube + buoyancy on lead
    eq = sp.Eq(rw * a ** 3 + m, rf * a ** 3 + rf * m / 11340)
    return float(sp.solve(eq, m)[0]) * 1000


CHECKS = {k: v for k, v in globals().items() if callable(v) and k in (
    "proj pulley damped roll collide hohmann shadow banked bridge rc rlc efield "
    "gascycle lenses doppler twins calorimeter sinker").split()}


def main():
    qs = json.load(open(sys.argv[1]))
    bad = 0
    for q in qs:
        kind = q["id"].rsplit("-", 1)[0]
        if kind not in CHECKS:
            print("%s: no checker for %r" % (q["id"], kind))
            bad += 1
            continue
        try:
            got = CHECKS[kind](q["prompt"])
        except Exception as e:  # unparsable prompt or failed assertion
            print("%s: checker error: %r" % (q["id"], e))
            bad += 1
            continue
        want = float(q["answer"])
        if abs(got - want) > q["tolerance"] / 5:
            print("%s: python %.8g vs answer %.8g (tolerance %.3g)" % (q["id"], got, want, q["tolerance"]))
            bad += 1
    print("physics: %d questions, %d mismatches" % (len(qs), bad))
    sys.exit(1 if bad else 0)


main()
