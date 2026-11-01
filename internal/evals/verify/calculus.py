"""Independent checker for the generated Calculus set.

Usage: python3 -I verify/calculus.py questions.json

Each question's numbers and expressions are read back out of its prompt text,
and the answer is recomputed by a route that differs from the Go code (the Go
side uses closed forms, Simpson, RK4 and bisection; here it is mpmath
quadrature, ODE series solvers, root finders, sympy's limit/series/integrate,
exact rational arithmetic and shell methods). Exit 1 and print every mismatch
where |python - answer| > tolerance/5.
"""
import json
import re
import sys
from fractions import Fraction

import mpmath as mp
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr,
    standard_transformations)

mp.mp.dps = 30
x, y, z, t, n = sp.symbols("x y z t n")
TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)


def norm(s):
    s = re.sub(r"e\^\(([^()]*)\)", r"exp(\1)", s)
    for a, b in (("−", "-"), ("·", "*"), ("²", "**2"), ("³", "**3"), ("⁴", "**4"),
                 ("√", "sqrt"), ("^", "**"), ("∞", "oo")):
        s = s.replace(a, b)
    return s


def E(s):
    """Prompt text -> sympy expression (floats made exact)."""
    return sp.nsimplify(parse_expr(norm(s), transformations=TRANS,
                                   local_dict={"x": x, "y": y, "z": z, "t": t, "n": n}),
                        rational=True)


def F(s):
    return mp.mpf(str(s))


def fn(expr, var=x):
    return sp.lambdify(var, expr, "mpmath")


def grab(pat, p, flags=re.S):
    m = re.search(pat, p, flags)
    if not m:
        raise ValueError("prompt does not match " + pat)
    return m.groups()


def integral(p):
    lo, hi, body = grab(r"∫ from (\S+) to (\S+) of (.+?) dx", p)
    return lo, hi, E(body)


def k_ibp(p):
    lo, hi, e = integral(p)
    return mp.quad(fn(e), [F(lo), F(hi)])


def k_partial(p):
    lo, hi, e = integral(p)
    return sp.N(sp.integrate(sp.apart(e, x), (x, E(lo), E(hi))), 20)


def k_trigsub(p):
    lo, hi, e = integral(p)
    return mp.quad(fn(e), [F(lo), F(hi)])


def k_improper(p):
    lo, hi, e = integral(p)
    f = fn(e)
    return mp.quad(f, [0, 1, 3, 8, 20, 50, mp.inf])


def k_limit(p):
    (body,) = grab(r"lim as x→0 of (.+?)\. Give", p)
    return sp.N(sp.limit(E(body), x, 0), 20)


def k_implicit(p):
    a, b, c, x0, y0 = grab(r"x³ \+ (\d+)·x·y \+ (\d*)·?y³ = (-?\d+) passes through the point \((-?\d+), (-?\d+)\)", p)
    b = b or "1"
    eq = x**3 + int(a) * x * y + int(b) * y**3 - int(c)
    assert eq.subs({x: int(x0), y: int(y0)}) == 0, "point not on curve"
    d2 = sp.idiff(eq, y, x, 2)
    return sp.N(d2.subs({x: int(x0), y: int(y0)}), 20)


def k_related(p):
    if "ladder" in p:
        L, s, d = (F(v) for v in grab(r"A (\S+)-ft ladder.*? at (\S+) ft/s\. At the instant when the bottom of the ladder is (\S+) ft", p))
        y0 = mp.sqrt(L * L - d * d)

        def area(tt):
            yy = y0 - s * tt
            return mp.sqrt(L * L - yy * yy) * yy / 2
        return mp.diff(area, 0)
    R, H, Q, k, h = (F(v) for v in grab(r"top radius (\S+) ft and height (\S+) ft\..*?constant (\S+) ft³/min.*?rate of (\S+)·√h.*?depth is (\S+) ft", p))

    def vol(hh):
        return mp.pi * R * R * hh ** 3 / (3 * H * H)
    return (Q - k * mp.sqrt(h)) / mp.diff(vol, h)


def k_optim(p):
    if "cardboard" in p:
        W, H = (E(v) for v in grab(r"cardboard (\S+) inches by (\S+) inches", p))
        V = x * (W - 2 * x) * (H - 2 * x)
        best = max(sp.N(V.subs(x, r), 20) for r in sp.solve(sp.diff(V, x), x)
                   if 0 < sp.N(r) < sp.N(min(W, H) / 2))
        return best
    V, ct, cb, cs = (F(v) for v in grab(r"exactly (\S+) cm³.*?top costs (\S+) cents.*?bottom (\S+) cents.*?side (\S+) cents", p))

    def cost(r):
        return 2 * mp.pi * r * (V / (mp.pi * r * r)) * cs + mp.pi * r * r * (ct + cb)
    r0 = min((cost(F(i) / 4), F(i) / 4) for i in range(1, 400))[1]
    r = mp.findroot(lambda q: mp.diff(cost, q), r0)
    return cost(r)


def k_arclen(p):
    a, b, arg, L = grab(r"y = (\S+)·x² \+ (\S+)·cos\((.+?)\) from x = 0 to x = (\S+)\.", p)
    yy = E(a) * x**2 + E(b) * sp.cos(E(arg))
    d = fn(sp.diff(yy, x))
    L = F(L)
    pts = [L * i / 12 for i in range(13)]
    return mp.quad(lambda u: mp.sqrt(1 + d(u) ** 2), pts)


def k_revol(p):
    a, b, k = (F(v) for v in grab(r"y = (\S+)·√x and y = x²/(\S+) .*?line y = -(\S+)\.", p))
    xc = (a * b) ** (mp.mpf(2) / 3)
    Y = a * mp.sqrt(xc)
    # horizontal shells in y: x runs from y^2/a^2 (upper curve) to sqrt(b*y)
    return mp.quad(lambda yy: 2 * mp.pi * (yy + k) * (mp.sqrt(b * yy) - yy ** 2 / a ** 2), [0, Y / 2, Y])


def k_areacross(p):
    q, r, c = (F(v) for v in grab(r"y = (\S+) − (\S+)·x² and y = (\S+)·e\^\(x/2\)", p))

    def g(u):
        return q - r * u * u - c * mp.exp(u / 2)
    roots = []
    grid = [F(i) / 20 for i in range(-200, 200)]
    for u0, u1 in zip(grid, grid[1:]):
        if g(u0) * g(u1) < 0:
            roots.append(mp.findroot(g, (u0, u1), solver="illinois"))
    assert len(roots) == 2, roots
    return mp.quad(g, [roots[0], 0, roots[1]])


def k_ode2(p):
    b, c, Fm, arg, y0, v0, t1 = grab(r"y'' \+ (\S+)·y' \+ (\S+)·y = (\S+)·cos\((.+?)\), with y\(0\) = (\S+) and y'\(0\) = (\S+)\. Find y\((\S+)\)", p)
    w = F(E(arg).subs(t, 1))
    b, c, Fm, y0, v0, t1 = (F(v) for v in (b, c, Fm, y0, v0, t1))
    sol = mp.odefun(lambda tt, Y: [Y[1], Fm * mp.cos(w * tt) - b * Y[1] - c * Y[0]], 0, [y0, v0])
    return sol(t1)[0]


def k_odefirst(p):
    if "e^(" in p:
        a, b, arg, y0, x1 = grab(r"dy/dx = \((\S+)·x² \+ (\S+)\)·e\^\((.+?)\), y\(0\) = (\S+)\. Find y\((\S+)\)", p)
        c = F(E(arg).subs(y, 1))
        a, b, y0, x1 = (F(v) for v in (a, b, y0, x1))
        sol = mp.odefun(lambda xx, Y: [(a * xx * xx + b) * mp.exp(c * Y[0])], 0, [y0])
        return sol(x1)[0]
    # linear: integrating factor x^a
    a, bb, y0, x1 = (F(v) for v in grab(r"dy/dx \+ \((\S+)/x\)·y = (\S+)·sin\(x\), y\(1\) = (\S+), for x > 0\. Find y\((\S+)\)", p))
    return (y0 + mp.quad(lambda u: u ** a * bb * mp.sin(u), [1, x1])) / x1 ** a


def k_series(p):
    (body,) = grab(r"to ∞ of (.+?)\. Give", p)
    return mp.nsum(fn(E(body), n), [1, mp.inf])


def k_taylor(p):
    k, body = grab(r"coefficient of x\^(\d+) in the Maclaurin series of (.+?)\. Give", p)
    return sp.N(sp.series(E(body), x, 0, int(k) + 1).removeO().coeff(x, int(k)), 20)


def k_double(p):
    body, pp = grab(r"f\(x, y\) = (.+?) over the region R bounded by the parabola y = x² and the line y = (\S+)·x", p)
    P = E(pp)
    inner = sp.integrate(E(body), (y, x**2, P * x))
    return sp.N(sp.integrate(sp.expand(inner), (x, 0, P)), 20)


def k_lagrange(p):
    f, g, K = grab(r"f\(x, y, z\) = (.+?) over positive x, y, z subject to the constraint (.+?) = (\S+)\.", p)
    f, g = E(f), E(g)
    gx = [sp.diff(g, v) for v in (x, y, z)]
    fx = [sp.diff(f, v) for v in (x, y, z)]
    eqs = [fx[0] * gx[1] - fx[1] * gx[0], fx[1] * gx[2] - fx[2] * gx[1], g - E(K)]
    best = None
    for s in (1, 0.7, 1.4, 0.4):
        c0 = float(sp.sqrt(sp.Rational(E(K)) / 6)) * s
        try:
            sol = sp.nsolve(eqs, (x, y, z), (c0, c0 * 0.9, c0 * 1.1), prec=25)
        except Exception:
            continue
        if all(v > 0 for v in sol):
            val = f.subs({x: sol[0], y: sol[1], z: sol[2]})
            best = val if best is None else max(best, val)
    assert best is not None, "no positive stationary point"
    return sp.N(best, 20)


def k_newton(p):
    a, b, x0 = (Fraction(v) for v in grab(r"x³ − (\S+)x − (\S+) starting from x₀ = (\S+)\.", p))
    v = x0
    for _ in range(3):
        v -= (v**3 - a * v - b) / (3 * v * v - a)
    return v.numerator / v.denominator


KINDS = {k[2:]: v for k, v in list(globals().items()) if k.startswith("k_")}


def main():
    qs = json.load(open(sys.argv[1]))
    bad = 0
    for q in qs:
        kind = q["id"].rsplit("-", 1)[0]
        try:
            got = float(KINDS[kind](q["prompt"]))
            want = float(q["answer"])
            if not abs(got - want) <= q["tolerance"] / 5:
                bad += 1
                print(f"MISMATCH {q['id']}: python {got!r} vs answer {want!r} (tol {q['tolerance']})\n  {q['prompt']}")
        except Exception as e:  # a prompt the checker cannot read is a failure too
            bad += 1
            print(f"ERROR {q['id']}: {type(e).__name__}: {e}\n  {q['prompt']}")
    print(f"calculus: {len(qs) - bad}/{len(qs)} questions verified")
    sys.exit(1 if bad else 0)


main()
