"""Independent checker for the generated "Hard math" set (gen_mathhard.go).

Usage: python3 -I verify/math-hard.py questions.json

Each question's parameters are read back out of the prompt text and the answer
is recomputed without using the Go code's method: brute force wherever the Go
code uses a formula or a DP, exact sympy arithmetic elsewhere.
"""
import json
import re
import sys
from fractions import Fraction
from itertools import combinations_with_replacement, permutations, product
from math import gcd

import sympy
from sympy import Matrix, Poly, divisors, symbols
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)


def ints(s):
    return [int(x) for x in re.findall(r"-?\d+", s)]


def avoid_strings(p):
    n = int(re.search(r"length (\d+)", p).group(1))
    k = int(re.search(r"first (\d+) letters", p).group(1))
    a, b = re.search(r'neither "([A-D]{3})" nor "([A-D]{3})"', p).groups()
    letters = "ABCD"[:k]
    cnt = 0
    for t in product(letters, repeat=n):
        s = "".join(t)
        if a not in s and b not in s:
            cnt += 1
    return cnt


def lattice_paths(p):
    pts = [tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", p)]
    a, b = pts[1]
    blocked = set(pts[2:])
    d = int(re.search(r"y <= x \+ (\d+)", p).group(1))
    assert pts[0] == (0, 0) and len(blocked) == 3
    # count backwards from the end by memoised recursion over the remaining path
    memo = {}

    def go(x, y):
        if x > a or y > b or y > x + d or (x, y) in blocked:
            return 0
        if (x, y) == (a, b):
            return 1
        if (x, y) not in memo:
            memo[(x, y)] = go(x + 1, y) + go(x, y + 1)
        return memo[(x, y)]

    return go(0, 0)


def exactly_divisible(p):
    N = int(re.search(r"from 1 to (\d+)", p).group(1))
    e = {"one": 1, "two": 2}[re.search(r"exactly (one|two)", p).group(1)]
    ns = ints(p.split("four numbers")[1])
    assert len(ns) == 4
    return sum(1 for n in range(1, N + 1) if sum(n % m == 0 for m in ns) == e)


def digit_sum(p):
    N, d, k, m = map(int, re.search(r"from 1 to (\d+) inclusive have no digit equal to (\d) .* remainder (\d+) when divided by (\d+)", p).groups())
    cnt = 0
    ds = str(d)
    for n in range(1, N + 1):
        s = str(n)
        if ds not in s and sum(map(int, s)) % m == k:
            cnt += 1
    return cnt


def power_tower(p):
    a, b, c, m = map(int, re.search(r"remainder when (\d+)\^\((\d+)\^(\d+)\) is divided by (\d+)", p).groups())
    return pow(a, b ** c, m)


def square_roots_mod(p):
    n, n1, a, n2 = map(int, re.search(r"0 <= x <= (\d+)/2 \(that is, 2x <= (\d+)\) such that x\^2 is congruent to (\d+) modulo (\d+)", p).groups())
    assert n == n1 == n2
    return sum(x for x in range(n + 1) if 2 * x <= n and (x * x - a) % n == 0)


def diophantine(p):
    c = list(map(int, re.search(r"satisfy (\d+)x \+ (\d+)y \+ (\d+)z \+ (\d+)w = (\d+)", p).groups()))
    N = c.pop()
    u = list(map(int, re.search(r"x at most (\d+), y at most (\d+), z at most (\d+) and w at most (\d+)", p).groups()))
    cnt = 0
    for x in range(u[0] + 1):
        for y in range(u[1] + 1):
            for z in range(u[2] + 1):
                for w in range(u[3] + 1):
                    if c[0] * x + c[1] * y + c[2] * z + c[3] * w == N:
                        cnt += 1
    return cnt


def random_walk(p):
    n = int(re.search(r"squares 0, 1, \.\.\., (\d+) and starts on square (\d+)", p).group(1))
    s = int(re.search(r"starts on square (\d+)", p).group(1))
    up, down, stay = map(int, re.search(r"up with probability (\d+)/10, one square down with probability (\d+)/10, and otherwise stays where it is \(probability (\d+)/10\)", p).groups())
    assert up + down + stay == 10
    # exact rational system (I - Q) E = 1 over the transient squares 1..n-1
    m = n - 1
    Q = sympy.zeros(m, m)
    for i in range(m):
        Q[i, i] = sympy.Rational(stay, 10)
        if i + 1 < m:
            Q[i, i + 1] = sympy.Rational(up, 10)
        if i > 0:
            Q[i, i - 1] = sympy.Rational(down, 10)
    E = (sympy.eye(m) - Q).LUsolve(Matrix([1] * m))
    return Fraction(int(E[s - 1].p), int(E[s - 1].q))


def dice_sum(p):
    m, sides = map(int, re.search(r"(\d+) fair (\d+)-sided dice", p).groups())
    faces = ints(re.search(r"faces of the special die show: ([\d, ]+)\.", p).group(1))
    S = int(re.search(r"total of exactly (\d+)", p).group(1))
    total = int(re.search(r"Out of the (\d+) equally", p).group(1))
    assert total == sides ** m * len(faces) == 6 * sides ** m
    cnt = 0
    for t in product(range(1, sides + 1), repeat=m):
        r = S - sum(t)
        cnt += faces.count(r)
    return cnt


def recurrence(p):
    a0, a1 = map(int, re.search(r"a_0 = (\d+), a_1 = (\d+)", p).groups())
    m = re.search(r"a_n = (-?\d+)\*a_\(n-1\) ([+-]) (\d+)\*a_\(n-2\) ([+-]) (\d+) for", p)
    pc = int(m.group(1))
    q = int(m.group(3)) * (1 if m.group(2) == "+" else -1)
    c = int(m.group(5)) * (1 if m.group(4) == "+" else -1)
    N, P = map(int, re.search(r"remainder when a_(\d+) is divided by (\d+)", p).groups())
    a = [a0, a1]
    for n in range(2, N + 1):  # exact big integers, reduced only at the end
        a.append(pc * a[-1] + q * a[-2] + c)
    return a[N] % P


def power_sums(p):
    m = re.search(r"roots of (.*) = 0, counted .* Find r_1\^(\d+)", p)
    x = symbols("x")
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    poly = Poly(parse_expr(m.group(1), transformations=tr, local_dict={"x": x}), x)
    K = int(m.group(2))
    co = poly.all_coeffs()
    assert co[0] == 1 and len(co) - 1 == int(re.search(r"the (\d+) roots", p).group(1))
    n = len(co) - 1
    # companion matrix: trace of C^K is the K-th power sum of the roots
    C = sympy.zeros(n, n)
    for i in range(1, n):
        C[i, i - 1] = 1
    for i in range(n):
        C[i, n - 1] = -co[n - i]
    return int((C ** K).trace())


def lattice_triangle(p):
    v = [tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", p)]
    assert len(v) == 3

    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    cnt = 0
    for x in range(61):
        for y in range(61):
            q = (x, y)
            s = [side(v[0], v[1], q), side(v[1], v[2], q), side(v[2], v[0], q)]
            if all(t > 0 for t in s) or all(t < 0 for t in s):
                cnt += 1
    return cnt


def monotone_functions(p):
    n, m = map(int, re.search(r"from \{1, 2, \.\.\., (\d+)\} to \{1, 2, \.\.\., (\d+)\}", p).groups())
    t = int(re.search(r"exactly (\d+) different values", p).group(1))
    cnt = 0
    for f in combinations_with_replacement(range(1, m + 1), n):  # sorted tuples = non-decreasing functions
        if len(set(f)) == t and all(f[i] != i + 1 for i in range(n)):
            cnt += 1
    return cnt


def shifted_derangements(p):
    n = int(re.search(r"modulo (\d+) to any", p).group(1))
    top = int(re.search(r"0, 1, \.\.\., (\d+) be arranged", p).group(1))
    assert top == n - 1
    ds = set(ints(p.split("any of the numbers")[1]))
    cnt = 0
    for perm in permutations(range(n)):
        if all((perm[i] - i) % n not in ds for i in range(n)):
            cnt += 1
    return cnt


def divisor_sum(p):
    N, k, m = map(int, re.search(r"divisors d of (\d+) \(including 1 and \d+ itself\) that leave remainder (\d+) when divided by (\d+)", p).groups())
    return sum(d for d in divisors(N) if d % m == k)


def floor_sum(p):
    a, c, b, N = map(int, re.search(r"floor\(\((\d+)\*k \+ (\d+)\)/(\d+)\) for k = 1, 2, \.\.\., (\d+),", p).groups())
    direct = sum((a * k + c) // b for k in range(1, N + 1))

    def fsum(n, m, a, b):  # sum_{i=0}^{n-1} floor((a*i+b)/m), Euclid-style
        ans = 0
        while True:
            if a >= m:
                ans += (n - 1) * n // 2 * (a // m)
                a %= m
            if b >= m:
                ans += n * (b // m)
                b %= m
            ym = a * n + b
            if ym < m:
                return ans
            n, b = ym // m, ym % m
            m, a = a, m

    # sum over k=1..N is the sum over i=0..N-1 with offset a + c
    assert fsum(N, b, a, a + c) == direct
    return direct


def proper_strings(p):
    k, n, t = map(int, re.search(r"strings of length (\d+) can be made from an alphabet of (\d+) letters .* exactly (\d+) different", p).groups())
    cnt = 0

    def go(pos, last, used):
        nonlocal cnt
        if pos == k:
            cnt += len(used) == t
            return
        for c in range(n):
            if c != last and (c in used or len(used) < t):
                go(pos + 1, c, used | {c})

    go(0, -1, frozenset())
    return cnt


CHECKS = {
    "avoid-strings": avoid_strings, "lattice-paths": lattice_paths,
    "exactly-divisible": exactly_divisible, "digit-sum": digit_sum,
    "power-tower": power_tower, "square-roots-mod": square_roots_mod,
    "diophantine": diophantine, "random-walk": random_walk,
    "dice-sum": dice_sum, "recurrence": recurrence, "power-sums": power_sums,
    "lattice-triangle": lattice_triangle, "monotone-functions": monotone_functions,
    "shifted-derangements": shifted_derangements, "divisor-sum": divisor_sum,
    "floor-sum": floor_sum, "proper-strings": proper_strings,
}


def main():
    qs = json.load(open(sys.argv[1]))
    bad = []
    seen = set()
    for q in qs:
        kind = q["id"].rsplit("-", 1)[0]
        seen.add(kind)
        fn = CHECKS.get(kind)
        if fn is None:
            bad.append(f"{q['id']}: no checker")
            continue
        try:
            want = fn(q["prompt"])
        except Exception as e:  # a prompt the checker can't read is a failure too
            bad.append(f"{q['id']}: cannot check ({type(e).__name__}: {e})")
            continue
        got = Fraction(q["answer"])
        tol = q.get("tolerance", 0)
        if kind == "random-walk":
            # the answer is the exact value rounded to 2 places
            ok = abs(got - want) <= Fraction(5, 1000) + Fraction(1, 10 ** 9) and abs(float(want) - float(got)) <= 0.005 + 1e-9
            ok = ok and abs(tol - 0.0051) < 1e-12
            if abs(want) < 5:
                bad.append(f"{q['id']}: expected value {float(want)} too small")
        else:
            ok = got == want and tol == 0.5 and want == int(want)
        if not ok:
            bad.append(f"{q['id']}: answer {q['answer']} (tolerance {tol}), recomputed {want if not isinstance(want, Fraction) else float(want)}")
    missing = set(CHECKS) - seen
    for m in sorted(missing):
        bad.append(f"{m}: no questions")
    if bad:
        print("\n".join(bad))
        print(f"math-hard: {len(bad)} problem(s) in {len(qs)} questions")
        sys.exit(1)
    print(f"math-hard: all {len(qs)} questions verified")


main()
