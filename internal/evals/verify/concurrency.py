"""Independent checker for the generated "concurrency" set.

Usage: python3 -E -P verify/concurrency.py questions.json

Each question's program is parsed back out of the fenced block in the prompt
text and the question line says what is asked (final values of a variable,
final pairs, or printed outputs). Every answer is recomputed with its own
interpreter: a breadth-first sweep over all reachable machine states (no
recursion, no shared code with the Go generator) and compared with the
stored answer. Exits 1 and prints every mismatch.
"""
import json
import re
import sys

ARITH = re.compile(r"^(\w+) = (\w+) ([+*-]) (\d+)$")
IF = re.compile(r"^if (\w+) (==|!=) (-?\d+): (.*)$")


def parse(prompt):
    block = re.search(r"```\n(.*?)```", prompt, re.S).group(1)
    shared, locks, threads = {}, [], []
    cur = None
    for raw in block.split("\n"):
        if not raw.strip():
            continue
        indent = (len(raw) - len(raw.lstrip(" "))) // 4
        line = raw.strip()
        if indent == 0:
            if line.startswith("shared "):
                for part in line[7:].split(", "):
                    k, v = part.split(" = ")
                    shared[k] = int(v)
            elif line.startswith("lock "):
                locks.append(line[5:])
            elif line.startswith("thread "):
                cur = (line[7:-1], [])
                threads.append(cur)
            else:
                raise ValueError("bad line " + raw)
            continue
        # statements: a stack of open repeat blocks, by indent
        body = cur[1]
        for _ in range(indent - 1):
            body = body[-1][2]
        m = re.match(r"^repeat (\d+) times:$", line)
        body.append(("repeat", int(m.group(1)), []) if m else tuple(["stmt", line]))
    names = [t[0] for t in threads]
    return shared, locks, names, [unroll(t[1], shared, names) for t in threads]


def stmt(line, shared, names):
    """One statement as a list of (guard, step) pairs; guard is None or
    (reg, equal, k). Steps are tuples starting with the kind."""
    m = IF.match(line)
    if m:
        g = (m.group(1), m.group(2) == "==", int(m.group(3)))
        return [(g, step) for _, step in stmt(m.group(4), shared, names)]
    m = re.match(r'^print\("(.*)"\)$', line)
    if m:
        return [(None, ("print", m.group(1)))]
    m = re.match(r"^(lock|unlock)\((\w+)\)$", line)
    if m:
        return [(None, (m.group(1), m.group(2)))]
    m = re.match(r"^join\((\w+)\)$", line)
    if m:
        return [(None, ("join", names.index(m.group(1))))]
    m = re.match(r"^wait_until\((\w+) == (-?\d+)\)$", line)
    if m:
        return [(None, ("wait", m.group(1), int(m.group(2))))]
    m = ARITH.match(line)
    if m:
        lhs, a, op, k = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if a in shared:  # x = x + 1: separate read then write
            assert a == lhs, line
            return [(None, ("load", "#tmp", a)), (None, ("store", lhs, ("#tmp", op, k)))]
        return [(None, ("store", lhs, (a, op, k)))]
    m = re.match(r"^(\w+) = (-?\w+)$", line)
    if m:
        lhs, rhs = m.groups()
        if lhs in shared:
            if re.fullmatch(r"-?\d+", rhs):
                return [(None, ("store", lhs, int(rhs)))]
            return [(None, ("store", lhs, (rhs, None, 0)))]
        assert rhs in shared, line
        return [(None, ("load", lhs, rhs))]
    raise ValueError("cannot parse " + line)


def unroll(body, shared, names):
    out = []
    for item in body:
        if item[0] == "repeat":
            for _ in range(item[1]):
                out += unroll(item[2], shared, names)
        else:
            out += stmt(item[1], shared, names)
    return out


def evaluate(e, regs):
    if isinstance(e, int):
        return e
    reg, op, k = e
    v = regs.get(reg, 0)
    return v if op is None else v + k if op == "+" else v - k if op == "-" else v * k


def explore(shared, locks, names, progs):
    """All final (shared values, output) pairs; raises on deadlock."""
    n = len(progs)
    init = (tuple([0] * n), tuple(sorted(shared.items())), (), "", tuple(() for _ in range(n)))
    # state: (pcs, shared items, held locks as ((lock, owner)...), output, regs per thread)
    frontier, seen, finals = [init], {init}, set()
    while frontier:
        nxt = []
        for pcs, sh, held, out, regs in frontier:
            if all(pcs[t] == len(progs[t]) for t in range(n)):
                finals.add((sh, out))
                continue
            moved = False
            sd, hd = dict(sh), dict(held)
            for t in range(n):
                if pcs[t] == len(progs[t]):
                    continue
                guard, step = progs[t][pcs[t]]
                kind = step[0]
                if kind == "lock" and step[1] in hd:
                    continue
                if kind == "join" and pcs[step[1]] != len(progs[step[1]]):
                    continue
                if kind == "wait" and sd[step[1]] != step[2]:
                    continue
                moved = True
                rg = dict(regs[t])
                sd2, hd2, out2 = dict(sd), dict(hd), out
                if guard is None or (rg.get(guard[0], 0) == guard[2]) == guard[1]:
                    if kind == "load":
                        rg[step[1]] = sd2[step[2]]
                    elif kind == "store":
                        sd2[step[1]] = evaluate(step[2], rg)
                    elif kind == "print":
                        out2 += step[1]
                    elif kind == "lock":
                        hd2[step[1]] = t
                    elif kind == "unlock":
                        hd2.pop(step[1], None)
                p2 = list(pcs)
                p2[t] += 1
                r2 = list(regs)
                r2[t] = tuple(sorted(rg.items()))
                s = (tuple(p2), tuple(sorted(sd2.items())), tuple(sorted(hd2.items())), out2, tuple(r2))
                if s not in seen:
                    seen.add(s)
                    nxt.append(s)
            if not moved:
                raise RuntimeError("deadlock")
        frontier = nxt
    return [(dict(sh), out) for sh, out in finals]


def answer(q):
    p = q["prompt"]
    shared, locks, names, progs = parse(p)
    finals = explore(shared, locks, names, progs)
    m = re.search(r"final values of `(\w+)`", p)
    if m:
        return ", ".join(str(v) for v in sorted({sh[m.group(1)] for sh, _ in finals}))
    m = re.search(r"final pairs `(\w+):(\w+)`", p)
    if m:
        pairs = sorted({(sh[m.group(1)], sh[m.group(2)]) for sh, _ in finals})
        return ", ".join("%d:%d" % pr for pr in pairs)
    if "possible outputs" in p:
        return ", ".join(sorted({o for _, o in finals}))
    raise ValueError("unknown question")


def main():
    qs = json.load(open(sys.argv[1]))
    bad = 0
    for q in qs:
        try:
            got = answer(q)
        except Exception as e:  # noqa: BLE001
            print("%s: checker error: %r" % (q["id"], e))
            bad += 1
            continue
        if got != q["answer"]:
            print("%s: stored %r, recomputed %r" % (q["id"], q["answer"], got))
            bad += 1
    print("%d questions, %d mismatches" % (len(qs), bad))
    sys.exit(1 if bad else 0)


main()
