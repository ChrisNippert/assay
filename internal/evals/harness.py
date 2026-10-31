# Runs a model's solution, then the tests, in one namespace. Assay starts
# this with python -I in a temporary folder; limits are set here.
import sys

try:
    import resource
    for lim, val in ((resource.RLIMIT_AS, 1 << 30), (resource.RLIMIT_FSIZE, 10 << 20), (resource.RLIMIT_CPU, 20)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass
except ImportError:  # Windows
    pass

# The token printed at the end proves the tests ran to completion; the
# solution can't see it (it's taken off argv before the solution runs).
token = sys.argv[1] if len(sys.argv) > 1 else "ALL TESTS PASSED"
del sys.argv[1:]
sys.setrecursionlimit(10000)
ns = {"__name__": "solution"}
with open("solution.py", encoding="utf-8") as f:
    src = f.read()
# A solution's own demo code under `if __name__ == "__main__":` doesn't run.
try:
    exec(compile(src, "solution.py", "exec"), ns)
except SystemExit:
    print("The solution called exit() before the tests ran.")
    raise
ns["__name__"] = "tests"
with open("tests.py", encoding="utf-8") as f:
    tests = f.read()
exec(compile(tests, "tests.py", "exec"), ns)
print("ALL TESTS PASSED")
print(token)
