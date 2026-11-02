package evals

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
)

// Concurrent outputs: small multi-threaded programs in pseudo-code under a
// sequentially consistent memory model. The answer is the set of every
// possible final value (or printed output), found by running every
// interleaving of the threads' steps. verify/concurrency.py parses the
// program back out of the prompt text and enumerates the interleavings
// with its own interpreter.

// cStmt is one pseudo-code statement.
type cStmt struct {
	// Op: read (Reg = Var), write (Var = expr), incr (Var = Var ExprOp K,
	// two steps), print (Lit), lock, unlock (Name), join (Name), wait
	// (Var == K), if (Reg ==/!= K: Inner), repeat (N times: Body).
	Op   string
	Reg  string
	Var  string
	Name string
	Lit  string
	// Expr for write: ExprReg "" means the constant K; ExprOp 0 means just
	// the register.
	ExprReg string
	ExprOp  byte
	K       int
	Eq      bool
	Inner   *cStmt
	N       int
	Body    []*cStmt
}

type cThread struct {
	Name string
	Body []*cStmt
}

type cShared struct {
	Name string
	Init int
}

type cProg struct {
	Shared  []cShared
	Locks   []string
	Threads []cThread
}

func (s *cStmt) text() string {
	switch s.Op {
	case "read":
		return s.Reg + " = " + s.Var
	case "write":
		switch {
		case s.ExprReg == "":
			return fmt.Sprintf("%s = %d", s.Var, s.K)
		case s.ExprOp == 0:
			return s.Var + " = " + s.ExprReg
		}
		return fmt.Sprintf("%s = %s %c %d", s.Var, s.ExprReg, s.ExprOp, s.K)
	case "incr":
		return fmt.Sprintf("%s = %s %c %d", s.Var, s.Var, s.ExprOp, s.K)
	case "print":
		return `print("` + s.Lit + `")`
	case "lock", "unlock", "join":
		return s.Op + "(" + s.Name + ")"
	case "wait":
		return fmt.Sprintf("wait_until(%s == %d)", s.Var, s.K)
	case "if":
		op := "!="
		if s.Eq {
			op = "=="
		}
		return fmt.Sprintf("if %s %s %d: %s", s.Reg, op, s.K, s.Inner.text())
	}
	panic("bad op " + s.Op)
}

func renderBody(b *strings.Builder, body []*cStmt, depth int) {
	pad := strings.Repeat("    ", depth)
	for _, s := range body {
		if s.Op == "repeat" {
			fmt.Fprintf(b, "%srepeat %d times:\n", pad, s.N)
			renderBody(b, s.Body, depth+1)
			continue
		}
		b.WriteString(pad + s.text() + "\n")
	}
}

// render writes the program as it appears in the prompt.
func (p *cProg) render() string {
	var b strings.Builder
	if len(p.Shared) > 0 {
		var sh []string
		for _, v := range p.Shared {
			sh = append(sh, fmt.Sprintf("%s = %d", v.Name, v.Init))
		}
		b.WriteString("shared " + strings.Join(sh, ", ") + "\n")
	}
	for _, l := range p.Locks {
		b.WriteString("lock " + l + "\n")
	}
	for _, t := range p.Threads {
		b.WriteString("thread " + t.Name + ":\n")
		renderBody(&b, t.Body, 1)
	}
	return b.String()
}

func (p *cProg) uses(f func(*cStmt) bool) bool {
	var walk func([]*cStmt) bool
	walk = func(body []*cStmt) bool {
		for _, s := range body {
			if f(s) || (s.Op == "repeat" && walk(s.Body)) {
				return true
			}
		}
		return false
	}
	return slices.ContainsFunc(p.Threads, func(t cThread) bool { return walk(t.Body) })
}

// micro is one indivisible step of a thread.
type micro struct {
	op       string // read write print lock unlock join wait
	reg, v   int    // register index, shared variable index
	name     int    // lock or thread index
	lit      string
	exprReg  int // -1: constant
	exprOp   byte
	k        int
	guardReg int // -1: none
	guardEq  bool
	guardK   int
}

// flatten unrolls loops and splits compound statements into steps; it also
// returns how many registers the thread uses.
func (p *cProg) flatten(ti int) ([]micro, int) {
	regs := map[string]int{}
	reg := func(n string) int {
		if _, ok := regs[n]; !ok {
			regs[n] = len(regs)
		}
		return regs[n]
	}
	idx := func(name string) int {
		for i, v := range p.Shared {
			if v.Name == name {
				return i
			}
		}
		panic("no variable " + name)
	}
	var out []micro
	temps := 0
	var emit func(s *cStmt, g *micro)
	emit = func(s *cStmt, g *micro) {
		m := micro{exprReg: -1, guardReg: -1}
		if g != nil {
			m.guardReg, m.guardEq, m.guardK = g.guardReg, g.guardEq, g.guardK
		}
		switch s.Op {
		case "read":
			m.op, m.reg, m.v = "read", reg(s.Reg), idx(s.Var)
		case "write":
			m.op, m.v, m.k, m.exprOp = "write", idx(s.Var), s.K, s.ExprOp
			if s.ExprReg != "" {
				m.exprReg = reg(s.ExprReg)
			}
		case "incr":
			temps++
			t := reg("#" + strconv.Itoa(temps))
			out = append(out, micro{op: "read", reg: t, v: idx(s.Var), exprReg: -1, guardReg: -1})
			m.op, m.v, m.k, m.exprOp, m.exprReg = "write", idx(s.Var), s.K, s.ExprOp, t
		case "print":
			m.op, m.lit = "print", s.Lit
		case "lock", "unlock":
			m.op, m.name = s.Op, slices.Index(p.Locks, s.Name)
		case "join":
			m.op = "join"
			for i, t := range p.Threads {
				if t.Name == s.Name {
					m.name = i
				}
			}
		case "wait":
			m.op, m.v, m.k = "wait", idx(s.Var), s.K
		case "if":
			emit(s.Inner, &micro{guardReg: reg(s.Reg), guardEq: s.Eq, guardK: s.K})
			return
		case "repeat":
			for i := 0; i < s.N; i++ {
				for _, c := range s.Body {
					emit(c, nil)
				}
			}
			return
		}
		out = append(out, m)
	}
	for _, s := range p.Threads[ti].Body {
		emit(s, nil)
	}
	return out, len(regs)
}

type cState struct {
	pc, regs, vars, holder []int
	out                    string
}

type cFinal struct {
	vars []int
	out  string
}

// run enumerates every interleaving (depth-first over the scheduler's
// choices, remembering states already seen) and returns the final states;
// an error means some schedule deadlocks.
func (p *cProg) run() (finals []cFinal, err error) {
	n := len(p.Threads)
	steps := make([][]micro, n)
	base := make([]int, n+1)
	for i := range steps {
		var nr int
		steps[i], nr = p.flatten(i)
		base[i+1] = base[i] + nr
	}
	st := cState{pc: make([]int, n), regs: make([]int, base[n]), vars: make([]int, len(p.Shared)), holder: make([]int, len(p.Locks))}
	for i, v := range p.Shared {
		st.vars[i] = v.Init
	}
	for i := range st.holder {
		st.holder[i] = -1
	}
	seen := map[string]bool{}
	var dfs func(s cState)
	dfs = func(s cState) {
		key := fmt.Sprint(s.pc, s.regs, s.vars, s.holder, s.out)
		if seen[key] {
			return
		}
		seen[key] = true
		done, moved := 0, false
		for t := 0; t < n; t++ {
			if s.pc[t] == len(steps[t]) {
				done++
				continue
			}
			m := steps[t][s.pc[t]]
			switch {
			case m.op == "lock" && s.holder[m.name] != -1,
				m.op == "join" && s.pc[m.name] != len(steps[m.name]),
				m.op == "wait" && s.vars[m.v] != m.k:
				continue
			}
			moved = true
			c := cState{pc: slices.Clone(s.pc), regs: slices.Clone(s.regs), vars: slices.Clone(s.vars), holder: slices.Clone(s.holder), out: s.out}
			c.pc[t]++
			r := func(i int) *int { return &c.regs[base[t]+i] }
			if m.guardReg < 0 || (*r(m.guardReg) == m.guardK) == m.guardEq {
				switch m.op {
				case "read":
					*r(m.reg) = c.vars[m.v]
				case "write":
					val := m.k
					if m.exprReg >= 0 {
						val = *r(m.exprReg)
						switch m.exprOp {
						case '+':
							val += m.k
						case '-':
							val -= m.k
						case '*':
							val *= m.k
						}
					}
					c.vars[m.v] = val
				case "print":
					c.out += m.lit
				case "lock":
					c.holder[m.name] = t
				case "unlock":
					c.holder[m.name] = -1
				}
			}
			dfs(c)
		}
		if done == n {
			finals = append(finals, cFinal{vars: s.vars, out: s.out})
		} else if !moved {
			err = fmt.Errorf("deadlock")
		}
	}
	dfs(st)
	return finals, err
}

// query kinds
const (
	qVar  = "var"
	qOut  = "out"
	qPair = "pair"
)

// answers returns the sorted distinct answers for a query, as strings.
func (p *cProg) answers(kind string, vars ...string) ([]string, error) {
	fin, err := p.run()
	if err != nil {
		return nil, err
	}
	vi := func(name string) int {
		return slices.IndexFunc(p.Shared, func(v cShared) bool { return v.Name == name })
	}
	var nums [][2]int
	var strs []string
	for _, f := range fin {
		switch kind {
		case qVar:
			nums = append(nums, [2]int{f.vars[vi(vars[0])], 0})
		case qPair:
			nums = append(nums, [2]int{f.vars[vi(vars[0])], f.vars[vi(vars[1])]})
		case qOut:
			strs = append(strs, f.out)
		}
	}
	if kind == qOut {
		slices.Sort(strs)
		return slices.Compact(strs), nil
	}
	slices.SortFunc(nums, func(a, b [2]int) int {
		if a[0] != b[0] {
			return a[0] - b[0]
		}
		return a[1] - b[1]
	})
	var res []string
	for _, v := range slices.Compact(nums) {
		if kind == qVar {
			res = append(res, strconv.Itoa(v[0]))
		} else {
			res = append(res, fmt.Sprintf("%d:%d", v[0], v[1]))
		}
	}
	return res, nil
}

// prompt words the question for a program: the rules mentioned are the
// ones the program uses.
func (p *cProg) prompt(kind string, vars ...string) string {
	var b strings.Builder
	b.WriteString("Here is a small concurrent program. All of its threads start together and each runs until it finishes.\n\n```\n")
	b.WriteString(p.render())
	b.WriteString("```\n\nRules:\n")
	if len(p.Shared) > 0 {
		b.WriteString("- The variables after `shared` are shared integers with the initial values shown. Any other name (such as `r`) is a register that belongs to the one thread using it and starts at 0; threads cannot see each other's registers.\n")
	}
	b.WriteString("- Each thread runs its statements in the order written; the threads' steps are interleaved in any possible way (sequential consistency: no other reordering, no caching).\n")
	if len(p.Shared) > 0 {
		b.WriteString("- Every statement is one indivisible step, so a statement such as `r = x` (read) or `x = r + 3` (write) cannot be interrupted.\n")
	}
	if p.uses(func(s *cStmt) bool { return s.Op == "incr" }) {
		b.WriteString("- A statement with the shared variable on both sides, such as `x = x + 2` or `x = x * 3`, is NOT indivisible: it is two steps, first reading x into a hidden register, then, as a separate step, writing the computed value back to x. Other threads' steps can run in between.\n")
	}
	if p.uses(func(s *cStmt) bool { return s.Op == "repeat" }) {
		b.WriteString("- `repeat N times:` runs the indented statements N times in a row (the statements inside are separate steps as usual).\n")
	}
	if p.uses(func(s *cStmt) bool { return s.Op == "if" }) {
		b.WriteString("- `if r == 0: statement` tests only the thread's own register; if the test is false the statement is skipped. The test itself is not a separate step.\n")
	}
	if p.uses(func(s *cStmt) bool { return s.Op == "print" }) {
		b.WriteString("- `print(\"A\")` is one indivisible step that appends its text to one shared output string.\n")
	}
	if len(p.Locks) > 0 {
		b.WriteString("- `lock(m)` is one step that waits until nobody holds the lock m and then takes it; `unlock(m)` releases it. A thread waiting for a held lock is blocked; every other thread can still run.\n")
	}
	if p.uses(func(s *cStmt) bool { return s.Op == "join" }) {
		b.WriteString("- `join(T)` waits until thread T has finished all its statements.\n")
	}
	if p.uses(func(s *cStmt) bool { return s.Op == "wait" }) {
		b.WriteString("- `wait_until(f == 1)` repeatedly reads the shared variable and continues once it sees the stated value; treat it as one step that can only happen at a moment when the variable has that value.\n")
	}
	b.WriteString("\n")
	switch kind {
	case qVar:
		fmt.Fprintf(&b, "What are all the possible final values of `%s` once every thread has finished? Give them in increasing order.", vars[0])
	case qPair:
		fmt.Fprintf(&b, "What are all the possible final pairs `%s:%s` (the values of `%s` and `%s` once every thread has finished, written like `3:4`)? Give them ordered by the first value, then the second, numerically.", vars[0], vars[1], vars[0], vars[1])
	case qOut:
		b.WriteString("The program's output is the complete string that has been printed once every thread has finished. What are all the possible outputs? Give each one as a string such as `AB`, in alphabetical order.")
	}
	return b.String()
}

// cMake builds the question; ok is false when the program deadlocks or
// has too few or too many answers.
func cMake(p *cProg, kind string, vars ...string) (Question, bool) {
	ans, err := p.answers(kind, vars...)
	if err != nil || len(ans) < 2 || len(ans) > 12 {
		return Question{}, false
	}
	return Question{Prompt: p.prompt(kind, vars...), Answer: strings.Join(ans, ", ")}, true
}

func cThreadNames(r *rand.Rand, n int) []string {
	set := pick(r, []string{"A", "B", "C"}, []string{"P", "Q", "R"}, []string{"T1", "T2", "T3"}, []string{"W1", "W2", "W3"}, []string{"U", "V", "W"})
	return set[:n]
}

func cShuffled(r *rand.Rand, n int, all ...string) []string {
	r.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	return all[:n]
}

func cVars(r *rand.Rand, n int) []string {
	return cShuffled(r, n, "x", "y", "z", "count", "total", "n", "hits", "balance", "score", "tally")
}

func cRegs(r *rand.Rand, n int) []string { return cShuffled(r, n, "r", "s", "t", "u", "w") }

func cLetters(r *rand.Rand, n int) []string {
	return cShuffled(r, n, "A", "B", "C", "D", "E", "H", "K", "M", "N", "X", "Y", "Z")
}

// rmw is a read-modify-write of v: the compound form (two hidden steps) or
// the written-out read and write.
func rmw(v, reg string, op byte, k int, compound bool) []*cStmt {
	if compound {
		return []*cStmt{{Op: "incr", Var: v, ExprOp: op, K: k}}
	}
	return []*cStmt{{Op: "read", Reg: reg, Var: v}, {Op: "write", Var: v, ExprReg: reg, ExprOp: op, K: k}}
}

// cOp picks +k (mostly) or *k.
func cOp(r *rand.Rand) (byte, int) {
	if r.IntN(4) == 0 {
		return '*', between(r, 2, 3)
	}
	return '+', between(r, 1, 6)
}

func cWrite(v string, k int) *cStmt { return &cStmt{Op: "write", Var: v, K: k} }
func cPrint(l string) *cStmt        { return &cStmt{Op: "print", Lit: l} }
func cLock(m string) *cStmt         { return &cStmt{Op: "lock", Name: m} }
func cUnlock(m string) *cStmt       { return &cStmt{Op: "unlock", Name: m} }

func init() {
	registerGen(&GenSpec{
		ID: "concurrency", Name: "Concurrent outputs", Short: "Threads",
		Description: "Small multi-threaded programs under a sequentially consistent memory model (lost updates, locks that only partly protect, check-then-act, spin-waits): list every possible final value or printed output. Answers come from enumerating every interleaving in code, cross-checked by an independent Python interpreter.",
		Check:       CheckSet, Seed: 3, PerGen: 5, MaxTokens: 16384,
		Gens: []Generator{
			// easy: two threads, two or three steps each
			{ID: "race2", Make: func(r *rand.Rand) (Question, bool) {
				x, nm, rg := cVars(r, 1)[0], cThreadNames(r, 2), cRegs(r, 2)
				o1, k1 := cOp(r)
				o2, k2 := cOp(r)
				c := r.IntN(2) == 0
				p := &cProg{Shared: []cShared{{x, between(r, 0, 3)}}, Threads: []cThread{
					{nm[0], rmw(x, rg[0], o1, k1, c)}, {nm[1], rmw(x, rg[1], o2, k2, c)}}}
				return cMake(p, qVar, x)
			}},
			{ID: "print2", Make: func(r *rand.Rand) (Question, bool) {
				nm, l := cThreadNames(r, 2), cLetters(r, 4)
				if r.IntN(2) == 0 {
					l[2] = l[1]
				}
				p := &cProg{Threads: []cThread{
					{nm[0], []*cStmt{cPrint(l[0]), cPrint(l[1])}}, {nm[1], []*cStmt{cPrint(l[2]), cPrint(l[3])}}}}
				return cMake(p, qOut)
			}},
			{ID: "overwrite", Make: func(r *rand.Rand) (Question, bool) {
				x, nm, rg := cVars(r, 1)[0], cThreadNames(r, 2), cRegs(r, 1)
				o, k := cOp(r)
				body := append([]*cStmt{cWrite(x, between(r, 1, 5))}, rmw(x, rg[0], o, k, r.IntN(2) == 0)...)
				p := &cProg{Shared: []cShared{{x, 0}}, Threads: []cThread{
					{nm[0], body}, {nm[1], []*cStmt{cWrite(x, between(r, 6, 9))}}}}
				return cMake(p, qVar, x)
			}},
			// medium: three threads, loops, join, a lock released in the middle
			{ID: "race3", Make: func(r *rand.Rand) (Question, bool) {
				x, nm, rg := cVars(r, 1)[0], cThreadNames(r, 3), cRegs(r, 3)
				p := &cProg{Shared: []cShared{{x, between(r, 0, 4)}}}
				c := r.IntN(2) == 0
				for i := 0; i < 3; i++ {
					o, k := cOp(r)
					p.Threads = append(p.Threads, cThread{nm[i], rmw(x, rg[i], o, k, c)})
				}
				return cMake(p, qVar, x)
			}},
			{ID: "loop2", Make: func(r *rand.Rand) (Question, bool) {
				x, nm, rg := cVars(r, 1)[0], cThreadNames(r, 2), cRegs(r, 2)
				k1, k2 := between(r, 1, 4), between(r, 1, 4)
				c := r.IntN(2) == 0
				t1 := []*cStmt{{Op: "repeat", N: 2, Body: rmw(x, rg[0], '+', k1, c)}}
				var t2 []*cStmt
				if r.IntN(2) == 0 {
					t2 = []*cStmt{{Op: "repeat", N: 2, Body: rmw(x, rg[1], '+', k2, c)}}
				} else {
					t2 = rmw(x, rg[1], '*', 2, c)
				}
				p := &cProg{Shared: []cShared{{x, 0}}, Threads: []cThread{{nm[0], t1}, {nm[1], t2}}}
				return cMake(p, qVar, x)
			}},
			{ID: "print3", Make: func(r *rand.Rand) (Question, bool) {
				nm, l := cThreadNames(r, 3), cLetters(r, 5)
				p := &cProg{Threads: []cThread{
					{nm[0], []*cStmt{cPrint(l[0]), cPrint(l[1])}},
					{nm[1], []*cStmt{cPrint(l[2]), cPrint(l[3])}},
					{nm[2], []*cStmt{{Op: "join", Name: nm[r.IntN(2)]}, cPrint(l[4])}}}}
				return cMake(p, qOut)
			}},
			{ID: "splitlock", Make: func(r *rand.Rand) (Question, bool) {
				x, nm, rg := cVars(r, 1)[0], cThreadNames(r, 2), cRegs(r, 2)
				m := pick(r, "m", "mu", "lk")
				split := func(reg string, k int) []*cStmt {
					return []*cStmt{cLock(m), {Op: "read", Reg: reg, Var: x}, cUnlock(m),
						cLock(m), {Op: "write", Var: x, ExprReg: reg, ExprOp: '+', K: k}, cUnlock(m)}
				}
				whole := func(reg string, k int) []*cStmt {
					return append(append([]*cStmt{cLock(m)}, rmw(x, reg, '+', k, false)...), cUnlock(m))
				}
				t1, t2 := split(rg[0], between(r, 1, 5)), whole(rg[1], between(r, 1, 5))
				if r.IntN(3) == 0 {
					t2 = split(rg[1], between(r, 1, 5))
				}
				p := &cProg{Shared: []cShared{{x, between(r, 0, 3)}}, Locks: []string{m}, Threads: []cThread{{nm[0], t1}, {nm[1], t2}}}
				return cMake(p, qVar, x)
			}},
			// hard: partial locking, check-then-act, spin-waits, two variables
			{ID: "partial-lock", Make: func(r *rand.Rand) (Question, bool) {
				x, nm, rg := cVars(r, 1)[0], cThreadNames(r, 3), cRegs(r, 3)
				m := pick(r, "m", "mu", "lk")
				p := &cProg{Shared: []cShared{{x, between(r, 0, 3)}}, Locks: []string{m}}
				var locked [3]bool
				for i := range locked {
					locked[i] = r.IntN(2) == 0
				}
				if locked[0] == locked[1] && locked[1] == locked[2] {
					locked[r.IntN(3)] = !locked[0]
				}
				for i := 0; i < 3; i++ {
					o, k := cOp(r)
					body := rmw(x, rg[i], o, k, false)
					if locked[i] {
						body = append(append([]*cStmt{cLock(m)}, body...), cUnlock(m))
					}
					p.Threads = append(p.Threads, cThread{nm[i], body})
				}
				return cMake(p, qVar, x)
			}},
			{ID: "check-act", Make: func(r *rand.Rand) (Question, bool) {
				n := between(r, 2, 3)
				v, nm, rg, l := cVars(r, 1)[0], cThreadNames(r, n), cRegs(r, n), cLetters(r, 2*n)
				p := &cProg{Shared: []cShared{{v, 0}}}
				for i := 0; i < n; i++ {
					body := []*cStmt{{Op: "read", Reg: rg[i], Var: v},
						{Op: "if", Reg: rg[i], Eq: true, K: 0, Inner: cWrite(v, i+1)},
						{Op: "if", Reg: rg[i], Eq: true, K: 0, Inner: cPrint(l[2*i])}}
					if r.IntN(2) == 0 {
						body = append(body, &cStmt{Op: "if", Reg: rg[i], Eq: false, K: 0, Inner: cPrint(l[2*i+1])})
					}
					p.Threads = append(p.Threads, cThread{nm[i], body})
				}
				return cMake(p, qOut)
			}},
			{ID: "spin-flag", Make: func(r *rand.Rand) (Question, bool) {
				vs, nm, rg := cVars(r, 2), cThreadNames(r, 3), cRegs(r, 2)
				data, flag := vs[0], vs[1]
				o, k := cOp(r)
				var c []*cStmt
				if r.IntN(2) == 0 {
					c = rmw(data, rg[1], '*', 2, false)
				} else {
					c = rmw(data, rg[1], '+', between(r, 1, 5), true)
				}
				p := &cProg{Shared: []cShared{{data, between(r, 0, 2)}, {flag, 0}}, Threads: []cThread{
					{nm[0], []*cStmt{cWrite(data, between(r, 3, 6)), cWrite(flag, 1)}},
					{nm[1], append([]*cStmt{{Op: "wait", Var: flag, K: 1}}, rmw(data, rg[0], o, k, false)...)},
					{nm[2], c}}}
				return cMake(p, qVar, data)
			}},
			{ID: "two-vars", Make: func(r *rand.Rand) (Question, bool) {
				vs, nm, rg := cVars(r, 2), cThreadNames(r, 3), cRegs(r, 2)
				x, y := vs[0], vs[1]
				o1, k1 := cOp(r)
				m := pick(r, "m", "mu")
				p := &cProg{Shared: []cShared{{x, between(r, 0, 2)}, {y, between(r, 0, 2)}}, Locks: []string{m}, Threads: []cThread{
					{nm[0], []*cStmt{{Op: "read", Reg: rg[0], Var: x}, {Op: "write", Var: y, ExprReg: rg[0], ExprOp: o1, K: k1}}},
					{nm[1], []*cStmt{cLock(m), {Op: "read", Reg: rg[1], Var: y}, {Op: "write", Var: x, ExprReg: rg[1], ExprOp: '+', K: between(r, 1, 4)}, cUnlock(m)}},
					{nm[2], []*cStmt{cLock(m), cWrite(y, between(r, 3, 7)), cUnlock(m)}}}}
				return cMake(p, qPair, x, y)
			}},
		},
	})
}
