package evals

import (
	"fmt"
	"math/big"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
)

// Hard math: AIME to light-olympiad counting, number theory, algebra and
// geometry problems with odd parameters, so the answers can't be recalled.
// verify/math-hard.py recomputes every answer from the prompt text, mostly
// by brute force.

func init() {
	registerGen(&GenSpec{
		ID: "math-hard", Name: "Hard math", Short: "Hard math",
		Description: "Generated competition-style problems with random numbers (counting, number theory, algebra, geometry); graded on the final number.",
		Seed:        1, PerGen: 5, MaxTokens: 16384,
		Gens: []Generator{
			{ID: "avoid-strings", Make: mhAvoidStrings},
			{ID: "lattice-paths", Make: mhLatticePaths},
			{ID: "exactly-divisible", Make: mhExactlyDivisible},
			{ID: "digit-sum", Make: mhDigitSum},
			{ID: "power-tower", Make: mhPowerTower},
			{ID: "square-roots-mod", Make: mhSquareRootsMod},
			{ID: "diophantine", Make: mhDiophantine},
			{ID: "random-walk", Make: mhRandomWalk},
			{ID: "dice-sum", Make: mhDiceSum},
			{ID: "recurrence", Make: mhRecurrence},
			{ID: "power-sums", Make: mhPowerSums},
			{ID: "lattice-triangle", Make: mhLatticeTriangle},
			{ID: "monotone-functions", Make: mhMonotoneFunctions},
			{ID: "shifted-derangements", Make: mhShiftedDerangements},
			{ID: "divisor-sum", Make: mhDivisorSum},
			{ID: "floor-sum", Make: mhFloorSum},
			{ID: "proper-strings", Make: mhProperStrings},
		},
	})
}

// mhIntQ is a question with an integer answer.
func mhIntQ(prompt string, ans int64) Question {
	return Question{Prompt: prompt, Answer: strconv.FormatInt(ans, 10), Tolerance: 0.5}
}

func mhList(xs []int, last string) string {
	s := make([]string, len(xs))
	for i, x := range xs {
		s[i] = strconv.Itoa(x)
	}
	if len(s) < 2 {
		return strings.Join(s, "")
	}
	return strings.Join(s[:len(s)-1], ", ") + " " + last + " " + s[len(s)-1]
}

func mhGCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func mhAbs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func mhAbsF(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func mhPow(b, e int) int {
	r := 1
	for ; e > 0; e-- {
		r *= b
	}
	return r
}

func mhBinom(n, k int) int64 {
	if k < 0 || k > n {
		return 0
	}
	return new(big.Int).Binomial(int64(n), int64(k)).Int64()
}

// mhSigned writes "+ 3" or "- 3" for a coefficient.
func mhSigned(c int) string {
	if c < 0 {
		return "- " + strconv.Itoa(-c)
	}
	return "+ " + strconv.Itoa(c)
}

func mhAvoidStrings(r *rand.Rand) (Question, bool) {
	k := pick(r, 3, 4)
	n := between(r, 10, 12)
	if k == 4 {
		n = between(r, 7, 9)
	}
	letters := "ABCD"[:k]
	word := func() string {
		b := make([]byte, 3)
		for i := range b {
			b[i] = letters[r.IntN(k)]
		}
		return string(b)
	}
	p1, p2 := word(), word()
	if p1 == p2 {
		return Question{}, false
	}
	// count by dynamic programming on the last two letters
	cur := map[string]int64{}
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			cur[string(letters[i])+string(letters[j])] = 1
		}
	}
	for step := 2; step < n; step++ {
		next := map[string]int64{}
		for s, c := range cur {
			for i := 0; i < k; i++ {
				w := s + string(letters[i])
				if w == p1 || w == p2 {
					continue
				}
				next[w[1:]] += c
			}
		}
		cur = next
	}
	var total int64
	for _, c := range cur {
		total += c
	}
	if total < 500 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("How many strings of length %d can be written using only the first %d letters of the alphabet (%s), with letters allowed to repeat, so that neither \"%s\" nor \"%s\" appears anywhere in the string as a block of three consecutive letters?", n, k, strings.Join(strings.Split(letters, ""), ", "), p1, p2), total), true
}

func mhLatticePaths(r *rand.Rand) (Question, bool) {
	a, b := between(r, 9, 13), between(r, 9, 13)
	d := between(r, 1, 3)
	if b > a+d {
		return Question{}, false
	}
	var blocked [][2]int
	for len(blocked) < 3 {
		x, y := between(r, 1, a), between(r, 1, b)
		if (x == a && y == b) || y > x+d || slices.Contains(blocked, [2]int{x, y}) {
			continue
		}
		blocked = append(blocked, [2]int{x, y})
	}
	ways := make([][]int64, a+1)
	for x := range ways {
		ways[x] = make([]int64, b+1)
	}
	for x := 0; x <= a; x++ {
		for y := 0; y <= b; y++ {
			if y > x+d || slices.Contains(blocked, [2]int{x, y}) {
				continue
			}
			if x == 0 && y == 0 {
				ways[x][y] = 1
				continue
			}
			if x > 0 {
				ways[x][y] += ways[x-1][y]
			}
			if y > 0 {
				ways[x][y] += ways[x][y-1]
			}
		}
	}
	ans := ways[a][b]
	if ans < 1000 || ans > 999999 {
		return Question{}, false
	}
	pts := make([]string, len(blocked))
	for i, p := range blocked {
		pts[i] = fmt.Sprintf("(%d, %d)", p[0], p[1])
	}
	return mhIntQ(fmt.Sprintf("A path goes from (0, 0) to (%d, %d) in unit steps, each step either one to the right or one up. Every point (x, y) the path visits, including the start and the end, must satisfy y <= x + %d, and the path must not visit any of the points %s, %s and %s. How many different paths are there?", a, b, d, pts[0], pts[1], pts[2]), ans), true
}

func mhExactlyDivisible(r *rand.Rand) (Question, bool) {
	var ns []int
	for len(ns) < 4 {
		x := between(r, 4, 45)
		if !slices.Contains(ns, x) {
			ns = append(ns, x)
		}
	}
	slices.Sort(ns)
	e := between(r, 1, 2)
	N := between(r, 200000, 900000)
	// inclusion-exclusion: exactly e = sum_{j>=e} (-1)^(j-e) C(j,e) S_j,
	// where S_j adds floor(N / lcm) over all j-subsets
	S := make([]int64, 5)
	for mask := 1; mask < 16; mask++ {
		l, j := 1, 0
		for i := 0; i < 4; i++ {
			if mask>>i&1 == 1 {
				l = l / mhGCD(l, ns[i]) * ns[i]
				j++
			}
		}
		S[j] += int64(N / l)
	}
	var ans int64
	for j := e; j <= 4; j++ {
		t := mhBinom(j, e) * S[j]
		if (j-e)%2 == 1 {
			t = -t
		}
		ans += t
	}
	word := map[int]string{1: "one", 2: "two"}[e]
	return mhIntQ(fmt.Sprintf("How many integers from 1 to %d inclusive are divisible by exactly %s of the four numbers %s?", N, word, mhList(ns, "and")), ans), true
}

func mhDigitSum(r *rand.Rand) (Question, bool) {
	N := between(r, 200000, 999999)
	d := between(r, 1, 9)
	m := between(r, 7, 13)
	k := r.IntN(m)
	var cnt int64
	for n := 1; n <= N; n++ {
		s, ok := 0, true
		for x := n; x > 0; x /= 10 {
			if x%10 == d {
				ok = false
				break
			}
			s += x % 10
		}
		if ok && s%m == k {
			cnt++
		}
	}
	return mhIntQ(fmt.Sprintf("How many integers from 1 to %d inclusive have no digit equal to %d in their ordinary decimal notation, and have a digit sum that leaves remainder %d when divided by %d?", N, d, k, m), cnt), true
}

func mhPowerTower(r *rand.Rand) (Question, bool) {
	ps := []int{2, 3, 5, 7, 11, 13}
	m := 1
	for _, i := range r.Perm(len(ps))[:between(r, 2, 3)] {
		m *= ps[i]
	}
	m *= pick(r, 1, ps[r.IntN(3)])
	if m < 1000 || m > 99999 {
		return Question{}, false
	}
	a, b, c := between(r, 2, 99), between(r, 2, 30), between(r, 10, 40)
	e := new(big.Int).Exp(big.NewInt(int64(b)), big.NewInt(int64(c)), nil)
	ans := new(big.Int).Exp(big.NewInt(int64(a)), e, big.NewInt(int64(m))).Int64()
	if ans < 2 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("What is the remainder when %d^(%d^%d) is divided by %d? (The exponent is %d raised to the power %d, evaluated first.)", a, b, c, m, b, c), ans), true
}

func mhSquareRootsMod(r *rand.Rand) (Question, bool) {
	// prime powers, one per prime, for a modulus with 2 or 3 prime factors
	opts := [][]int{{4, 8, 16, 2}, {3, 9, 27}, {5, 25}, {7, 49}, {11, 121}, {13}, {17}, {19}, {23}, {29}}
	n := 1
	for _, i := range r.Perm(len(opts))[:between(r, 2, 3)] {
		n *= pick(r, opts[i]...)
	}
	if n < 1000 || n > 60000 {
		return Question{}, false
	}
	x0 := between(r, 2, n-1)
	a := x0 * x0 % n
	var sum int64
	cnt := 0
	for x := 0; x < n; x++ {
		if x*x%n == a {
			if 2*x <= n {
				sum += int64(x)
			}
			cnt++
		}
	}
	if cnt < 4 || a == 0 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("Find the sum of all integers x with 0 <= x <= %d/2 (that is, 2x <= %d) such that x^2 is congruent to %d modulo %d.", n, n, a, n), sum), true
}

func mhDiophantine(r *rand.Rand) (Question, bool) {
	c := [4]int{between(r, 2, 9), between(r, 3, 12), between(r, 4, 14), between(r, 5, 17)}
	u := [4]int{between(r, 8, 20), between(r, 8, 20), between(r, 8, 20), between(r, 8, 20)}
	mx := 0
	for i := range c {
		mx += c[i] * u[i]
	}
	N := between(r, mx/3, mx*2/3)
	var cnt int64
	for x := 0; x <= u[0]; x++ {
		for y := 0; y <= u[1]; y++ {
			for z := 0; z <= u[2]; z++ {
				rest := N - c[0]*x - c[1]*y - c[2]*z
				if rest >= 0 && rest%c[3] == 0 && rest/c[3] <= u[3] {
					cnt++
				}
			}
		}
	}
	if cnt < 100 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("How many ordered quadruples (x, y, z, w) of nonnegative integers satisfy %dx + %dy + %dz + %dw = %d, with x at most %d, y at most %d, z at most %d and w at most %d?", c[0], c[1], c[2], c[3], N, u[0], u[1], u[2], u[3]), cnt), true
}

func mhRandomWalk(r *rand.Rand) (Question, bool) {
	n := between(r, 6, 9)
	a, b := between(r, 1, 7), between(r, 1, 7)
	if a == b || a+b > 9 {
		return Question{}, false
	}
	s := between(r, 2, n-2)
	// expected turns E_i: (a+b) E_i - a E_(i+1) - b E_(i-1) = 10, E_0 = E_n = 0;
	// solved by Gaussian elimination
	m := n - 1
	A := make([][]float64, m)
	for i := range A {
		A[i] = make([]float64, m+1)
		A[i][i] = float64(a + b)
		A[i][m] = 10
		if i+1 < m {
			A[i][i+1] = -float64(a)
		}
		if i > 0 {
			A[i][i-1] = -float64(b)
		}
	}
	for c := 0; c < m; c++ {
		piv := c
		for i := c + 1; i < m; i++ {
			if mhAbsF(A[i][c]) > mhAbsF(A[piv][c]) {
				piv = i
			}
		}
		A[c], A[piv] = A[piv], A[c]
		for i := 0; i < m; i++ {
			if i != c {
				f := A[i][c] / A[c][c]
				for j := c; j <= m; j++ {
					A[i][j] -= f * A[c][j]
				}
			}
		}
	}
	e := A[s-1][m] / A[s-1][s-1]
	return Question{
		Prompt: fmt.Sprintf("A marker sits on one of the squares 0, 1, ..., %d and starts on square %d. Each turn, if it is on a square from 1 to %d, it moves one square up with probability %d/10, one square down with probability %d/10, and otherwise stays where it is (probability %d/10), independently of earlier turns. Once it reaches square 0 or square %d it stops for good. What is the expected number of turns until it stops? Give the answer rounded to 2 decimal places.", n, s, n-1, a, b, 10-a-b, n),
		Answer: strconv.FormatFloat(e, 'f', 2, 64), Tolerance: 0.0051,
	}, true
}

func mhDiceSum(r *rand.Rand) (Question, bool) {
	sides := between(r, 6, 9)
	faces := make([]int, 6)
	for i := range faces {
		faces[i] = between(r, 0, 12)
	}
	slices.Sort(faces)
	if faces[0] == faces[5] {
		return Question{}, false
	}
	poly := []int64{1}
	for i := 0; i < 5; i++ {
		next := make([]int64, len(poly)+sides)
		for t, c := range poly {
			for v := 1; v <= sides; v++ {
				next[t+v] += c
			}
		}
		poly = next
	}
	S := between(r, 5*(sides+1)/2, 5*(sides+1)/2+10)
	var cnt int64
	for _, f := range faces {
		if S-f >= 0 && S-f < len(poly) {
			cnt += poly[S-f]
		}
	}
	if cnt < 1000 {
		return Question{}, false
	}
	fs := make([]string, 6)
	for i, f := range faces {
		fs[i] = strconv.Itoa(f)
	}
	return mhIntQ(fmt.Sprintf("5 fair %d-sided dice (faces numbered 1 to %d) are rolled together with one special die with six equally likely faces. The faces of the special die show: %s. All the dice are distinguishable, and a value that appears twice on the special die counts as two different faces. Out of the %d equally likely outcomes, how many give a total of exactly %d?", sides, sides, strings.Join(fs, ", "), mhPow(sides, 5)*6, S), cnt), true
}

func mhRecurrence(r *rand.Rand) (Question, bool) {
	a0, a1 := between(r, 1, 9), between(r, 1, 9)
	p, q, c := between(r, 2, 9), between(r, -6, 9), between(r, -9, 9)
	if q == 0 || c == 0 {
		return Question{}, false
	}
	N := between(r, 30, 60)
	P := pick(r, 997, 1009, 10007, 100003)
	x, y := int64(a0), int64(a1)
	M := int64(P)
	for i := 2; i <= N; i++ {
		x, y = y, ((int64(p)*y+int64(q)*x+int64(c))%M+M)%M
	}
	return mhIntQ(fmt.Sprintf("Define a_0 = %d, a_1 = %d and a_n = %d*a_(n-1) %s*a_(n-2) %s for n >= 2. What is the remainder when a_%d is divided by %d? (The remainder is between 0 and %d.)", a0, a1, p, mhSigned(q), mhSigned(c), N, P, P-1), y), true
}

// mhPolyString writes a polynomial from its coefficients (co[i] is for x^i).
func mhPolyString(co []int) string {
	var sb strings.Builder
	for d := len(co) - 1; d >= 0; d-- {
		c := co[d]
		if c == 0 {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteString(" ")
			if c < 0 {
				sb.WriteString("- ")
			} else {
				sb.WriteString("+ ")
			}
		} else if c < 0 {
			sb.WriteString("-")
		}
		a := mhAbs(c)
		if a != 1 || d == 0 {
			sb.WriteString(strconv.Itoa(a))
		}
		if d >= 1 {
			sb.WriteString("x")
		}
		if d >= 2 {
			sb.WriteString("^" + strconv.Itoa(d))
		}
	}
	return sb.String()
}

func mhPowerSums(r *rand.Rand) (Question, bool) {
	deg := between(r, 3, 4)
	co := make([]int, deg+1)
	co[deg] = 1
	for i := 0; i < deg; i++ {
		co[i] = between(r, -9, 9)
	}
	if co[0] == 0 || co[deg-1] == 0 {
		return Question{}, false
	}
	K := between(r, 5, 8)
	// Newton's identities: p_k = e_1 p_(k-1) - e_2 p_(k-2) + ... (+/-) k e_k
	e := make([]int64, deg+1)
	for k := 1; k <= deg; k++ {
		e[k] = int64(co[deg-k])
		if k%2 == 1 {
			e[k] = -e[k]
		}
	}
	p := make([]int64, K+1)
	p[0] = int64(deg)
	for k := 1; k <= K; k++ {
		var s int64
		for i := 1; i <= deg && i <= k; i++ {
			term := e[i] * p[k-i]
			if i == k {
				term = e[i] * int64(k)
			}
			if i%2 == 0 {
				term = -term
			}
			s += term
		}
		p[k] = s
	}
	if v := mhAbs(int(p[K])); v < 200 || v > 5000000 {
		return Question{}, false
	}
	poly := strings.TrimPrefix(mhPolyString(co), "x^"+strconv.Itoa(deg)+" ")
	return mhIntQ(fmt.Sprintf("Let the %d roots of x^%d %s = 0, counted with multiplicity and including any complex ones, be r_1, ..., r_%d. Find r_1^%d + r_2^%d + ... + r_%d^%d.", deg, deg, poly, deg, K, K, deg, K), p[K]), true
}

func mhLatticeTriangle(r *rand.Rand) (Question, bool) {
	var v [3][2]int
	for i := range v {
		v[i] = [2]int{between(r, 0, 60), between(r, 0, 60)}
	}
	cross := mhAbs((v[1][0]-v[0][0])*(v[2][1]-v[0][1]) - (v[2][0]-v[0][0])*(v[1][1]-v[0][1]))
	bd := 0
	for i := 0; i < 3; i++ {
		j := (i + 1) % 3
		bd += mhGCD(mhAbs(v[i][0]-v[j][0]), mhAbs(v[i][1]-v[j][1]))
	}
	in := (cross - bd + 2) / 2 // Pick's theorem
	if in < 200 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("How many points with integer coordinates lie strictly inside the triangle with vertices (%d, %d), (%d, %d) and (%d, %d)? Points on the sides of the triangle do not count.", v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1]), int64(in)), true
}

func mhMonotoneFunctions(r *rand.Rand) (Question, bool) {
	n, m, t := between(r, 7, 9), between(r, 9, 12), between(r, 3, 5)
	// dp[last value][distinct values so far]
	newTable := func() [][]int64 {
		d := make([][]int64, m+1)
		for v := range d {
			d[v] = make([]int64, t+2)
		}
		return d
	}
	dp := newTable()
	for v := 2; v <= m; v++ { // f(1) != 1
		dp[v][1] = 1
	}
	for i := 2; i <= n; i++ {
		nx := newTable()
		for v := 1; v <= m; v++ {
			if v == i {
				continue
			}
			for w := 1; w <= v; w++ {
				for k := 1; k <= t; k++ {
					if w == v {
						nx[v][k] += dp[w][k]
					} else {
						nx[v][k+1] += dp[w][k]
					}
				}
			}
		}
		dp = nx
	}
	var cnt int64
	for v := 1; v <= m; v++ {
		cnt += dp[v][t]
	}
	if cnt < 100 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("How many functions f from {1, 2, ..., %d} to {1, 2, ..., %d} satisfy f(1) <= f(2) <= ... <= f(%d), f(i) != i for every i, and take exactly %d different values?", n, m, n, t), cnt), true
}

func mhShiftedDerangements(r *rand.Rand) (Question, bool) {
	n := between(r, 8, 9)
	var ds []int
	size := between(r, 2, 3)
	for len(ds) < size {
		d := r.IntN(n)
		if !slices.Contains(ds, d) {
			ds = append(ds, d)
		}
	}
	slices.Sort(ds)
	dp := make([]int64, 1<<n)
	dp[0] = 1
	for mask := 0; mask < 1<<n-1; mask++ {
		if dp[mask] == 0 {
			continue
		}
		i := 0
		for t := mask; t > 0; t &= t - 1 {
			i++
		}
		for v := 0; v < n; v++ {
			if mask>>v&1 == 0 && !slices.Contains(ds, ((v-i)%n+n)%n) {
				dp[mask|1<<v] += dp[mask]
			}
		}
	}
	cnt := dp[1<<n-1]
	if cnt < 500 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("In how many ways can the numbers 0, 1, ..., %d be arranged in a row, as a sequence p_0, p_1, ..., p_%d using each number exactly once, so that for no position i is p_i - i congruent modulo %d to any of the numbers %s?", n-1, n-1, n, mhList(ds, "or")), cnt), true
}

func mhDivisorSum(r *rand.Rand) (Question, bool) {
	N := 1
	for _, p := range []int{2, 3, 5, 7, 11} {
		N *= mhPow(p, between(r, 0, 5))
	}
	if N < 100000 || N > 5000000 {
		return Question{}, false
	}
	m := between(r, 5, 12)
	k := r.IntN(m)
	var sum int64
	for d := 1; d <= N; d++ {
		if N%d == 0 && d%m == k {
			sum += int64(d)
		}
	}
	if sum < 1000 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("Find the sum of all positive divisors d of %d (including 1 and %d itself) that leave remainder %d when divided by %d.", N, N, k, m), sum), true
}

func mhFloorSum(r *rand.Rand) (Question, bool) {
	a, b := between(r, 100, 999), between(r, 100, 999)
	c := between(r, 0, b)
	N := between(r, 100, 400)
	if a == b {
		return Question{}, false
	}
	var s int64
	for k := 1; k <= N; k++ {
		s += int64((a*k + c) / b)
	}
	return mhIntQ(fmt.Sprintf("Compute the sum of floor((%d*k + %d)/%d) for k = 1, 2, ..., %d, where floor(x) is the greatest integer not exceeding x.", a, c, b, N), s), true
}

func mhProperStrings(r *rand.Rand) (Question, bool) {
	n, k, t := between(r, 4, 7), between(r, 6, 9), between(r, 3, 5)
	// choose which t letters appear, then inclusion-exclusion over how many of
	// them may be used: j*(j-1)^(k-1) strings of length k with no equal neighbours
	var sum int64
	for j := 1; j <= t; j++ {
		term := mhBinom(t, j) * int64(j) * int64(mhPow(j-1, k-1))
		if (t-j)%2 == 1 {
			term = -term
		}
		sum += term
	}
	ans := mhBinom(n, t) * sum
	if ans < 100 {
		return Question{}, false
	}
	return mhIntQ(fmt.Sprintf("How many strings of length %d can be made from an alphabet of %d letters such that no two neighbouring letters are equal and exactly %d different letters appear in the string?", k, n, t), ans), true
}
