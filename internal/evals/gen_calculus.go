package evals

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"strings"
)

// The Calculus set: hard single-variable and multivariable calculus problems
// written by code from random numbers. Closed forms are used where they exist
// and careful numerics (Simpson, RK4, bisection) where they don't;
// verify/calculus.py recomputes every answer from the prompt text by another
// route.

const calcAsk = " Give the answer as a decimal to 4 significant figures."

func init() {
	registerGen(&GenSpec{
		ID: "calculus", Name: "Calculus", Short: "Calculus",
		Description: "Hard calculus problems written by code: integration techniques, limits, derivatives, optimisation, differential equations, series and multivariable problems, each with a computed numeric answer.",
		Seed:        1, PerGen: 5, RelTol: 0.001, MaxTokens: 16384,
		Gens: []Generator{
			{ID: "ibp", Make: calcIBP},
			{ID: "partial", Make: calcPartial},
			{ID: "trigsub", Make: calcTrigSub},
			{ID: "improper", Make: calcImproper},
			{ID: "limit", Make: calcLimit},
			{ID: "implicit", Make: calcImplicit},
			{ID: "related", Make: calcRelated},
			{ID: "optim", Make: calcOptim},
			{ID: "arclen", Make: calcArcLen},
			{ID: "revol", Make: calcRevol},
			{ID: "areacross", Make: calcAreaCross},
			{ID: "ode2", Make: calcODE2},
			{ID: "odefirst", Make: calcODEFirst},
			{ID: "series", Make: calcSeries},
			{ID: "taylor", Make: calcTaylor},
			{ID: "double", Make: calcDouble},
			{ID: "lagrange", Make: calcLagrange},
			{ID: "newton", Make: calcNewton},
		},
	})
}

// calcSimpson integrates f over [a, b] with 20000 Simpson intervals (error
// far below the 0.1% tolerance for the smooth integrands used here).
func calcSimpson(f func(float64) float64, a, b float64) float64 {
	const n = 20000
	h := (b - a) / n
	s := f(a) + f(b)
	for i := 1; i < n; i++ {
		w := 4.0
		if i%2 == 0 {
			w = 2
		}
		s += w * f(a+float64(i)*h)
	}
	return s * h / 3
}

// calcRK4 integrates y' = f(t, y) from t0 to t1 in the given number of steps.
func calcRK4(f func(t float64, y []float64) []float64, t0, t1 float64, y0 []float64, steps int) []float64 {
	h := (t1 - t0) / float64(steps)
	y := append([]float64(nil), y0...)
	tmp := make([]float64, len(y))
	add := func(a, b []float64, c float64) []float64 {
		for i := range tmp {
			tmp[i] = a[i] + c*b[i]
		}
		return tmp
	}
	for s := 0; s < steps; s++ {
		t := t0 + float64(s)*h
		k1 := f(t, y)
		k2 := f(t+h/2, add(y, k1, h/2))
		k3 := f(t+h/2, add(y, k2, h/2))
		k4 := f(t+h, add(y, k3, h))
		for i := range y {
			y[i] += h / 6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
		}
	}
	return y
}

// calcOK reports whether x is finite and has a size lo <= |x| <= hi.
func calcOK(x, lo, hi float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0) && math.Abs(x) >= lo && math.Abs(x) <= hi
}

// calcAx writes c·v the way a person would: x, -x, 2.5x.
func calcAx(c float64, v string) string {
	switch c {
	case 1:
		return v
	case -1:
		return "-" + v
	}
	return show(c) + v
}

// calcCo writes an integer coefficient with a trailing dot, or nothing for 1.
func calcCo(c int) string {
	if c == 1 {
		return ""
	}
	return fmt.Sprintf("%d·", c)
}

func calcFact(n int) float64 {
	f := 1.0
	for i := 2; i <= n; i++ {
		f *= float64(i)
	}
	return f
}

func calcIBP(r *rand.Rand) (Question, bool) {
	a := pick(r, -1.5, -1.0, -0.5, 0.0, 0.5)
	b := a + pick(r, 1.5, 2.0, 2.5, 3.0)
	var ans float64
	var p string
	if r.IntN(2) == 0 {
		n := between(r, 2, 3)
		c := pick(r, -2.0, -1.5, -0.5, 0.5, 1.5, 2.0)
		F := func(x float64) float64 {
			s := 0.0
			for k := 0; k <= n; k++ {
				s += math.Pow(-1, float64(k)) * calcFact(n) / calcFact(n-k) * math.Pow(x, float64(n-k)) / math.Pow(c, float64(k+1))
			}
			return math.Exp(c*x) * s
		}
		ans = F(b) - F(a)
		p = fmt.Sprintf("Evaluate the definite integral ∫ from %s to %s of x^%d·e^(%s) dx.", show(a), show(b), n, calcAx(c, "x"))
	} else {
		pp := pick(r, -1.0, -0.5, 0.5, 1.0)
		q := float64(between(r, 1, 4))
		z := complex(pp, q)
		F := func(x float64) float64 {
			return imag(cmplx.Exp(z*complex(x, 0)) * (complex(x, 0)/z - 1/(z*z)))
		}
		ans = F(b) - F(a)
		p = fmt.Sprintf("Evaluate the definite integral ∫ from %s to %s of x·e^(%s)·sin(%s) dx.", show(a), show(b), calcAx(pp, "x"), calcAx(q, "x"))
	}
	if !calcOK(ans, 0.05, 500) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcPartial(r *rand.Rand) (Question, bool) {
	P, Q := between(r, 1, 9), between(r, 1, 9)
	rr, s := between(r, 1, 3), between(r, 2, 9)
	B := between(r, 2, 6)
	f := func(x float64) float64 {
		return (float64(P)*x + float64(Q)) / (math.Pow(x+float64(rr), 2) * (x*x + float64(s)))
	}
	ans := calcSimpson(f, 0, float64(B))
	p := fmt.Sprintf("Evaluate ∫ from 0 to %d of (%dx + %d)/((x + %d)²·(x² + %d)) dx.", B, P, Q, rr, s)
	if !calcOK(ans, 0.02, 100) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcTrigSub(r *rand.Rand) (Question, bool) {
	var ans float64
	var p string
	switch r.IntN(3) {
	case 0:
		k := float64(between(r, 2, 12))
		a := pick(r, 0.0, 0.5, 1.0)
		b := a + pick(r, 2.0, 2.5, 3.0, 4.0)
		F := func(x float64) float64 { return x*math.Sqrt(x*x+k)/2 - k/2*math.Asinh(x/math.Sqrt(k)) }
		ans = F(b) - F(a)
		p = fmt.Sprintf("Evaluate ∫ from %s to %s of x²/√(x² + %s) dx.", show(a), show(b), show(k))
	case 1:
		c := float64(between(r, 2, 6))
		a := pick(r, 0.0, 0.5, 1.0)
		b := c - pick(r, 0.0, 0.5, 1.0)
		F := func(x float64) float64 {
			return x*(2*x*x-c*c)*math.Sqrt(c*c-x*x)/8 + math.Pow(c, 4)/8*math.Asin(x/c)
		}
		ans = F(b) - F(a)
		p = fmt.Sprintf("Evaluate ∫ from %s to %s of x²·√(%s − x²) dx.", show(a), show(b), show(c*c))
	default:
		k := float64(between(r, 2, 12))
		a := pick(r, 0.5, 1.0, 1.5)
		b := a + pick(r, 1.5, 2.0, 3.0)
		F := func(x float64) float64 { return -math.Sqrt(x*x+k) / (k * x) }
		ans = F(b) - F(a)
		p = fmt.Sprintf("Evaluate ∫ from %s to %s of 1/(x²·√(x² + %s)) dx.", show(a), show(b), show(k))
	}
	if !calcOK(ans, 0.02, 500) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcImproper(r *rand.Rand) (Question, bool) {
	var ans float64
	var p string
	if r.IntN(2) == 0 {
		m := between(r, 2, 3)
		a := pick(r, 1.0, 1.5, 2.0, 2.5)
		b := float64(between(r, 1, 4))
		z := complex(a, -b)
		v := complex(calcFact(m), 0) / cmplx.Pow(z, complex(float64(m+1), 0))
		ans = real(v)
		scale := calcFact(m) / math.Pow(a*a+b*b, float64(m+1)/2)
		if math.Abs(ans) < 0.1*scale {
			return Question{}, false
		}
		p = fmt.Sprintf("Evaluate the improper integral ∫ from 0 to ∞ of x^%d·e^(%s)·cos(%s) dx.", m, calcAx(-a, "x"), calcAx(b, "x"))
	} else {
		A, B := between(r, 2, 20), between(r, 2, 20)
		m := between(r, 1, 12)
		if A == B || m == A || m == B {
			return Question{}, false
		}
		al := (float64(m) - float64(A)) / (float64(B) - float64(A))
		be := (float64(B) - float64(m)) / (float64(B) - float64(A))
		ans = al*math.Pi/(2*math.Sqrt(float64(A))) + be*math.Pi/(2*math.Sqrt(float64(B)))
		p = fmt.Sprintf("Evaluate the improper integral ∫ from 0 to ∞ of (x² + %d)/((x² + %d)(x² + %d)) dx.", m, A, B)
	}
	if !calcOK(ans, 0.005, 500) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcLimit(r *rand.Rand) (Question, bool) {
	var ans float64
	var p string
	switch r.IntN(3) {
	case 0:
		a, b := float64(between(r, 2, 6)), float64(between(r, 2, 6))
		ans = a*b*b/2 - a*a*a/6
		p = fmt.Sprintf("Find lim as x→0 of (sin(%s) − %s·cos(%s))/x³.", calcAx(a, "x"), calcAx(a, "x"), calcAx(b, "x"))
	case 1:
		a, b := decimal(r, 0.5, 4, 0.5), decimal(r, 0.5, 4, 0.5)
		ans = (a*a - b*b) / 2
		p = fmt.Sprintf("Find lim as x→0 of (e^(%s)·cos(%s) − 1 − %s)/x².", calcAx(a, "x"), calcAx(b, "x"), calcAx(a, "x"))
	default:
		a, b := float64(between(r, 1, 5)), float64(between(r, 1, 5))
		if a == b {
			return Question{}, false
		}
		c := (a*a + b*b) / 2
		ans = (math.Pow(a, 4)+math.Pow(b, 4))/24 + a*a*b*b/4
		p = fmt.Sprintf("Find lim as x→0 of (cos(%s)·cos(%s) − 1 + %s·x²)/x⁴.", calcAx(a, "x"), calcAx(b, "x"), show(c))
	}
	if !calcOK(ans, 0.3, 1000) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcImplicit(r *rand.Rand) (Question, bool) {
	a, b := between(r, 2, 5), between(r, 1, 3)
	x0, y0 := between(r, -3, 3), between(r, -3, 3)
	x, y := float64(x0), float64(y0)
	c := x*x*x + float64(a)*x*y + float64(b)*y*y*y
	Fx := 3*x*x + float64(a)*y
	Fy := float64(a)*x + 3*float64(b)*y*y
	if math.Abs(Fy) < 1 {
		return Question{}, false
	}
	yp := -Fx / Fy
	ypp := -(6*x + 2*float64(a)*yp + 6*float64(b)*y*yp*yp) / Fy
	if !calcOK(ypp, 0.05, 1000) {
		return Question{}, false
	}
	p := fmt.Sprintf("The curve x³ + %d·x·y + %sy³ = %s passes through the point (%d, %d). Treating y as a function of x near that point, find the second derivative d²y/dx² there.", a, calcCo(b), show(c), x0, y0)
	return Question{Prompt: p + calcAsk, Answer: num(ypp)}, true
}

func calcRelated(r *rand.Rand) (Question, bool) {
	var ans float64
	var p string
	if r.IntN(2) == 0 {
		L := float64(between(r, 12, 30))
		s := decimal(r, 0.5, 3, 0.1)
		d := decimal(r, 2, L-2, 0.5)
		y := math.Sqrt(L*L - d*d)
		ans = s * (y*y - d*d) / (2 * d)
		p = fmt.Sprintf("A %s-ft ladder leans against a vertical wall. Its top slides down the wall at %s ft/s. At the instant when the bottom of the ladder is %s ft from the wall, how fast is the area of the right triangle formed by the ladder, the wall and the ground changing, in ft²/s? (Give a negative number if the area is decreasing.)", show(L), show(s), show(d))
	} else {
		R, H := decimal(r, 2, 6, 0.5), decimal(r, 4, 12, 0.5)
		h := decimal(r, 1, H-0.5, 0.5)
		k := decimal(r, 1, 4, 0.5)
		Q := math.Ceil((k*math.Sqrt(h)+decimal(r, 2, 10, 0.5))*10) / 10
		ans = (Q - k*math.Sqrt(h)) / (math.Pi * R * R * h * h / (H * H))
		p = fmt.Sprintf("A conical tank with its point down has top radius %s ft and height %s ft. Water is pumped in at a constant %s ft³/min while it leaks out at a rate of %s·√h ft³/min, where h is the current water depth in ft. When the depth is %s ft, how fast is the depth changing, in ft/min?", show(R), show(H), show(Q), show(k), show(h))
	}
	if !calcOK(ans, 0.05, 1000) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcOptim(r *rand.Rand) (Question, bool) {
	var ans float64
	var p string
	if r.IntN(2) == 0 {
		W, H := float64(between(r, 14, 40)), float64(between(r, 14, 40))
		if W == H {
			return Question{}, false
		}
		x := ((W + H) - math.Sqrt(W*W-W*H+H*H)) / 6
		ans = x * (W - 2*x) * (H - 2*x)
		p = fmt.Sprintf("An open box is made from a rectangular sheet of cardboard %s inches by %s inches by cutting equal squares from the four corners and folding up the sides. What is the largest volume the box can have, in cubic inches?", show(W), show(H))
	} else {
		V := float64(between(r, 300, 2000))
		ct := decimal(r, 0.5, 3, 0.1)
		cb := decimal(r, 0.5, 3, 0.1)
		cs := decimal(r, 0.2, 2, 0.1)
		rad := math.Cbrt(V * cs / (math.Pi * (ct + cb)))
		h := V / (math.Pi * rad * rad)
		ans = 2*math.Pi*rad*h*cs + math.Pi*rad*rad*(ct+cb)
		p = fmt.Sprintf("A closed cylindrical can must hold exactly %s cm³. The material for the top costs %s cents per cm², for the bottom %s cents per cm² and for the curved side %s cents per cm². What is the smallest possible material cost of the can, in cents?", show(V), show(ct), show(cb), show(cs))
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcArcLen(r *rand.Rand) (Question, bool) {
	pp := decimal(r, 0.1, 0.9, 0.1)
	q := decimal(r, 0.5, 2.5, 0.1)
	w := float64(between(r, 1, 3))
	L := float64(between(r, 2, 5))
	f := func(x float64) float64 {
		d := 2*pp*x - q*w*math.Sin(w*x)
		return math.Sqrt(1 + d*d)
	}
	ans := calcSimpson(f, 0, L)
	p := fmt.Sprintf("Find the length of the curve y = %s·x² + %s·cos(%s) from x = 0 to x = %s.", show(pp), show(q), calcAx(w, "x"), show(L))
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcRevol(r *rand.Rand) (Question, bool) {
	a := decimal(r, 1, 3, 0.5)
	b := decimal(r, 1, 4, 0.5)
	k := decimal(r, 0.5, 3, 0.5)
	xc := math.Pow(a*b, 2.0/3)
	f := func(x float64) float64 {
		u, l := a*math.Sqrt(x)+k, x*x/b+k
		return math.Pi * (u*u - l*l)
	}
	ans := calcSimpson(f, 0, xc)
	if !calcOK(ans, 1, 2000) {
		return Question{}, false
	}
	p := fmt.Sprintf("The region in the first quadrant between the curves y = %s·√x and y = x²/%s (which meet at a point with x > 0) is revolved about the horizontal line y = -%s. Find the volume of the solid.", show(a), show(b), show(k))
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcAreaCross(r *rand.Rand) (Question, bool) {
	q := decimal(r, 3, 9, 0.5)
	rr := decimal(r, 0.5, 2, 0.1)
	pp := decimal(r, 0.5, 3, 0.5)
	if pp >= q {
		return Question{}, false
	}
	g := func(x float64) float64 { return q - rr*x*x - pp*math.Exp(x/2) }
	var roots []float64
	const step = 0.01
	for x := -10.0; x < 10; x += step {
		if (g(x) > 0) != (g(x+step) > 0) {
			lo, hi := x, x+step
			for i := 0; i < 60; i++ {
				m := (lo + hi) / 2
				if (g(m) > 0) == (g(lo) > 0) {
					lo = m
				} else {
					hi = m
				}
			}
			roots = append(roots, (lo+hi)/2)
		}
	}
	if len(roots) != 2 {
		return Question{}, false
	}
	ans := calcSimpson(g, roots[0], roots[1])
	p := fmt.Sprintf("Find the area of the region enclosed by the curves y = %s − %s·x² and y = %s·e^(x/2). (The curves cross at two points; you must find them numerically.)", show(q), show(rr), show(pp))
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcODE2(r *rand.Rand) (Question, bool) {
	b := decimal(r, 0.2, 2, 0.1)
	c := decimal(r, 1, 9, 0.5)
	F := decimal(r, 1, 6, 0.5)
	w := decimal(r, 0.5, 3, 0.5)
	y0, v0 := decimal(r, -2, 2, 0.5), decimal(r, -3, 3, 0.5)
	t1 := decimal(r, 1.5, 4, 0.5)
	f := func(t float64, y []float64) []float64 {
		return []float64{y[1], F*math.Cos(w*t) - b*y[1] - c*y[0]}
	}
	ans := calcRK4(f, 0, t1, []float64{y0, v0}, 40000)[0]
	if !calcOK(ans, 0.05, 100) {
		return Question{}, false
	}
	p := fmt.Sprintf("Solve the initial value problem y'' + %s·y' + %s·y = %s·cos(%s), with y(0) = %s and y'(0) = %s. Find y(%s).", show(b), show(c), show(F), calcAx(w, "t"), show(y0), show(v0), show(t1))
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcODEFirst(r *rand.Rand) (Question, bool) {
	var ans float64
	var p string
	if r.IntN(2) == 0 {
		a, b := decimal(r, 0.5, 3, 0.5), decimal(r, 0.5, 3, 0.5)
		c := pick(r, 0.5, 1.5, 2.0, 3.0)
		y0 := decimal(r, -1, 1, 0.5)
		x1 := pick(r, 1.0, 1.5, 2.0, 2.5)
		F := a*x1*x1*x1/3 + b*x1
		ans = math.Log(math.Exp(c*y0)+c*F) / c
		p = fmt.Sprintf("Solve the initial value problem dy/dx = (%s·x² + %s)·e^(%s), y(0) = %s. Find y(%s).", show(a), show(b), calcAx(-c, "y"), show(y0), show(x1))
	} else {
		a := float64(between(r, 1, 4))
		bb := decimal(r, 1, 5, 0.5)
		y0 := decimal(r, -2, 3, 0.5)
		x1 := pick(r, 2.0, 2.5, 3.0, 4.0)
		f := func(x float64, y []float64) []float64 { return []float64{-a/x*y[0] + bb*math.Sin(x)} }
		ans = calcRK4(f, 1, x1, []float64{y0}, 20000)[0]
		p = fmt.Sprintf("Solve the initial value problem dy/dx + (%s/x)·y = %s·sin(x), y(1) = %s, for x > 0. Find y(%s).", show(a), show(bb), show(y0), show(x1))
	}
	if !calcOK(ans, 0.05, 100) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcSeries(r *rand.Rand) (Question, bool) {
	var ans float64
	var p string
	if r.IntN(2) == 0 {
		a := pick(r, 2.0, 3.0, 4.0, 5.0, 2.5, 1.5)
		b, c := between(r, 1, 5), between(r, 1, 5)
		x := 1 / a
		s0 := x / (1 - x)
		s1 := x / ((1 - x) * (1 - x))
		s2 := x * (1 + x) / math.Pow(1-x, 3)
		ans = s2 + float64(b)*s1 + float64(c)*s0
		p = fmt.Sprintf("Find the sum of the infinite series Σ from n = 1 to ∞ of (n² + %sn + %d)/%s^n.", calcNNN(b), c, show(a))
	} else {
		rs := []int{between(r, 0, 7), between(r, 0, 7), between(r, 0, 7)}
		if rs[0] == rs[1] || rs[0] == rs[2] || rs[1] == rs[2] {
			return Question{}, false
		}
		sum := 0.0
		for i, ri := range rs {
			A := 1.0
			for j, rj := range rs {
				if j != i {
					A /= float64(rj - ri)
				}
			}
			h := 0.0
			for k := 1; k <= ri; k++ {
				h += 1 / float64(k)
			}
			sum -= A * h
		}
		ans = sum
		fs := make([]string, 3)
		for i, ri := range rs {
			if ri == 0 {
				fs[i] = "n"
			} else {
				fs[i] = fmt.Sprintf("(n + %d)", ri)
			}
		}
		p = "Find the sum of the infinite series Σ from n = 1 to ∞ of 1/(" + strings.Join(fs, "·") + ")."
	}
	if !calcOK(ans, 0.01, 1000) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcTaylor(r *rand.Rand) (Question, bool) {
	n := between(r, 4, 7)
	a := float64(between(r, -3, 3))
	b := float64(between(r, 1, 4))
	if a == 0 {
		return Question{}, false
	}
	kind := r.IntN(3)
	g := func(j int) float64 {
		switch kind {
		case 0: // cos(bx)
			if j%2 != 0 {
				return 0
			}
			return math.Pow(-1, float64(j/2)) * math.Pow(b, float64(j)) / calcFact(j)
		case 1: // sin(bx)
			if j%2 == 0 {
				return 0
			}
			return math.Pow(-1, float64((j-1)/2)) * math.Pow(b, float64(j)) / calcFact(j)
		default: // ln(1 + bx)
			if j == 0 {
				return 0
			}
			return math.Pow(-1, float64(j+1)) * math.Pow(b, float64(j)) / float64(j)
		}
	}
	ans := 0.0
	for k := 0; k <= n; k++ {
		ans += math.Pow(a, float64(k)) / calcFact(k) * g(n-k)
	}
	fn := [3]string{"cos(%s)", "sin(%s)", "ln(1 + %s)"}[kind]
	p := fmt.Sprintf("Find the coefficient of x^%d in the Maclaurin series of e^(%s)·"+fn+".", n, calcAx(a, "x"), calcAx(b, "x"))
	if !calcOK(ans, 0.01, 1e4) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcMono(c, i, j int) string {
	s := fmt.Sprintf("%d", c)
	for _, t := range []struct {
		v string
		e int
	}{{"x", i}, {"y", j}} {
		switch t.e {
		case 1:
			s += "·" + t.v
		case 2:
			s += "·" + t.v + "²"
		}
	}
	return s
}

func calcDouble(r *rand.Rand) (Question, bool) {
	pp := decimal(r, 1.5, 3, 0.5)
	var terms []string
	ans := 0.0
	for t := 0; t < 2; t++ {
		c, i, j := between(r, 1, 5), between(r, 0, 2), between(r, 0, 2)
		terms = append(terms, calcMono(c, i, j))
		ans += float64(c) * math.Pow(pp, float64(i+2*j+3)) / float64(j+1) * (1/float64(i+j+2) - 1/float64(i+2*j+3))
	}
	if !calcOK(ans, 0.1, 5000) || terms[0] == terms[1] {
		return Question{}, false
	}
	p := fmt.Sprintf("Evaluate the double integral of f(x, y) = %s + %s over the region R bounded by the parabola y = x² and the line y = %s·x (the region lies between the two curves, for 0 ≤ x ≤ %s).", terms[0], terms[1], show(pp), show(pp))
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcLagrange(r *rand.Rand) (Question, bool) {
	pe, qe, re := between(r, 1, 3), between(r, 1, 3), between(r, 1, 3)
	a, b, c := between(r, 1, 6), between(r, 1, 6), between(r, 1, 6)
	K := float64(between(r, 20, 90))
	s := float64(pe + qe + re)
	x := math.Sqrt(float64(pe) * K / (s * float64(a)))
	y := math.Sqrt(float64(qe) * K / (s * float64(b)))
	z := math.Sqrt(float64(re) * K / (s * float64(c)))
	ans := math.Pow(x, float64(pe)) * math.Pow(y, float64(qe)) * math.Pow(z, float64(re))
	ex := func(v string, e int) string {
		if e == 1 {
			return v
		}
		return v + map[int]string{2: "²", 3: "³"}[e]
	}
	p := fmt.Sprintf("Using Lagrange multipliers, find the maximum value of f(x, y, z) = %s·%s·%s over positive x, y, z subject to the constraint %sx² + %sy² + %sz² = %s.", ex("x", pe), ex("y", qe), ex("z", re), calcCo(a), calcCo(b), calcCo(c), show(K))
	if !calcOK(ans, 1, 1e6) {
		return Question{}, false
	}
	return Question{Prompt: p + calcAsk, Answer: num(ans)}, true
}

func calcNewton(r *rand.Rand) (Question, bool) {
	a := float64(between(r, 1, 9))
	b := float64(between(r, 1, 9))
	x := decimal(r, 1.5, 4, 0.5)
	x0 := x
	for i := 0; i < 3; i++ {
		d := 3*x*x - a
		if math.Abs(d) < 2 {
			return Question{}, false
		}
		x -= (x*x*x - a*x - b) / d
	}
	if !calcOK(x, 0.5, 20) {
		return Question{}, false
	}
	p := fmt.Sprintf("Newton's method is applied to f(x) = x³ − %sx − %s starting from x₀ = %s. Compute x₃, the value after exactly three iterations (use the unrounded intermediate values).", show(a), show(b), show(x0))
	return Question{Prompt: p + calcAsk, Answer: num(x)}, true
}

func calcNNN(b int) string {
	if b == 1 {
		return ""
	}
	return fmt.Sprint(b)
}
