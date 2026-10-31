package evals

import "math"

// Point is one model with both numbers: X is its score on your
// benchmarks (0–1), Y its published index score.
type Point struct {
	X, Y float64
}

// Fit is a straight line Y = A + B·X through the anchor models.
type Fit struct {
	A, B float64
	// N is how many models it was fitted on; R2 how much of the spread in
	// published scores the line explains (1 is perfect); SE the typical
	// miss, in index points.
	N  int
	R2 float64
	SE float64
	// MinX and MaxX are the range of your scores it was fitted on; an
	// estimate outside it is an extrapolation.
	MinX, MaxX float64
}

// MinAnchors is the fewest models a fit needs.
const MinAnchors = 3

// FitLine fits Y on X by least squares. ok is false with fewer than
// MinAnchors points or when every X is the same.
func FitLine(pts []Point) (f Fit, ok bool) {
	n := len(pts)
	if n < MinAnchors {
		return Fit{N: n}, false
	}
	var sx, sy float64
	f.MinX, f.MaxX = math.Inf(1), math.Inf(-1)
	for _, p := range pts {
		sx += p.X
		sy += p.Y
		f.MinX, f.MaxX = math.Min(f.MinX, p.X), math.Max(f.MaxX, p.X)
	}
	mx, my := sx/float64(n), sy/float64(n)
	var sxx, sxy, syy float64
	for _, p := range pts {
		sxx += (p.X - mx) * (p.X - mx)
		sxy += (p.X - mx) * (p.Y - my)
		syy += (p.Y - my) * (p.Y - my)
	}
	if sxx < 1e-12 {
		return Fit{N: n}, false
	}
	f.N = n
	f.B = sxy / sxx
	f.A = my - f.B*mx
	var sse float64
	for _, p := range pts {
		d := p.Y - f.Predict(p.X)
		sse += d * d
	}
	if syy > 0 {
		f.R2 = 1 - sse/syy
	}
	if n > 2 {
		f.SE = math.Sqrt(sse / float64(n-2))
	}
	return f, true
}

// Predict places a score on the index.
func (f Fit) Predict(x float64) float64 { return f.A + f.B*x }

// Outside says x lies beyond the scores the line was fitted on.
func (f Fit) Outside(x float64) bool { return x < f.MinX-1e-9 || x > f.MaxX+1e-9 }
