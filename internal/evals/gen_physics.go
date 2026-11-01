package evals

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Physics: multi-step problems with random numbers. Every question states
// its assumptions, units and precision. verify/physics.py recomputes each
// answer by another route from the numbers in the prompt text.

const physG = 9.81

func physQ(prompt string, ans float64) (Question, bool) {
	return Question{Prompt: prompt, Answer: num(ans)}, true
}

func rad(deg int) float64 { return float64(deg) * math.Pi / 180 }

// signed writes a charge like +3.2 or -4.
func signed(x float64) string {
	if x >= 0 {
		return "+" + show(x)
	}
	return show(x)
}

// solveLin solves a small linear system by Gaussian elimination.
func solveLin(a [][]float64, b []float64) []float64 {
	n := len(b)
	for i := 0; i < n; i++ {
		p := i
		for j := i + 1; j < n; j++ {
			if math.Abs(a[j][i]) > math.Abs(a[p][i]) {
				p = j
			}
		}
		a[i], a[p] = a[p], a[i]
		b[i], b[p] = b[p], b[i]
		for j := i + 1; j < n; j++ {
			f := a[j][i] / a[i][i]
			for k := i; k < n; k++ {
				a[j][k] -= f * a[i][k]
			}
			b[j] -= f * b[i]
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := b[i]
		for k := i + 1; k < n; k++ {
			s -= a[i][k] * x[k]
		}
		x[i] = s / a[i][i]
	}
	return x
}

func init() {
	registerGen(&GenSpec{
		ID: "physics", Name: "Physics", Short: "Physics",
		Description: "Multi-step physics problems (mechanics, circuits, optics, thermodynamics, relativity) with random numbers, answers computed by code and cross-checked by an independent Python script.",
		Seed:        1, PerGen: 5, RelTol: 0.005, MaxTokens: 16384,
		Gens: []Generator{
			{ID: "proj", Make: func(r *rand.Rand) (Question, bool) {
				h := decimal(r, 12, 60, 0.5)
				H := decimal(r, 1, h-4, 0.5)
				v := decimal(r, 8, 30, 0.5)
				th := between(r, 15, 70)
				vx, vy := v*math.Cos(rad(th)), v*math.Sin(rad(th))
				t := (vy + math.Sqrt(vy*vy+2*physG*(h-H))) / physG
				return physQ(fmt.Sprintf("A ball is launched from a point %s m above the ground with a speed of %s m/s, directed %d° above the horizontal. It flies over level ground and lands on a wide, flat terrace whose surface is %s m above the ground (the terrace is below the launch point and its edge is far enough away that the ball's first contact is with the terrace surface). Take g = 9.81 m/s² and ignore air resistance. What horizontal distance does the ball travel from the launch point to the landing point? Give the answer in metres, to 3 significant figures.", show(h), show(v), th, show(H)), vx*t)
			}},
			{ID: "pulley", Make: func(r *rand.Rand) (Question, bool) {
				m1, m2 := decimal(r, 1, 9, 0.1), decimal(r, 2, 12, 0.1)
				th, mu := between(r, 15, 50), decimal(r, 0.1, 0.5, 0.01)
				a := (m2*physG - m1*physG*math.Sin(rad(th)) - mu*m1*physG*math.Cos(rad(th))) / (m1 + m2)
				if a < 0.3 {
					return Question{}, false
				}
				return physQ(fmt.Sprintf("A block of mass %s kg rests on a rough incline that makes an angle of %d° with the horizontal; the coefficient of kinetic friction between block and incline is %s. A light string attached to the block runs up the incline, over a frictionless massless pulley at the top edge, and supports a hanging mass of %s kg. The system is released from rest and the hanging mass descends (so the block slides up the incline). Take g = 9.81 m/s². What is the tension in the string while the block is moving? Give the answer in newtons, to 3 significant figures.", show(m1), th, show(mu), show(m2)), m2*(physG-a))
			}},
			{ID: "damped", Make: func(r *rand.Rand) (Question, bool) {
				m, k := decimal(r, 0.2, 2, 0.05), float64(between(r, 20, 200))
				zeta := decimal(r, 0.04, 0.3, 0.01)
				b := math.Round(2*zeta*math.Sqrt(m*k)*100) / 100
				x0 := decimal(r, 2, 15, 0.5)
				t := decimal(r, 0.3, 3, 0.1)
				if b*b >= 4*m*k || b <= 0 {
					return Question{}, false
				}
				g := b / (2 * m)
				wd := math.Sqrt(k/m - g*g)
				w0sq := k / m
				x0m := x0 / 100
				e := math.Exp(-g * t)
				x := x0m * e * (math.Cos(wd*t) + g/wd*math.Sin(wd*t))
				v := -x0m * w0sq / wd * e * math.Sin(wd*t)
				E := 0.5*m*v*v + 0.5*k*x*x
				if E/(0.5*k*x0m*x0m) < 0.03 {
					return Question{}, false
				}
				return physQ(fmt.Sprintf("A %s kg mass on a horizontal frictionless surface is attached to a spring of constant %s N/m and also to a linear damper that exerts a force of magnitude b·v with b = %s kg/s opposing the velocity. The mass is pulled %s cm from equilibrium and released from rest at t = 0. What is the total mechanical energy (kinetic plus spring potential) of the mass-spring system at t = %s s? Give the answer in millijoules, to 3 significant figures.", show(m), show(k), show(b), show(x0), show(t)), E*1000)
			}},
			{ID: "roll", Make: func(r *rand.Rand) (Question, bool) {
				shapes := []struct {
					name string
					beta float64
				}{{"uniform solid sphere", 0.4}, {"uniform solid cylinder", 0.5}, {"thin-walled hollow cylinder", 1}, {"thin spherical shell", 2.0 / 3}}
				s := shapes[r.IntN(len(shapes))]
				L, th, H := decimal(r, 1, 4, 0.1), between(r, 12, 40), decimal(r, 0.8, 3, 0.1)
				v := math.Sqrt(2 * physG * L * math.Sin(rad(th)) / (1 + s.beta))
				vx, vy := v*math.Cos(rad(th)), v*math.Sin(rad(th))
				t := (-vy + math.Sqrt(vy*vy+2*physG*H)) / physG
				return physQ(fmt.Sprintf("A %s is released from rest at the top of a rough incline of length %s m inclined at %d° to the horizontal, and rolls without slipping all the way down. The bottom end of the incline is the edge of a ledge %s m above level ground, and the object leaves the edge moving along the direction of the incline (so its velocity points %d° below the horizontal). Take g = 9.81 m/s² and ignore air resistance. How far horizontally from the ledge edge does the object first hit the ground? Give the answer in metres, to 3 significant figures.", s.name, show(L), th, show(H), th), vx*t)
			}},
			{ID: "collide", Make: func(r *rand.Rand) (Question, bool) {
				m1, v1 := decimal(r, 2, 9, 0.1), decimal(r, 3, 12, 0.1)
				m2, v2 := decimal(r, 1, 8, 0.1), decimal(r, 1, 6, 0.1)
				s, mu1 := decimal(r, 1, 6, 0.1), decimal(r, 0.05, 0.4, 0.01)
				th, mu2 := between(r, 10, 35), decimal(r, 0.05, 0.4, 0.01)
				vc := (m1*v1 - m2*v2) / (m1 + m2)
				w := vc*vc - 2*mu1*physG*s
				if m1*v1 <= m2*v2 || w < 1 {
					return Question{}, false
				}
				d := w / (2 * physG * (math.Sin(rad(th)) + mu2*math.Cos(rad(th))))
				return physQ(fmt.Sprintf("On a frictionless horizontal surface, a %s kg block moving right at %s m/s collides head-on with a %s kg block moving left at %s m/s, and the two stick together. Immediately afterwards the combined body slides onto a rough horizontal patch %s m long (coefficient of kinetic friction %s), and then directly onto a rough incline rising to the right at %d° above the horizontal (coefficient of kinetic friction %s); the transition onto the incline is smooth with no loss of speed. Take g = 9.81 m/s². How far along the incline (measured along its surface from its bottom) does the combined body travel before first coming to rest? Give the answer in metres, to 3 significant figures.", show(m1), show(v1), show(m2), show(v2), show(s), show(mu1), th, show(mu2)), d)
			}},
			{ID: "hohmann", Make: func(r *rand.Rand) (Question, bool) {
				h1, h2 := float64(between(r, 300, 1500)), float64(between(r, 8000, 40000))
				r1, r2 := (6371+h1)*1e3, (6371+h2)*1e3
				a := (r1 + r2) / 2
				t := math.Pi * math.Sqrt(a*a*a/3.986e14) / 60
				return physQ(fmt.Sprintf("A spacecraft is in a circular orbit around Earth at an altitude of %s km. It fires its engine briefly, tangentially to the orbit, and so moves onto an elliptical orbit whose highest point (apoapsis) is at an altitude of %s km. Treat Earth as a uniform sphere of radius 6371 km with gravitational parameter GM = 3.986×10^14 m³/s², neglect everything else, and treat the burn as instantaneous. How long does the spacecraft take to travel from the burn to the apoapsis? Give the answer in minutes, to 3 significant figures.", show(h1), show(h2)), t)
			}},
			{ID: "shadow", Make: func(r *rand.Rand) (Question, bool) {
				h := float64(between(r, 300, 6000))
				rr := (6371 + h) * 1e3
				T := 2 * math.Pi * math.Sqrt(rr*rr*rr/3.986e14)
				t := T * math.Asin(6.371e6/rr) / math.Pi / 60
				return physQ(fmt.Sprintf("A satellite is in a circular orbit around Earth at an altitude of %s km, in a plane that contains the direction to the Sun. Treat Sunlight as a parallel beam, Earth as a sphere of radius 6371 km that casts a cylindrical shadow (no penumbra, no atmosphere) and gravitational parameter GM = 3.986×10^14 m³/s². For how long in each orbit is the satellite inside Earth's shadow? Give the answer in minutes, to 3 significant figures.", show(h)), t)
			}},
			{ID: "banked", Make: func(r *rand.Rand) (Question, bool) {
				R, th, mu := float64(between(r, 30, 200)), between(r, 8, 30), decimal(r, 0.1, 0.6, 0.01)
				s, c := math.Sin(rad(th)), math.Cos(rad(th))
				v := math.Sqrt(physG*R*(s+mu*c)/(c-mu*s)) * 3.6
				return physQ(fmt.Sprintf("A road curve is banked at %d° and follows a horizontal circle of radius %s m. The coefficient of static friction between tyres and road is %s. Treating the car as a point mass and taking g = 9.81 m/s², what is the maximum speed at which a car can take the curve without sliding up the bank? Give the answer in km/h, to 3 significant figures.", th, show(R), show(mu)), v)
			}},
			{ID: "bridge", Make: func(r *rand.Rand) (Question, bool) {
				var R [5]float64
				for i := range R {
					R[i] = float64(between(r, 10, 99))
				}
				E, ri := decimal(r, 6, 24, 0.5), decimal(r, 0.5, 5, 0.5)
				// unknowns: Va, Vc, Vd (node B = ground)
				g := func(x float64) float64 { return 1 / x }
				A := [][]float64{
					{1/ri + g(R[0]) + g(R[1]), -g(R[0]), -g(R[1])},
					{-g(R[0]), g(R[0]) + g(R[2]) + g(R[4]), -g(R[4])},
					{-g(R[1]), -g(R[4]), g(R[1]) + g(R[3]) + g(R[4])},
				}
				x := solveLin(A, []float64{E / ri, 0, 0})
				i5 := math.Abs(x[1]-x[2]) / R[4] * 1000
				if i5 < 0.03*E/(R[0]+R[2])*1000 {
					return Question{}, false
				}
				return physQ(fmt.Sprintf("A battery of EMF %s V and internal resistance %s Ω is connected across terminals A (positive) and B (negative) of a resistor network. Four resistors form a bridge: R1 = %s Ω connects A to C, R2 = %s Ω connects A to D, R3 = %s Ω connects C to B, and R4 = %s Ω connects D to B. A fifth resistor R5 = %s Ω connects C to D. All wires are ideal. What is the magnitude of the current through R5? Give the answer in milliamperes, to 3 significant figures.", show(E), show(ri), show(R[0]), show(R[1]), show(R[2]), show(R[3]), show(R[4])), i5)
			}},
			{ID: "rc", Make: func(r *rand.Rand) (Question, bool) {
				V := decimal(r, 5, 24, 0.5)
				R1, R2, C := decimal(r, 1, 9, 0.1), decimal(r, 1, 9, 0.1), decimal(r, 0.5, 5, 0.1)
				tau1 := R1 * R2 / (R1 + R2) * C
				tau2 := R2 * C
				t1 := math.Round(decimal(r, 0.5, 3, 0.1)*tau1*10) / 10
				t2 := math.Round(decimal(r, 0.2, 2.5, 0.1)*tau2*10) / 10
				if t1 < 0.5 || t2 < 0.5 {
					return Question{}, false
				}
				vinf := V * R2 / (R1 + R2)
				v1 := vinf * (1 - math.Exp(-t1/tau1))
				v2 := v1 * math.Exp(-t2/tau2)
				if v2 < 0.02 {
					return Question{}, false
				}
				return physQ(fmt.Sprintf("In a circuit, a resistor R2 = %s kΩ and an initially uncharged capacitor C = %s μF are connected in parallel between a node and ground. At t = 0 a switch closes, connecting an ideal %s V source in series with a resistor R1 = %s kΩ to that node. At t = %s ms the switch opens again, completely disconnecting the source and R1 from the node, leaving only R2 in parallel with C. What is the voltage across the capacitor at t = %s ms (that is, %s ms after the switch opens)? Give the answer in volts, to 3 significant figures.", show(R2), show(C), show(V), show(R1), show(t1), show(math.Round((t1+t2)*10)/10), show(t2)), v2)
			}},
			{ID: "rlc", Make: func(r *rand.Rand) (Question, bool) {
				R, L, C := float64(between(r, 10, 200)), float64(between(r, 10, 200)), float64(between(r, 1, 50))
				V := decimal(r, 5, 50, 0.5)
				f0 := 1 / (2 * math.Pi * math.Sqrt(L*1e-3*C*1e-6))
				f := math.Round(f0 * decimal(r, 0.6, 1.5, 0.05))
				w := 2 * math.Pi * f
				X := w*L*1e-3 - 1/(w*C*1e-6)
				I := V / math.Hypot(R, X)
				return physQ(fmt.Sprintf("A series circuit contains a resistor R = %s Ω, an ideal inductor L = %s mH and an ideal capacitor C = %s μF, driven by an ideal AC source with voltage amplitude %s V at frequency f = %s Hz. In the steady state, what is the amplitude of the voltage across the capacitor alone? Give the answer in volts, to 3 significant figures.", show(R), show(L), show(C), show(V), show(f)), I/(w*C*1e-6))
			}},
			{ID: "efield", Make: func(r *rand.Rand) (Question, bool) {
				type ch struct{ q, x, y float64 }
				var cs [3]ch
				px, py := float64(between(r, -15, 15)), float64(between(r, -15, 15))
				var ex, ey, mx float64
				for i := range cs {
					cs[i] = ch{decimal(r, 1, 9, 0.1), float64(between(r, -30, 30)), float64(between(r, -30, 30))}
					if r.IntN(2) == 0 {
						cs[i].q = -cs[i].q
					}
					dx, dy := px-cs[i].x, py-cs[i].y
					d := math.Hypot(dx, dy)
					if d < 6 {
						return Question{}, false
					}
					e := 8.99e9 * cs[i].q * 1e-9 / (d * d * 1e-4)
					mx = math.Max(mx, math.Abs(e))
					ex += e * dx / d
					ey += e * dy / d
				}
				E := math.Hypot(ex, ey)
				if E < 0.25*mx {
					return Question{}, false
				}
				s := ""
				for i, c := range cs {
					s += fmt.Sprintf("q%d = %s nC at (x, y) = (%s, %s) cm; ", i+1, signed(c.q), show(c.x), show(c.y))
				}
				return physQ(fmt.Sprintf("Three point charges lie in a plane: %sTake k = 8.99×10^9 N·m²/C² and ignore everything except these charges. What is the magnitude of the net electric field at the point (x, y) = (%s, %s) cm? Give the answer in kilovolts per metre (kV/m), to 3 significant figures.", s, show(px), show(py)), E/1000)
			}},
			{ID: "gascycle", Make: func(r *rand.Rand) (Question, bool) {
				n, T := decimal(r, 1, 3, 0.1), float64(between(r, 300, 600))
				V1, ratio := decimal(r, 5, 30, 0.5), decimal(r, 1.8, 4, 0.1)
				cv, name := 2.5, "diatomic"
				if r.IntN(2) == 0 {
					cv, name = 1.5, "monatomic"
				}
				ln := math.Log(ratio)
				qAB := ln
				qCA := cv * (1 - 1/ratio)
				W := ln - (1 - 1/ratio)
				return physQ(fmt.Sprintf("n = %s mol of an ideal %s gas (constant heat capacities) undergoes a reversible cycle A → B → C → A. At A its temperature is %s K and its volume is %s L. A → B: isothermal expansion to a volume %s times larger. B → C: isobaric compression until the volume is back to the original %s L. C → A: heating at constant volume back to the starting state. What is the thermal efficiency of the cycle (net work done by the gas divided by the total heat absorbed by the gas during the steps in which it absorbs heat)? Give the answer in percent, to 3 significant figures.", show(n), name, show(T), show(V1), show(ratio), show(V1)), 100*W/(qAB+qCA))
			}},
			{ID: "lenses", Make: func(r *rand.Rand) (Question, bool) {
				f1 := float64(between(r, 8, 30))
				f2 := float64(between(r, 8, 30))
				if r.IntN(2) == 0 {
					f2 = -f2
				}
				d, s := float64(between(r, 5, 40)), float64(between(r, 10, 60))
				if math.Abs(s-f1) < 2 {
					return Question{}, false
				}
				i1 := f1 * s / (s - f1)
				s2 := d - i1
				if math.Abs(s2-f2) < 2 {
					return Question{}, false
				}
				i2 := f2 * s2 / (s2 - f2)
				if math.Abs(i2) > 150 || math.Abs(i2) < 2 {
					return Question{}, false
				}
				return physQ(fmt.Sprintf("Two thin lenses share an optical axis, with light travelling left to right. Lens 1 has focal length %s cm; lens 2 has focal length %s cm (negative means diverging) and sits %s cm to the right of lens 1. A small object stands on the axis %s cm to the left of lens 1. Use the paraxial thin-lens model. Where is the final image, measured from lens 2? Give the signed distance in centimetres, positive if the image is to the right of lens 2 and negative if it is to the left, to 3 significant figures.", show(f1), show(f2), show(d), show(s)), i2)
			}},
			{ID: "doppler", Make: func(r *rand.Rand) (Question, bool) {
				f0, vs, vo, w := float64(between(r, 400, 1200)), float64(between(r, 10, 60)), float64(between(r, 5, 40)), float64(between(r, 2, 15))
				vos, ws := vo, w
				od, wdir := "away from the source (to the right)", "from the source towards the observer (to the right)"
				if r.IntN(2) == 0 {
					vos, od = -vo, "towards the source (to the left)"
				}
				if r.IntN(2) == 0 {
					ws, wdir = -w, "from the observer towards the source (to the left)"
				}
				c := 343.0
				f := f0 * (c + ws - vos) / (c + ws - vs)
				return physQ(fmt.Sprintf("On a straight line, a source emitting a steady tone of %s Hz moves to the right at %s m/s (relative to the ground), chasing an observer who is initially 500 m to its right. The observer moves at %s m/s (relative to the ground) %s. A steady wind of %s m/s (relative to the ground) blows %s. The speed of sound relative to the air is 343 m/s. What frequency does the observer hear? Give the answer in hertz, to 4 significant figures.", show(f0), show(vs), show(vo), od, show(w), wdir), f)
			}},
			{ID: "twins", Make: func(r *rand.Rand) (Question, bool) {
				b1 := decimal(r, 0.6, 0.9, 0.01)
				u := decimal(r, 0.7, 0.99, 0.01)
				b2 := (u - b1) / (1 - u*b1)
				if u < b1+0.05 || b2 < 0.3 {
					return Question{}, false
				}
				D := decimal(r, 3, 15, 0.5)
				T := D/b1 + D/b2
				tau := D/b1*math.Sqrt(1-b1*b1) + D/b2*math.Sqrt(1-b2*b2)
				return physQ(fmt.Sprintf("A star is %s light-years from Earth, at rest relative to Earth. Ship A leaves Earth at %s c (relative to Earth) heading straight for the star. As A passes the star, a passenger instantly steps onto ship B, which is moving straight back towards Earth at %s c as measured in ship A's rest frame. Ignore accelerations, the star's gravity and any time spent transferring. When the passenger arrives back at Earth, how many years older is a twin who stayed on Earth than the passenger? Give the answer in years, to 3 significant figures.", show(D), show(b1), show(u)), T-tau)
			}},
			{ID: "calorimeter", Make: func(r *rand.Rand) (Question, bool) {
				mi, Ti := float64(between(r, 50, 400)), float64(between(r, 5, 30))
				mw, Tw, mc := float64(between(r, 200, 800)), float64(between(r, 20, 70)), float64(between(r, 50, 200))
				mel := r.IntN(2) == 0
				qHave := (mw*4186 + mc*900) * Tw / 1000
				qNeed := mi * (2100*Ti + 334000) / 1000
				if (qHave >= qNeed) != mel {
					return Question{}, false
				}
				head := fmt.Sprintf("An aluminium calorimeter cup of mass %s g (specific heat 900 J/(kg·K)) contains %s g of liquid water, both at %s °C. Then %s g of ice at −%s °C is added and the contents are stirred until equilibrium, with no heat exchanged with the surroundings and no evaporation. Use: specific heat of ice 2100 J/(kg·K), of liquid water 4186 J/(kg·K), latent heat of fusion 334 kJ/kg, and a melting point of 0 °C. ", show(mc), show(mw), show(Tw), show(mi), show(Ti))
				if mel {
					T := (qHave - qNeed) / ((mi+mw)*4186 + mc*900) * 1000
					if T < 1 {
						return Question{}, false
					}
					return physQ(head+"What is the final equilibrium temperature? Give the answer in °C, to 3 significant figures.", T)
				}
				left := (qNeed - qHave) / 334 // g
				if left < 5 || left > mi+mw-10 || qHave < mi*2100*Ti/1000 {
					return Question{}, false
				}
				return physQ(head+"How much ice (solid) is present at equilibrium? Give the answer in grams, to 3 significant figures.", left)
			}},
			{ID: "sinker", Make: func(r *rand.Rand) (Question, bool) {
				a := decimal(r, 8, 20, 0.5)
				rw := float64(between(r, 40, 85) * 10)
				rf := pick(r, 1000.0, 1025.0, 1100.0, 1260.0)
				if rw >= rf-100 {
					return Question{}, false
				}
				m := (rf - rw) * math.Pow(a/100, 3) / (1 - rf/11340) * 1000
				return physQ(fmt.Sprintf("A wooden cube of side %s cm and density %s kg/m³ floats in a liquid of density %s kg/m³. A lead sinker (density 11340 kg/m³) is hung from the bottom of the cube by a thin string of negligible mass and volume, and the mass of the sinker is chosen so that the whole assembly is in equilibrium with the cube just fully submerged (its top face exactly level with the liquid surface) and the sinker completely under the surface. What is the mass of the sinker? Give the answer in grams, to 3 significant figures.", show(a), show(rw), show(rf)), m)
			}},
		},
	})
}
