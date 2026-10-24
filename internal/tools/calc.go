package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
)

func calculatorTool() Tool {
	return Tool{
		Name:        "calculator",
		Description: "Evaluate an arithmetic expression exactly. Use it for any calculation with more than trivial numbers instead of working it out yourself. Supports + - * / ^ (power) % (remainder), parentheses, pi, e, and sqrt, abs, sin, cos, tan, asin, acos, atan, ln, log (base 10), log2, exp, floor, ceil, round, min, max, pow. Example: (1.07^10) * 2500",
		Parameters:  objSchema("expression", "The expression to evaluate, e.g. \"2^10 + sqrt(144)\""),
		Run: func(ctx context.Context, args json.RawMessage) Result {
			expr, err := strArg(args, "expression")
			if err != nil {
				return fail("Calculated", err.Error())
			}
			v, err := Eval(expr)
			if err != nil {
				return fail("Couldn’t calculate “"+short(expr, 40)+"”", err.Error())
			}
			out := FormatNumber(v)
			return Result{Text: out, Label: "Calculated " + short(expr, 60) + " = " + out}
		},
	}
}

// FormatNumber writes a result without trailing noise: integers as
// integers, others with up to 12 significant digits.
func FormatNumber(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatFloat(v, 'f', -1, 64)
	}
	s := strconv.FormatFloat(v, 'g', 12, 64)
	if f, err := strconv.ParseFloat(s, 64); err == nil && math.Abs(f) < 1e15 && math.Abs(f) >= 1e-4 {
		s = strconv.FormatFloat(f, 'f', -1, 64)
	}
	return s
}

// Eval computes an arithmetic expression. It is a small recursive-descent
// parser: nothing in the input is ever run as code.
func Eval(expr string) (float64, error) {
	if len(expr) > 500 {
		return 0, errors.New("the expression is too long")
	}
	p := &parser{s: []rune(expr)}
	v, err := p.expr(0)
	if err != nil {
		return 0, err
	}
	p.skip()
	if p.i < len(p.s) {
		return 0, fmt.Errorf("unexpected %q at position %d", string(p.s[p.i]), p.i+1)
	}
	if math.IsNaN(v) {
		return 0, errors.New("the result is not a number")
	}
	if math.IsInf(v, 0) {
		return 0, errors.New("the result is too large")
	}
	return v, nil
}

type parser struct {
	s []rune
	i int
}

func (p *parser) skip() {
	for p.i < len(p.s) && unicode.IsSpace(p.s[p.i]) {
		p.i++
	}
}

func (p *parser) peek() rune {
	p.skip()
	if p.i < len(p.s) {
		return p.s[p.i]
	}
	return 0
}

// expr: term (('+'|'-') term)*
func (p *parser) expr(depth int) (float64, error) {
	if depth > 60 {
		return 0, errors.New("too many nested parentheses")
	}
	v, err := p.term(depth)
	if err != nil {
		return 0, err
	}
	for {
		switch p.peek() {
		case '+':
			p.i++
			r, err := p.term(depth)
			if err != nil {
				return 0, err
			}
			v += r
		case '-', '−':
			p.i++
			r, err := p.term(depth)
			if err != nil {
				return 0, err
			}
			v -= r
		default:
			return v, nil
		}
	}
}

// term: unary (('*'|'/'|'%') unary)*
func (p *parser) term(depth int) (float64, error) {
	v, err := p.unary(depth)
	if err != nil {
		return 0, err
	}
	for {
		c := p.peek()
		switch c {
		case '*', '×', '/', '÷', '%':
			p.i++
			r, err := p.unary(depth)
			if err != nil {
				return 0, err
			}
			switch c {
			case '*', '×':
				v *= r
			case '%':
				if r == 0 {
					return 0, errors.New("remainder of division by zero")
				}
				v = math.Mod(v, r)
			default:
				if r == 0 {
					return 0, errors.New("division by zero")
				}
				v /= r
			}
		default:
			return v, nil
		}
	}
}

// unary: ('-'|'+') unary | power. A minus binds looser than ^, so -2^2 = -4.
func (p *parser) unary(depth int) (float64, error) {
	switch p.peek() {
	case '-', '−':
		p.i++
		v, err := p.unary(depth)
		return -v, err
	case '+':
		p.i++
		return p.unary(depth)
	}
	return p.power(depth)
}

// power: atom ('^' unary)?  (right-associative; ** works too)
func (p *parser) power(depth int) (float64, error) {
	b, err := p.atom(depth)
	if err != nil {
		return 0, err
	}
	if p.peek() == '*' && p.i+1 < len(p.s) && p.s[p.i+1] == '*' {
		p.i++
		p.s[p.i] = '^'
	}
	if p.peek() == '^' {
		p.i++
		e, err := p.unary(depth + 1)
		if err != nil {
			return 0, err
		}
		return math.Pow(b, e), nil
	}
	return b, nil
}

func (p *parser) atom(depth int) (float64, error) {
	c := p.peek()
	switch {
	case c == 0:
		return 0, errors.New("the expression ends too soon")
	case c == '(':
		p.i++
		v, err := p.expr(depth + 1)
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, errors.New("missing closing parenthesis")
		}
		p.i++
		return v, nil
	case unicode.IsDigit(c) || c == '.':
		return p.number()
	case unicode.IsLetter(c) || c == 'π':
		return p.name(depth)
	}
	return 0, fmt.Errorf("unexpected %q at position %d", string(c), p.i+1)
}

func (p *parser) number() (float64, error) {
	start := p.i
	for p.i < len(p.s) && (unicode.IsDigit(p.s[p.i]) || p.s[p.i] == '.' || p.s[p.i] == '_') {
		p.i++
	}
	// exponent: 1e6, 2.5E-3
	if p.i < len(p.s) && (p.s[p.i] == 'e' || p.s[p.i] == 'E') {
		j := p.i + 1
		if j < len(p.s) && (p.s[j] == '+' || p.s[j] == '-') {
			j++
		}
		if j < len(p.s) && unicode.IsDigit(p.s[j]) {
			for j < len(p.s) && unicode.IsDigit(p.s[j]) {
				j++
			}
			p.i = j
		}
	}
	text := strings.ReplaceAll(string(p.s[start:p.i]), "_", "")
	v, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return 0, fmt.Errorf("%q isn't a number", text)
	}
	return v, nil
}

var funcs1 = map[string]func(float64) float64{
	"sqrt": math.Sqrt, "abs": math.Abs, "sin": math.Sin, "cos": math.Cos, "tan": math.Tan,
	"asin": math.Asin, "acos": math.Acos, "atan": math.Atan, "ln": math.Log, "log": math.Log10,
	"log10": math.Log10, "log2": math.Log2, "exp": math.Exp, "floor": math.Floor, "ceil": math.Ceil,
	"round": math.Round, "cbrt": math.Cbrt,
}

func (p *parser) name(depth int) (float64, error) {
	start := p.i
	for p.i < len(p.s) && (unicode.IsLetter(p.s[p.i]) || unicode.IsDigit(p.s[p.i])) {
		p.i++
	}
	name := strings.ToLower(string(p.s[start:p.i]))
	if p.peek() != '(' {
		switch name {
		case "pi", "π":
			return math.Pi, nil
		case "e":
			return math.E, nil
		}
		return 0, fmt.Errorf("unknown name %q", name)
	}
	p.i++
	var args []float64
	if p.peek() != ')' {
		for {
			v, err := p.expr(depth + 1)
			if err != nil {
				return 0, err
			}
			args = append(args, v)
			if p.peek() == ',' {
				p.i++
				continue
			}
			break
		}
	}
	if p.peek() != ')' {
		return 0, errors.New("missing closing parenthesis")
	}
	p.i++
	if f, ok := funcs1[name]; ok {
		if len(args) != 1 {
			return 0, fmt.Errorf("%s takes one number", name)
		}
		return f(args[0]), nil
	}
	switch name {
	case "min", "max":
		if len(args) == 0 {
			return 0, fmt.Errorf("%s needs at least one number", name)
		}
		v := args[0]
		for _, a := range args[1:] {
			if name == "min" {
				v = math.Min(v, a)
			} else {
				v = math.Max(v, a)
			}
		}
		return v, nil
	case "pow":
		if len(args) != 2 {
			return 0, errors.New("pow takes two numbers")
		}
		return math.Pow(args[0], args[1]), nil
	}
	return 0, fmt.Errorf("unknown function %q", name)
}
