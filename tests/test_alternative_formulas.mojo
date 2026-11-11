from std.math import exp, log, sqrt
from std.testing import TestSuite, assert_almost_equal
from finrecipes.normdist import N
from finrecipes.black_scholes import option_price_call_black_scholes
from finrecipes.alternative_formulas import (
    option_price_call_merton_jump_diffusion,
    heston_call_option_price,
    heston_Pj,
    heston_integrand_j,
    HestonParms,
    integration_qagiu,
    RealFunction,
    Complex,
    complex_sqrt,
    complex_exp,
    complex_log,
)

comptime PI: Float64 = 3.141592653589793


def test_merton_jump_diffusion() raises:
    # C++ Code 19.1 example: Merton Jump diffusion call = 23.2074
    assert_almost_equal(
        option_price_call_merton_jump_diffusion(100.0, 100.0, 0.05, 0.3, 1.0, 0.5, 0.5, 0.5), 23.2074, atol=1e-4
    )


def test_merton_without_jumps_is_black_scholes() raises:
    assert_almost_equal(
        option_price_call_merton_jump_diffusion(100.0, 100.0, 0.05, 0.3, 1.0, 0.0, 0.5, 0.5),
        option_price_call_black_scholes(100.0, 100.0, 0.05, 0.3, 1.0),
        atol=1e-12,
    )


def test_heston_call() raises:
    # C++ Code 19.2 example: heston call price 3.06944
    assert_almost_equal(
        heston_call_option_price(100.0, 100.0, 0.01, 0.01, 0.5, 0.0, 2.0, 0.0, 0.01, 0.01), 3.06944, atol=1e-5
    )


def test_heston_near_constant_volatility_is_black_scholes() raises:
    # with almost no volatility of variance and v = theta, the variance stays
    # at v and the price is the Black Scholes price with sigma = sqrt(v)
    var heston = heston_call_option_price(100.0, 95.0, 0.03, 0.04, 1.0, 0.0, 2.0, 0.0, 0.04, 0.0001)
    assert_almost_equal(heston, option_price_call_black_scholes(100.0, 95.0, 0.03, 0.2, 1.0), atol=1e-4)
    # P1 is then the delta N(d1) and P2 the risk neutral probability of exercise N(d2)
    var d1 = (log(Float64(100.0 / 95.0)) + 0.03) / 0.2 + 0.1
    var d2 = d1 - 0.2
    var P1 = heston_Pj(100.0, 95.0, 0.03, 0.04, 1.0, 0.0001, 2.0, 0.0, 0.0, 0.04, 1)
    var P2 = heston_Pj(100.0, 95.0, 0.03, 0.04, 1.0, 0.0001, 2.0, 0.0, 0.0, 0.04, 2)
    assert_almost_equal(P1, N(d1), atol=1e-5)
    assert_almost_equal(P2, N(d2), atol=1e-5)
    assert_almost_equal(100.0 * P1 - 95.0 * exp(Float64(-0.03)) * P2, heston, atol=1e-12)


def test_heston_with_correlation() raises:
    # Reference values from an independent evaluation of the formula of
    # C++ Code 19.2 (Python complex arithmetic, composite Simpson rule with
    # 400000 steps on a truncated range). They exercise rho != 0 and
    # lambda != 0, which the book's example leaves at zero.
    assert_almost_equal(
        heston_call_option_price(100.0, 90.0, 0.03, 0.05, 1.0, -0.5, 1.5, 0.1, 0.04, 0.3), 15.920833, atol=1e-5
    )
    assert_almost_equal(heston_Pj(100.0, 90.0, 0.03, 0.05, 1.0, 0.3, 1.5, 0.1, -0.5, 0.04, 1), 0.8064053, atol=1e-6)
    assert_almost_equal(heston_Pj(100.0, 90.0, 0.03, 0.05, 1.0, 0.3, 1.5, 0.1, -0.5, 0.04, 2), 0.7410079, atol=1e-6)
    assert_almost_equal(
        heston_call_option_price(100.0, 110.0, 0.02, 0.03, 0.25, 0.4, 3.0, 0.0, 0.06, 0.5), 1.201605, atol=1e-5
    )


def test_heston_integrand() raises:
    # reference values from the same independent evaluation
    var parms = HestonParms(90.0, log(Float64(100.0)), 0.03, 0.05, 1.0, 1.5, 0.04, -0.5, 0.3, 0.1, 1)
    assert_almost_equal(heston_integrand_j(0.5, parms), 0.15548668383495978, atol=1e-10)
    var parms2 = HestonParms(90.0, log(Float64(100.0)), 0.03, 0.05, 1.0, 1.5, 0.04, -0.5, 0.3, 0.1, 2)
    assert_almost_equal(heston_integrand_j(40.0, parms2), -7.43288875892638e-07, atol=1e-12)
    assert_almost_equal(parms.eval(0.5), heston_integrand_j(0.5, parms), atol=1e-15)
    # far out exp(d tau) overflows; the integrand has decayed to zero there
    assert_almost_equal(heston_integrand_j(1e8, parms), 0.0, atol=1e-300)


struct Exponential(RealFunction, Copyable, Movable):
    var rate: Float64

    def __init__(out self, rate: Float64):
        self.rate = rate

    def eval(self, x: Float64) -> Float64:
        return exp(-self.rate * x)


struct Lorentz(RealFunction, Copyable, Movable):
    def __init__(out self):
        pass

    def eval(self, x: Float64) -> Float64:
        return 1.0 / (1.0 + x * x)


struct RootExponential(RealFunction, Copyable, Movable):
    def __init__(out self):
        pass

    def eval(self, x: Float64) -> Float64:
        return sqrt(x) * exp(-x)


def test_integration_to_infinity() raises:
    # integral of exp(-2x) over [0, inf) = 1/2, over [1, inf) = exp(-2)/2
    assert_almost_equal(integration_qagiu(Exponential(2.0), 0.0, 1e-10, 1e-10, 1000), 0.5, atol=1e-9)
    assert_almost_equal(
        integration_qagiu(Exponential(2.0), 1.0, 1e-10, 1e-10, 1000), 0.5 * exp(Float64(-2.0)), atol=1e-9
    )
    # integral of 1/(1+x^2) over [0, inf) = pi/2
    assert_almost_equal(integration_qagiu(Lorentz(), 0.0, 1e-10, 1e-10, 1000), PI / 2.0, atol=1e-9)
    # integral of sqrt(x) exp(-x) over [0, inf) = sqrt(pi)/2: the slope is
    # unbounded at 0, so the rule has to be refined there
    assert_almost_equal(integration_qagiu(RootExponential(), 0.0, 1e-10, 1e-10, 1000), sqrt(PI) / 2.0, atol=1e-9)


def test_complex_arithmetic() raises:
    var a = Complex(1.0, 2.0)
    var b = Complex(3.0, -1.0)
    var s = a + b
    assert_almost_equal(s.re, 4.0, atol=1e-14)
    assert_almost_equal(s.im, 1.0, atol=1e-14)
    var m = a - b
    assert_almost_equal(m.re, -2.0, atol=1e-14)
    assert_almost_equal(m.im, 3.0, atol=1e-14)
    var p = a * b  # (1+2i)(3-i) = 5+5i
    assert_almost_equal(p.re, 5.0, atol=1e-14)
    assert_almost_equal(p.im, 5.0, atol=1e-14)
    var q = p / b  # back to a
    assert_almost_equal(q.re, 1.0, atol=1e-14)
    assert_almost_equal(q.im, 2.0, atol=1e-14)
    var t = a * 2.0
    assert_almost_equal(t.re, 2.0, atol=1e-14)
    assert_almost_equal(t.im, 4.0, atol=1e-14)


def test_complex_functions() raises:
    var r1 = complex_sqrt(Complex(3.0, 4.0))  # 2+i
    assert_almost_equal(r1.re, 2.0, atol=1e-14)
    assert_almost_equal(r1.im, 1.0, atol=1e-14)
    var r2 = complex_sqrt(Complex(-4.0, 0.0))  # 2i
    assert_almost_equal(r2.re, 0.0, atol=1e-14)
    assert_almost_equal(r2.im, 2.0, atol=1e-14)
    var r3 = complex_sqrt(Complex(-3.0, -4.0))  # 1-2i
    assert_almost_equal(r3.re, 1.0, atol=1e-14)
    assert_almost_equal(r3.im, -2.0, atol=1e-14)
    var r4 = complex_sqrt(Complex(0.0, 0.0))
    assert_almost_equal(r4.re, 0.0, atol=1e-14)
    assert_almost_equal(r4.im, 0.0, atol=1e-14)
    var e = complex_exp(Complex(0.0, PI))  # -1
    assert_almost_equal(e.re, -1.0, atol=1e-14)
    assert_almost_equal(e.im, 0.0, atol=1e-14)
    var l = complex_log(Complex(-1.0, 0.0))  # i pi
    assert_almost_equal(l.re, 0.0, atol=1e-14)
    assert_almost_equal(l.im, PI, atol=1e-14)
    var l2 = complex_log(complex_exp(Complex(0.5, -1.25)))
    assert_almost_equal(l2.re, 0.5, atol=1e-14)
    assert_almost_equal(l2.im, -1.25, atol=1e-14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
