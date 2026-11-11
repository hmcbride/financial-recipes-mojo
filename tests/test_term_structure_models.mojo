from std.testing import TestSuite, assert_almost_equal
from finrecipes.term_structure import TermStructure, bonds_price
from finrecipes.term_structure_models import (
    term_structure_yield_nelson_siegel,
    TermStructureNelsonSiegel,
    term_structure_yield_svensson,
    TermStructureSvensson,
    term_structure_discount_factor_cubic_spline,
    TermStructureCubicSpline,
    term_structure_discount_factor_cir,
    TermStructureCir,
    term_structure_discount_factor_vasicek,
    TermStructureVasicek,
)


def test_nelson_siegel() raises:
    # Book example, C++ Code 22.1 and 22.2
    var beta0: Float64 = 0.01
    var beta1: Float64 = 0.01
    var beta2: Float64 = 0.01
    var lambda_: Float64 = 5.0
    var t: Float64 = 1.0
    assert_almost_equal(term_structure_yield_nelson_siegel(t, beta0, beta1, beta2, lambda_), 0.0363142, atol=1e-7)
    var ns = TermStructureNelsonSiegel(beta0, beta1, beta2, lambda_)
    assert_almost_equal(ns.r(t), 0.0363142, atol=1e-7)
    assert_almost_equal(ns.d(t), 0.964337, atol=1e-6)
    assert_almost_equal(ns.f(1.0, 2.0), 0.0300602, atol=1e-7)


def test_nelson_siegel_at_zero() raises:
    assert_almost_equal(term_structure_yield_nelson_siegel(0.0, 0.01, 0.02, 0.03, 5.0), 0.01, atol=1e-15)
    var ns = TermStructureNelsonSiegel(0.01, 0.02, 0.03, 5.0)
    assert_almost_equal(ns.r(0.0), 0.01, atol=1e-15)
    assert_almost_equal(ns.r(-1.0), 0.01, atol=1e-15)


def test_svensson() raises:
    # C++ Code 22.3 and 22.4 have no book example. Expected values calculated
    # from Formula 22.2 with beta0=beta1=beta2=0.01, beta3=0.02, tau1=5, tau2=2.
    assert_almost_equal(
        term_structure_yield_svensson(1.0, 0.01, 0.01, 0.01, 0.02, 5.0, 2.0), 0.0235477775787, atol=1e-12
    )
    assert_almost_equal(
        term_structure_yield_svensson(2.0, 0.01, 0.01, 0.01, 0.02, 5.0, 2.0), 0.0250656195910, atol=1e-12
    )
    # without the second hump term
    assert_almost_equal(
        term_structure_yield_svensson(1.0, 0.01, 0.01, 0.01, 0.0, 5.0, 2.0), 0.0199396171614, atol=1e-12
    )
    var sv = TermStructureSvensson(0.01, 0.01, 0.01, 0.02, 5.0, 2.0)
    assert_almost_equal(sv.r(1.0), 0.0235477775787, atol=1e-12)
    assert_almost_equal(sv.d(1.0), 0.976727307888, atol=1e-11)
    assert_almost_equal(sv.f(1.0, 2.0), 0.0265834616033, atol=1e-11)


def test_svensson_at_zero() raises:
    assert_almost_equal(term_structure_yield_svensson(0.0, 0.01, 0.02, 0.03, 0.04, 5.0, 2.0), 0.01, atol=1e-15)
    var sv = TermStructureSvensson(0.01, 0.02, 0.03, 0.04, 5.0, 2.0)
    assert_almost_equal(sv.r(0.0), 0.01, atol=1e-15)
    assert_almost_equal(sv.r(-1.0), 0.01, atol=1e-15)


def test_cubic_spline() raises:
    # Book example, C++ Code 22.5 and 22.6
    var b: Float64 = 0.1
    var c: Float64 = 0.1
    var d: Float64 = -0.1
    var f: List[Float64] = [0.01, 0.01, -0.01]
    var knots: List[Float64] = [2.0, 7.0, 12.0]
    assert_almost_equal(term_structure_discount_factor_cubic_spline(1.0, b, c, d, f, knots), 1.1, atol=1e-6)
    var cs = TermStructureCubicSpline(b, c, d, f, knots)
    assert_almost_equal(cs.r(1.0), -0.0953102, atol=1e-7)
    assert_almost_equal(cs.d(1.0), 1.1, atol=1e-6)
    assert_almost_equal(cs.f(1.0, 2.0), 0.318454, atol=1e-6)


def test_cubic_spline_knots() raises:
    var f: List[Float64] = [0.01, 0.01, -0.01]
    var knots: List[Float64] = [2.0, 7.0, 12.0]
    # d(0) = 1
    assert_almost_equal(term_structure_discount_factor_cubic_spline(0.0, 0.1, 0.1, -0.1, f, knots), 1.0, atol=1e-15)
    # past the first knot: 1 + 0.3 + 0.9 - 2.7 + 0.01 * 1^3
    assert_almost_equal(term_structure_discount_factor_cubic_spline(3.0, 0.1, 0.1, -0.1, f, knots), -0.49, atol=1e-12)
    # past the second knot: 1 + 0.8 + 6.4 - 51.2 + 0.01 * 6^3 + 0.01 * 1^3
    assert_almost_equal(term_structure_discount_factor_cubic_spline(8.0, 0.1, 0.1, -0.1, f, knots), -40.83, atol=1e-12)
    # past all three: 1 + 1.3 + 16.9 - 219.7 + 0.01 * 11^3 + 0.01 * 6^3 - 0.01 * 1^3
    assert_almost_equal(
        term_structure_discount_factor_cubic_spline(13.0, 0.1, 0.1, -0.1, f, knots), -185.04, atol=1e-12
    )


def test_cubic_spline_mismatched_sizes() raises:
    # C++ Code 22.6: if f and knots differ in length the class keeps neither
    var f: List[Float64] = [0.01, 0.01]
    var knots: List[Float64] = [2.0, 7.0, 12.0]
    var cs = TermStructureCubicSpline(0.1, 0.1, -0.1, f, knots)
    assert_almost_equal(cs.d(3.0), 1.0 + 0.3 + 0.9 - 2.7, atol=1e-12)


def test_cir() raises:
    # Book example, C++ Code 22.7 and 22.8
    var r: Float64 = 0.05
    var kappa: Float64 = 0.01
    var sigma: Float64 = 0.1
    var theta: Float64 = 0.08
    var lambda_: Float64 = 0.0
    assert_almost_equal(term_structure_discount_factor_cir(1.0, r, kappa, lambda_, theta, sigma), 0.951166, atol=1e-6)
    var cir = TermStructureCir(r, kappa, lambda_, theta, sigma)
    assert_almost_equal(cir.r(1.0), 0.0500668, atol=1e-7)
    assert_almost_equal(cir.d(1.0), 0.951166, atol=1e-6)
    assert_almost_equal(cir.f(1.0, 2.0), 0.0498756, atol=1e-7)


def test_vasicek() raises:
    # Book example, C++ Code 22.9 and 22.10
    var r: Float64 = 0.05
    var a: Float64 = -0.1
    var b: Float64 = 0.1
    var sigma: Float64 = 0.1
    assert_almost_equal(term_structure_discount_factor_vasicek(1.0, r, a, b, sigma), 0.955408, atol=1e-6)
    var vc = TermStructureVasicek(r, a, b, sigma)
    assert_almost_equal(vc.r(1.0), 0.0456168, atol=1e-7)
    assert_almost_equal(vc.d(1.0), 0.955408, atol=1e-6)
    assert_almost_equal(vc.f(1.0, 2.0), 0.0281476, atol=1e-7)


def test_vasicek_a_zero() raises:
    # With a = 0 the discount factor is exp(sigma^2 t^3 / 6 - r t). The book's
    # listing has the division by 6 outside the exponential (see the docstring
    # of term_structure_discount_factor_vasicek); the corrected branch must be
    # the limit of the a != 0 branch.
    var d0 = term_structure_discount_factor_vasicek(2.0, 0.05, 0.0, 0.1, 0.1)
    assert_almost_equal(d0, 0.9169827055972, atol=1e-12)
    var d_plus = term_structure_discount_factor_vasicek(2.0, 0.05, 1e-3, 0.1, 0.1)
    var d_minus = term_structure_discount_factor_vasicek(2.0, 0.05, -1e-3, 0.1, 0.1)
    assert_almost_equal(d_plus, d0, atol=2e-4)
    assert_almost_equal(d_minus, d0, atol=2e-4)
    assert_almost_equal(0.5 * (d_plus + d_minus), d0, atol=1e-6)
    # a payment today is worth its amount
    assert_almost_equal(term_structure_discount_factor_vasicek(0.0, 0.05, 0.0, 0.1, 0.1), 1.0, atol=1e-15)


def price_two_year_bond[T: TermStructure](ts: T) -> Float64:
    var times: List[Float64] = [1.0, 2.0]
    var cashflows: List[Float64] = [10.0, 110.0]
    return bonds_price(times, cashflows, ts)


def test_models_are_term_structures() raises:
    # Every wrapper can be passed where a TermStructure is expected:
    # price = 10 d(1) + 110 d(2)
    var ns = TermStructureNelsonSiegel(0.01, 0.01, 0.01, 5.0)
    assert_almost_equal(price_two_year_bond(ns), 10.0 * ns.d(1.0) + 110.0 * ns.d(2.0), atol=1e-12)
    var sv = TermStructureSvensson(0.01, 0.01, 0.01, 0.02, 5.0, 2.0)
    assert_almost_equal(price_two_year_bond(sv), 10.0 * sv.d(1.0) + 110.0 * sv.d(2.0), atol=1e-12)
    var cs = TermStructureCubicSpline(0.1, 0.1, -0.1, [0.01, 0.01, -0.01], [2.0, 7.0, 12.0])
    assert_almost_equal(price_two_year_bond(cs), 10.0 * 1.1 + 110.0 * 0.8, atol=1e-10)
    var cir = TermStructureCir(0.05, 0.01, 0.0, 0.08, 0.1)
    assert_almost_equal(price_two_year_bond(cir), 10.0 * cir.d(1.0) + 110.0 * cir.d(2.0), atol=1e-12)
    var vc = TermStructureVasicek(0.05, -0.1, 0.1, 0.1)
    assert_almost_equal(price_two_year_bond(vc), 10.0 * vc.d(1.0) + 110.0 * vc.d(2.0), atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
