from std.testing import TestSuite, assert_almost_equal
from finrecipes.term_structure import (
    term_structure_discount_factor_from_yield,
    term_structure_yield_from_discount_factor,
    term_structure_forward_rate_from_discount_factors,
    term_structure_forward_rate_from_yields,
    term_structure_yield_linearly_interpolated,
    TermStructureFlat,
    TermStructureInterpolated,
    bonds_price,
    bonds_duration,
    bonds_convexity,
)


def test_transformations() raises:
    # Book example, C++ Code 5.1
    var d_t1 = term_structure_discount_factor_from_yield(0.05, 1.0)
    assert_almost_equal(d_t1, 0.951229, atol=1e-6)
    var r_t2 = term_structure_yield_from_discount_factor(0.9, 2.0)
    assert_almost_equal(r_t2, 0.0526803, atol=1e-7)
    assert_almost_equal(term_structure_forward_rate_from_discount_factors(d_t1, 0.9, 1.0), 0.0553605, atol=1e-7)
    assert_almost_equal(term_structure_forward_rate_from_yields(0.05, r_t2, 1.0, 2.0), 0.0553605, atol=1e-7)


def test_flat_term_structure() raises:
    var ts = TermStructureFlat(0.05)
    assert_almost_equal(ts.d(1.0), 0.951229, atol=1e-6)
    assert_almost_equal(ts.d(2.0), 0.904837, atol=1e-6)
    assert_almost_equal(ts.r(1.0), 0.05, atol=1e-9)
    assert_almost_equal(ts.r(2.0), 0.05, atol=1e-9)
    assert_almost_equal(ts.f(1.0, 2.0), 0.05, atol=1e-9)
    ts.set_int_rate(0.1)
    assert_almost_equal(ts.r(3.0), 0.1, atol=1e-12)


def test_linear_interpolation() raises:
    var times: List[Float64] = [0.1, 0.5, 1.0, 5.0, 10.0]
    var yields: List[Float64] = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert_almost_equal(term_structure_yield_linearly_interpolated(0.1, times, yields), 0.1, atol=1e-9)
    assert_almost_equal(term_structure_yield_linearly_interpolated(0.5, times, yields), 0.2, atol=1e-9)
    assert_almost_equal(term_structure_yield_linearly_interpolated(1.0, times, yields), 0.3, atol=1e-9)
    assert_almost_equal(term_structure_yield_linearly_interpolated(3.0, times, yields), 0.35, atol=1e-9)
    assert_almost_equal(term_structure_yield_linearly_interpolated(5.0, times, yields), 0.4, atol=1e-9)
    assert_almost_equal(term_structure_yield_linearly_interpolated(10.0, times, yields), 0.5, atol=1e-9)
    assert_almost_equal(term_structure_yield_linearly_interpolated(20.0, times, yields), 0.5, atol=1e-9)


def test_interpolated_term_structure() raises:
    var ts = TermStructureInterpolated([0.1, 1.0, 5.0], [0.05, 0.07, 0.08])
    assert_almost_equal(ts.d(1.0), 0.932394, atol=1e-6)
    assert_almost_equal(ts.d(2.0), 0.865022, atol=1e-6)
    assert_almost_equal(ts.r(1.0), 0.07, atol=1e-9)
    assert_almost_equal(ts.r(2.0), 0.0725, atol=1e-9)
    assert_almost_equal(ts.f(1.0, 2.0), 0.075, atol=1e-9)


def test_bond_with_term_structure() raises:
    var times: List[Float64] = [1.0, 2.0]
    var cashflows: List[Float64] = [10.0, 110.0]
    var tsflat = TermStructureFlat(0.1)
    assert_almost_equal(bonds_price(times, cashflows, tsflat), 99.1088, atol=1e-4)
    assert_almost_equal(bonds_duration(times, cashflows, tsflat), 1.9087, atol=1e-4)
    assert_almost_equal(bonds_convexity(times, cashflows, tsflat), 3.72611, atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
