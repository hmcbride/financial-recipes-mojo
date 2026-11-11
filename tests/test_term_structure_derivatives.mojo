from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.term_structure_derivatives import bond_option_price_call_zero_vasicek
from finrecipes.term_structure_models import term_structure_discount_factor_vasicek


def test_vasicek_call_on_zero() raises:
    # Book example, C++ Code 26.1
    var a: Float64 = 0.1
    var b: Float64 = 0.1
    var sigma: Float64 = 0.02
    var r: Float64 = 0.05
    var X: Float64 = 0.9
    assert_almost_equal(bond_option_price_call_zero_vasicek(X, r, 1.0, 5.0, a, b, sigma), 0.000226833, atol=1e-9)


def test_deep_in_the_money_call() raises:
    # With a very low exercise price the call is worth P(t,s) - X P(t,T)
    var a: Float64 = 0.1
    var b: Float64 = 0.1
    var sigma: Float64 = 0.02
    var r: Float64 = 0.05
    var X: Float64 = 0.1
    var P_s = term_structure_discount_factor_vasicek(5.0, r, a, b, sigma)
    var P_T = term_structure_discount_factor_vasicek(1.0, r, a, b, sigma)
    assert_almost_equal(bond_option_price_call_zero_vasicek(X, r, 1.0, 5.0, a, b, sigma), P_s - X * P_T, atol=1e-12)


def test_a_zero_branch() raises:
    # The a == 0 branch is the limit of the general case
    var c0 = bond_option_price_call_zero_vasicek(0.8, 0.05, 1.0, 5.0, 0.0, 0.1, 0.02)
    var c_plus = bond_option_price_call_zero_vasicek(0.8, 0.05, 1.0, 5.0, 1e-3, 0.1, 0.02)
    var c_minus = bond_option_price_call_zero_vasicek(0.8, 0.05, 1.0, 5.0, -1e-3, 0.1, 0.02)
    assert_true(c0 > 0.0)
    assert_almost_equal(0.5 * (c_plus + c_minus), c0, atol=1e-5)


def test_call_price_falls_with_exercise_price() raises:
    var low = bond_option_price_call_zero_vasicek(0.7, 0.05, 1.0, 5.0, 0.1, 0.1, 0.02)
    var mid = bond_option_price_call_zero_vasicek(0.8, 0.05, 1.0, 5.0, 0.1, 0.1, 0.02)
    var high = bond_option_price_call_zero_vasicek(0.9, 0.05, 1.0, 5.0, 0.1, 0.1, 0.02)
    assert_true(low > mid)
    assert_true(mid > high)
    assert_true(high > 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
