from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.black_scholes import option_price_call_black_scholes
from finrecipes.warrants import warrant_price_adjusted_black_scholes


def test_warrant_price() raises:
    # C++ Code 10.1 example: warrant price = 10.142
    var w = warrant_price_adjusted_black_scholes(48.0, 40.0, 0.08, 0.30, 0.5, 1000.0, 10000.0)
    assert_almost_equal(w, 10.142, atol=1e-3)


def test_warrant_price_solves_the_warrant_equation() raises:
    # W = n/(n+m) C_BS(S + (m/n) W, K, r, sigma, T), to the accuracy EPSILON of the listing
    var m: Float64 = 1000.0
    var n: Float64 = 10000.0
    var w = warrant_price_adjusted_black_scholes(48.0, 40.0, 0.08, 0.30, 0.5, m, n)
    var rhs = (n / (n + m)) * option_price_call_black_scholes(48.0 + (m / n) * w, 40.0, 0.08, 0.30, 0.5)
    assert_almost_equal(w, rhs, atol=1e-5)


def test_warrant_dilution() raises:
    # a warrant is worth less than the corresponding call, and with no
    # warrants outstanding there is no dilution
    var c = option_price_call_black_scholes(48.0, 40.0, 0.08, 0.30, 0.5)
    var w = warrant_price_adjusted_black_scholes(48.0, 40.0, 0.08, 0.30, 0.5, 1000.0, 10000.0)
    assert_true(w < c)
    var w0 = warrant_price_adjusted_black_scholes(48.0, 40.0, 0.08, 0.30, 0.5, 0.0, 10000.0)
    assert_almost_equal(w0, c, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
