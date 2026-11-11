from std.math import exp
from std.testing import TestSuite, assert_almost_equal
from finrecipes.black_scholes import (
    option_price_call_black_scholes,
    option_price_put_black_scholes,
    option_price_delta_call_black_scholes,
    option_price_delta_put_black_scholes,
    option_price_partials_call_black_scholes,
    option_price_partials_put_black_scholes,
    option_price_implied_volatility_call_black_scholes_bisections,
    option_price_implied_volatility_call_black_scholes_newton,
)


def test_call_price() raises:
    assert_almost_equal(option_price_call_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50), 5.45325, atol=1e-5)
    # Matlab example in the book
    assert_almost_equal(option_price_call_black_scholes(100.0, 100.0, 0.1, 0.1, 1.0), 10.308, atol=1e-3)


def test_put_call_parity() raises:
    var c = option_price_call_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50)
    var p = option_price_put_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50)
    assert_almost_equal(c - p, 50.0 - 50.0 * exp(Float64(-0.05)), atol=1e-6)


def test_partials() raises:
    var g = option_price_partials_call_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50)
    assert_almost_equal(g.delta, 0.633737, atol=1e-6)
    assert_almost_equal(g.gamma, 0.0354789, atol=1e-7)
    assert_almost_equal(g.theta, -6.61473, atol=1e-5)
    assert_almost_equal(g.vega, 13.3046, atol=1e-4)
    assert_almost_equal(g.rho, 13.1168, atol=1e-4)
    assert_almost_equal(option_price_delta_call_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50), 0.633737, atol=1e-6)
    var gp = option_price_partials_put_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50)
    assert_almost_equal(gp.delta, g.delta - 1.0, atol=1e-6)
    assert_almost_equal(option_price_delta_put_black_scholes(50.0, 50.0, 0.10, 0.30, 0.50), gp.delta, atol=1e-12)
    assert_almost_equal(gp.gamma, g.gamma, atol=1e-12)


def test_implied_volatility() raises:
    assert_almost_equal(
        option_price_implied_volatility_call_black_scholes_newton(50.0, 50.0, 0.10, 0.50, 2.5), 0.0500427, atol=1e-6
    )
    assert_almost_equal(
        option_price_implied_volatility_call_black_scholes_bisections(50.0, 50.0, 0.10, 0.50, 2.5), 0.0500419, atol=1e-6
    )
    # a price below the arbitrage bound gives 0
    assert_almost_equal(
        option_price_implied_volatility_call_black_scholes_newton(50.0, 40.0, 0.10, 0.50, 1.0), 0.0, atol=1e-12
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
