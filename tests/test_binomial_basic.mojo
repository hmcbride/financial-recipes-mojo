from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_equal
from finrecipes.binomial_basic import (
    option_price_call_european_binomial_single_period,
    option_price_put_european_binomial_single_period,
    binomial_tree,
    option_price_call_european_binomial_multi_period_given_ud,
    option_price_put_european_binomial_multi_period_given_ud,
)


def test_one_period_call() raises:
    # Chapter 8 example: one period european call = 3.64342
    var u: Float64 = 1.05
    var d = 1.0 / u
    assert_almost_equal(
        option_price_call_european_binomial_single_period(100.0, 100.0, 0.025, u, d), 3.64342, atol=1e-5
    )


def test_two_period_call() raises:
    # Chapter 8 example: two period european call = 5.44255
    var u: Float64 = 1.05
    var d = 1.0 / u
    assert_almost_equal(
        option_price_call_european_binomial_multi_period_given_ud(100.0, 100.0, 0.025, u, d, 2), 5.44255, atol=1e-5
    )


def test_multi_period_with_one_period_is_single_period() raises:
    var u: Float64 = 1.05
    var d = 1.0 / u
    assert_almost_equal(
        option_price_call_european_binomial_multi_period_given_ud(100.0, 100.0, 0.025, u, d, 1),
        option_price_call_european_binomial_single_period(100.0, 100.0, 0.025, u, d),
        atol=1e-12,
    )
    assert_almost_equal(
        option_price_put_european_binomial_multi_period_given_ud(100.0, 100.0, 0.025, u, d, 1),
        option_price_put_european_binomial_single_period(100.0, 100.0, 0.025, u, d),
        atol=1e-12,
    )


def test_put_call_parity() raises:
    # c - p = S - K exp(-r T), with r the interest rate per period
    var u: Float64 = 1.05
    var d = 1.0 / u
    var c1 = option_price_call_european_binomial_single_period(100.0, 100.0, 0.025, u, d)
    var p1 = option_price_put_european_binomial_single_period(100.0, 100.0, 0.025, u, d)
    assert_almost_equal(c1 - p1, 100.0 - 100.0 * exp(Float64(-0.025)), atol=1e-10)
    var c2 = option_price_call_european_binomial_multi_period_given_ud(100.0, 100.0, 0.025, u, d, 2)
    var p2 = option_price_put_european_binomial_multi_period_given_ud(100.0, 100.0, 0.025, u, d, 2)
    assert_almost_equal(c2 - p2, 100.0 - 100.0 * exp(Float64(-0.05)), atol=1e-10)
    var u5: Float64 = 1.1
    var d5 = 1.0 / u5
    var c5 = option_price_call_european_binomial_multi_period_given_ud(100.0, 95.0, 0.01, u5, d5, 5)
    var p5 = option_price_put_european_binomial_multi_period_given_ud(100.0, 95.0, 0.01, u5, d5, 5)
    assert_almost_equal(c5 - p5, 100.0 - 95.0 * exp(Float64(-0.05)), atol=1e-10)


def test_binomial_tree() raises:
    # C++ Code 8.2: row i has i + 1 prices, S0 u^j d^(i-j), lowest first
    var tree = binomial_tree(100.0, 1.1, 0.9, 3)
    assert_equal(len(tree), 3)
    assert_equal(len(tree[0]), 1)
    assert_equal(len(tree[1]), 2)
    assert_equal(len(tree[2]), 3)
    assert_almost_equal(tree[0][0], 100.0, atol=1e-10)
    assert_almost_equal(tree[1][0], 90.0, atol=1e-10)
    assert_almost_equal(tree[1][1], 110.0, atol=1e-10)
    assert_almost_equal(tree[2][0], 81.0, atol=1e-10)
    assert_almost_equal(tree[2][1], 99.0, atol=1e-10)
    assert_almost_equal(tree[2][2], 121.0, atol=1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
