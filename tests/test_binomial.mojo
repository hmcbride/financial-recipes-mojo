from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.black_scholes import option_price_call_black_scholes, option_price_put_black_scholes
from finrecipes.binomial import (
    option_price_call_european_binomial,
    option_price_put_european_binomial,
    option_price_call_american_binomial,
    option_price_put_american_binomial,
    option_price_delta_american_call_binomial,
    option_price_partials_american_call_binomial,
    option_price_call_american_proportional_dividends_binomial,
    option_price_call_american_discrete_dividends_binomial,
    futures_option_price_call_american_binomial,
    currency_option_price_call_american_binomial,
)


def test_european_and_american_prices() raises:
    # Book example, section 12.2: S=100, K=100, r=0.1, sigma=0.25, 1 year, 100 steps
    assert_almost_equal(option_price_call_european_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100), 14.9505, atol=1e-4)
    assert_almost_equal(option_price_call_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100), 14.9505, atol=1e-4)
    assert_almost_equal(option_price_put_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100), 6.54691, atol=1e-5)


def test_matlab_example() raises:
    # Matlab output in the same example: C = 14.951, P = 6.5469
    assert_almost_equal(option_price_call_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100), 14.951, atol=1e-3)
    assert_almost_equal(option_price_put_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100), 6.5469, atol=1e-4)


def test_european_put() raises:
    # put-call parity holds exactly in the binomial tree for European options
    var c = option_price_call_european_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100)
    var p = option_price_put_european_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100)
    var pv_strike = option_price_call_black_scholes(100.0, 100.0, 0.1, 0.25, 1.0) - option_price_put_black_scholes(
        100.0, 100.0, 0.1, 0.25, 1.0
    )
    assert_almost_equal(c - p, pv_strike, atol=1e-9)
    # the American put is worth more than the European one
    assert_true(option_price_put_american_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100) > p)


def test_convergence_to_black_scholes() raises:
    # section 12.3: the binomial price moves towards the Black Scholes value
    var bs = option_price_call_black_scholes(100.0, 100.0, 0.1, 0.25, 1.0)
    assert_almost_equal(option_price_call_european_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 1000), bs, atol=1e-2)
    var bs_put = option_price_put_black_scholes(100.0, 100.0, 0.1, 0.25, 1.0)
    assert_almost_equal(option_price_put_european_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 1000), bs_put, atol=1e-2)


def test_partials() raises:
    # Book example, section 12.3.1
    var g = option_price_partials_american_call_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100)
    assert_almost_equal(g.delta, 0.699792, atol=1e-6)
    assert_almost_equal(g.gamma, 0.0140407, atol=1e-7)
    assert_almost_equal(g.theta, -9.89067, atol=1e-5)
    assert_almost_equal(g.vega, 34.8536, atol=1e-4)
    assert_almost_equal(g.rho, 56.9652, atol=1e-4)
    # C++ Code 12.3 computes the same delta on its own
    assert_almost_equal(
        option_price_delta_american_call_binomial(100.0, 100.0, 0.1, 0.25, 1.0, 100), 0.699792, atol=1e-6
    )


def test_payouts_and_dividends() raises:
    # Book example, section 12.5.3
    assert_almost_equal(option_price_call_american_binomial(100.0, 100.0, 0.10, 0.02, 0.25, 1.0, 100), 13.5926, atol=1e-4)
    var dividend_times: List[Float64] = [0.25, 0.75]
    var dividend_yields: List[Float64] = [0.025, 0.025]
    assert_almost_equal(
        option_price_call_american_proportional_dividends_binomial(
            100.0, 100.0, 0.10, 0.25, 1.0, 100, dividend_times, dividend_yields
        ),
        11.8604,
        atol=1e-4,
    )
    var dividend_amounts: List[Float64] = [2.5, 2.5]
    assert_almost_equal(
        option_price_call_american_discrete_dividends_binomial(
            100.0, 100.0, 0.10, 0.25, 1.0, 100, dividend_times, dividend_amounts
        ),
        12.0233,
        atol=1e-4,
    )


def test_no_dividends_falls_back() raises:
    var none = List[Float64]()
    var plain = option_price_call_american_binomial(100.0, 100.0, 0.10, 0.25, 1.0, 100)
    assert_almost_equal(
        option_price_call_american_proportional_dividends_binomial(100.0, 100.0, 0.10, 0.25, 1.0, 100, none, none),
        plain,
        atol=1e-12,
    )
    assert_almost_equal(
        option_price_call_american_discrete_dividends_binomial(100.0, 100.0, 0.10, 0.25, 1.0, 100, none, none),
        plain,
        atol=1e-12,
    )
    # a zero payout rate gives the plain price too
    assert_almost_equal(option_price_call_american_binomial(100.0, 100.0, 0.10, 0.0, 0.25, 1.0, 100), plain, atol=1e-12)


def test_futures_option() raises:
    # Book example, section 12.6
    assert_almost_equal(futures_option_price_call_american_binomial(50.0, 45.0, 0.08, 0.2, 0.5, 100), 5.74254, atol=1e-5)


def test_currency_option() raises:
    # Book example, section 12.7
    assert_almost_equal(
        currency_option_price_call_american_binomial(50.0, 52.0, 0.08, 0.05, 0.2, 0.5, 100), 2.23129, atol=1e-5
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
