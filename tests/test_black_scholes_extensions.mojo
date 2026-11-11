from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.black_scholes import option_price_call_black_scholes, option_price_put_black_scholes
from finrecipes.black_scholes_extensions import (
    option_price_european_call_payout,
    option_price_european_put_payout,
    option_price_european_call_dividends,
    option_price_european_put_dividends,
    option_price_american_call_one_dividend,
    futures_option_price_call_european_black,
    futures_option_price_put_european_black,
    currency_option_price_call_european,
    currency_option_price_put_european,
    option_price_american_perpetual_call,
    option_price_american_perpetual_put,
)


def test_european_call_continuous_payout() raises:
    # C++ Code 11.1 example: european stock call option with continuous dividend = 11.7344
    assert_almost_equal(option_price_european_call_payout(100.0, 100.0, 0.1, 0.05, 0.25, 1.0), 11.7344, atol=1e-4)
    # no payout: the Black Scholes price
    assert_almost_equal(
        option_price_european_call_payout(100.0, 100.0, 0.1, 0.0, 0.25, 1.0),
        option_price_call_black_scholes(100.0, 100.0, 0.1, 0.25, 1.0),
        atol=1e-12,
    )


def test_european_put_continuous_payout() raises:
    # Exercise 11.1; put call parity: c - p = S exp(-q T) - X exp(-r T)
    var c = option_price_european_call_payout(100.0, 100.0, 0.1, 0.05, 0.25, 1.0)
    var p = option_price_european_put_payout(100.0, 100.0, 0.1, 0.05, 0.25, 1.0)
    assert_almost_equal(c - p, 100.0 * exp(Float64(-0.05)) - 100.0 * exp(Float64(-0.1)), atol=1e-6)
    assert_almost_equal(
        option_price_european_put_payout(100.0, 100.0, 0.1, 0.0, 0.25, 1.0),
        option_price_put_black_scholes(100.0, 100.0, 0.1, 0.25, 1.0),
        atol=1e-12,
    )


def test_european_call_discrete_dividends() raises:
    # C++ Code 11.2 example: european stock call option with discrete dividend = 11.8094
    var dividend_times: List[Float64] = [0.25, 0.75]
    var dividend_amounts: List[Float64] = [2.5, 2.5]
    assert_almost_equal(
        option_price_european_call_dividends(100.0, 100.0, 0.1, 0.25, 1.0, dividend_times, dividend_amounts),
        11.8094,
        atol=1e-4,
    )
    # dividends paid after maturity do not matter
    var late_times: List[Float64] = [1.25, 1.75]
    assert_almost_equal(
        option_price_european_call_dividends(100.0, 100.0, 0.1, 0.25, 1.0, late_times, dividend_amounts),
        option_price_call_black_scholes(100.0, 100.0, 0.1, 0.25, 1.0),
        atol=1e-12,
    )


def test_european_put_discrete_dividends() raises:
    # put call parity on the dividend adjusted stock price
    var dividend_times: List[Float64] = [0.25, 0.75]
    var dividend_amounts: List[Float64] = [2.5, 2.5]
    var c = option_price_european_call_dividends(100.0, 100.0, 0.1, 0.25, 1.0, dividend_times, dividend_amounts)
    var p = option_price_european_put_dividends(100.0, 100.0, 0.1, 0.25, 1.0, dividend_times, dividend_amounts)
    var adjusted_S = 100.0 - 2.5 * exp(Float64(-0.025)) - 2.5 * exp(Float64(-0.075))
    assert_almost_equal(c - p, adjusted_S - 100.0 * exp(Float64(-0.1)), atol=1e-6)


def test_american_call_one_dividend() raises:
    # C++ Code 11.3 example: american call price with one dividend = 10.0166
    assert_almost_equal(
        option_price_american_call_one_dividend(100.0, 100.0, 0.1, 0.25, 1.0, 10.0, 0.5), 10.0166, atol=1e-4
    )


def test_american_call_one_dividend_bounds() raises:
    # the American call is worth at least the European call on the dividend paying stock
    var dividend_times: List[Float64] = [0.5]
    var dividend_amounts: List[Float64] = [10.0]
    var european = option_price_european_call_dividends(
        100.0, 100.0, 0.1, 0.25, 1.0, dividend_times, dividend_amounts
    )
    var american = option_price_american_call_one_dividend(100.0, 100.0, 0.1, 0.25, 1.0, 10.0, 0.5)
    assert_true(american > european)
    # and at most the call on a stock that pays no dividend
    assert_true(american < option_price_call_black_scholes(100.0, 100.0, 0.1, 0.25, 1.0))
    # a small dividend, D1 <= K (1 - exp(-r (tau - tau1))): never exercised
    # early, so the price is the European price
    var small_amounts: List[Float64] = [1.0]
    assert_almost_equal(
        option_price_american_call_one_dividend(100.0, 100.0, 0.1, 0.25, 1.0, 1.0, 0.5),
        option_price_european_call_dividends(100.0, 100.0, 0.1, 0.25, 1.0, dividend_times, small_amounts),
        atol=1e-12,
    )


def test_futures_option_black() raises:
    # C++ Code 11.4 example. The program prints the label "european futures
    # call option" but calls futures_option_price_put_european_black, and
    # 0.851476 is the put price.
    assert_almost_equal(futures_option_price_put_european_black(50.0, 45.0, 0.08, 0.2, 0.5), 0.851476, atol=1e-6)
    # put call parity: c - p = exp(-r T) (F - K)
    var c = futures_option_price_call_european_black(50.0, 45.0, 0.08, 0.2, 0.5)
    var p = futures_option_price_put_european_black(50.0, 45.0, 0.08, 0.2, 0.5)
    assert_almost_equal(c - p, exp(Float64(-0.04)) * 5.0, atol=1e-6)
    assert_almost_equal(c, 0.851476 + exp(Float64(-0.04)) * 5.0, atol=1e-5)


def test_currency_option() raises:
    # C++ Code 11.5 example: european currency call option = 2.22556
    assert_almost_equal(currency_option_price_call_european(50.0, 52.0, 0.08, 0.05, 0.2, 0.5), 2.22556, atol=1e-5)
    # Exercise 11.4; put call parity: c - p = S exp(-r_f T) - X exp(-r T)
    var c = currency_option_price_call_european(50.0, 52.0, 0.08, 0.05, 0.2, 0.5)
    var p = currency_option_price_put_european(50.0, 52.0, 0.08, 0.05, 0.2, 0.5)
    assert_almost_equal(c - p, 50.0 * exp(Float64(-0.025)) - 52.0 * exp(Float64(-0.04)), atol=1e-6)
    # a currency option is an option on an asset with payout r_f
    assert_almost_equal(c, option_price_european_call_payout(50.0, 52.0, 0.08, 0.05, 0.2, 0.5), atol=1e-12)
    assert_almost_equal(p, option_price_european_put_payout(50.0, 52.0, 0.08, 0.05, 0.2, 0.5), atol=1e-12)


def test_perpetual_call() raises:
    # C++ Code 11.6 example: perpetual call price = 19.4767
    assert_almost_equal(option_price_american_perpetual_call(50.0, 40.0, 0.05, 0.02, 0.05), 19.4767, atol=1e-4)


def test_perpetual_put() raises:
    # Exercise 11.5. Without payouts (q = 0) h2 = -2r/sigma^2 and the formula
    # is Merton's perpetual put: K/(1+y) ((1+y)/y S/K)^(-y), y = 2r/sigma^2.
    # r = 0.08, sigma = 0.4 gives y = 1 and P = K^2 / (4 S).
    assert_almost_equal(option_price_american_perpetual_put(50.0, 40.0, 0.08, 0.0, 0.4), 8.0, atol=1e-10)
    assert_almost_equal(option_price_american_perpetual_put(100.0, 100.0, 0.08, 0.0, 0.4), 25.0, atol=1e-10)
    # at the optimal exercise price S* = K h2/(h2-1) the put is worth K - S*.
    # Here h2 = -1, so S* = K/2.
    assert_almost_equal(option_price_american_perpetual_put(20.0, 40.0, 0.08, 0.0, 0.4), 20.0, atol=1e-10)
    # with a payout the put is worth more than its exercise value above the exercise boundary
    var p = option_price_american_perpetual_put(50.0, 40.0, 0.05, 0.02, 0.2)
    assert_true(p > 0.0)
    assert_true(p < 40.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
