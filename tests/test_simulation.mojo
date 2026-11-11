from std.math import exp, log, sqrt
from std.testing import TestSuite, assert_almost_equal, assert_true
from finrecipes.black_scholes import (
    option_price_call_black_scholes,
    option_price_put_black_scholes,
    option_price_delta_call_black_scholes,
    option_price_delta_put_black_scholes,
)
from finrecipes.normdist import N
from finrecipes.random import random_seed
from finrecipes.simulation import (
    simulate_lognormal_random_variable,
    option_price_call_european_simulated,
    option_price_put_european_simulated,
    option_price_delta_call_european_simulated,
    option_price_delta_put_european_simulated,
    payoff_call,
    payoff_put,
    payoff_cash_or_nothing_call,
    payoff_asset_or_nothing_call,
    derivative_price_simulate_european_option_generic,
    derivative_price_simulate_european_option_generic_with_control_variate,
    derivative_price_simulate_european_option_generic_with_antithetic_variate,
)

# All examples in chapter 14 use S=100, K=100, r=0.1, sigma=0.25, time=1.
#
# The book's simulated numbers come from the C library rand() and cannot be
# reproduced digit for digit. Each Monte Carlo estimate is instead compared
# with its analytical value, with tolerance 4 standard errors, where
# standard error = (standard deviation of one discounted payoff) / sqrt(no_sims).
# The standard deviations quoted in the comments were computed by numerical
# integration over the normal density for these parameters. The generator is
# seeded, so the runs are reproducible.

comptime S: Float64 = 100.0
comptime K: Float64 = 100.0
comptime r: Float64 = 0.1
comptime sigma: Float64 = 0.25
comptime time: Float64 = 1.0


def four_standard_errors(sd: Float64, no_sims: Int) -> Float64:
    return 4.0 * sd / sqrt(Float64(no_sims))


def test_black_scholes_values_of_examples() raises:
    # the deterministic lines of the "Output from C++ program" boxes
    assert_almost_equal(option_price_call_black_scholes(S, K, r, sigma, time), 14.9758, atol=1e-4)
    assert_almost_equal(option_price_put_black_scholes(S, K, r, sigma, time), 5.45954, atol=1e-5)
    assert_almost_equal(option_price_delta_call_black_scholes(S, K, r, sigma, time), 0.700208, atol=1e-6)
    assert_almost_equal(option_price_delta_put_black_scholes(S, K, r, sigma, time), -0.299792, atol=1e-6)


def test_simulate_lognormal_random_variable() raises:
    # C++ Code 14.1. S_T has mean S exp(r T) = 110.517 and standard deviation
    # 28.067; log(S_T) has mean log(S) + (r - sigma^2/2) T and standard deviation 0.25.
    random_seed(141)
    var no_sims = 200000
    var sum: Float64 = 0.0
    var sum_log: Float64 = 0.0
    for _ in range(no_sims):
        var S_T = simulate_lognormal_random_variable(S, r, sigma, time)
        assert_true(S_T > 0.0)
        sum += S_T
        sum_log += log(S_T)
    assert_almost_equal(sum / Float64(no_sims), S * exp(r * time), atol=four_standard_errors(28.067, no_sims))
    assert_almost_equal(
        sum_log / Float64(no_sims),
        log(S) + (r - 0.5 * sigma * sigma) * time,
        atol=four_standard_errors(0.25, no_sims),
    )


def test_option_price_european_simulated_book_example() raises:
    # C++ Code 14.2 with the book's 5000 simulations. Book output: call
    # 14.8404 (Black Scholes 14.9758), put 5.74588 (Black Scholes 5.45954).
    # One discounted call payoff has standard deviation 19.92, a put payoff
    # 9.199, so 4 standard errors are 1.13 and 0.52.
    random_seed(142)
    var no_sims = 5000
    var c = option_price_call_european_simulated(S, K, r, sigma, time, no_sims)
    var p = option_price_put_european_simulated(S, K, r, sigma, time, no_sims)
    assert_almost_equal(c, 14.9758, atol=four_standard_errors(19.92, no_sims))
    assert_almost_equal(p, 5.45954, atol=four_standard_errors(9.199, no_sims))


def test_option_price_european_simulated_converges() raises:
    # the same routines with 1,000,000 simulations: 4 standard errors are
    # 0.080 for the call and 0.037 for the put
    random_seed(1420)
    var no_sims = 1000000
    var c = option_price_call_european_simulated(S, K, r, sigma, time, no_sims)
    var p = option_price_put_european_simulated(S, K, r, sigma, time, no_sims)
    assert_almost_equal(c, 14.9758, atol=four_standard_errors(19.92, no_sims))
    assert_almost_equal(p, 5.45954, atol=four_standard_errors(9.199, no_sims))


def test_option_price_delta_european_simulated() raises:
    # C++ Code 14.3. Book output with 5000 simulations: call delta 0.701484
    # (Black Scholes 0.700208), put delta -0.307211 (Black Scholes -0.299792).
    # The estimator is the finite difference (c(S+q) - c(S)) / q with q = 0.01 S,
    # whose expectation is the same finite difference of the Black Scholes
    # price (0.70709 for the call, -0.29291 for the put). One term of the
    # estimate has standard deviation 0.579 (call) and 0.373 (put).
    var q = S * 0.01
    var fd_call = (option_price_call_black_scholes(S + q, K, r, sigma, time)
        - option_price_call_black_scholes(S, K, r, sigma, time)) / q
    var fd_put = (option_price_put_black_scholes(S + q, K, r, sigma, time)
        - option_price_put_black_scholes(S, K, r, sigma, time)) / q
    random_seed(143)
    var no_sims = 200000
    var delta_call = option_price_delta_call_european_simulated(S, K, r, sigma, time, no_sims)
    var delta_put = option_price_delta_put_european_simulated(S, K, r, sigma, time, no_sims)
    assert_almost_equal(delta_call, fd_call, atol=four_standard_errors(0.579, no_sims))
    assert_almost_equal(delta_put, fd_put, atol=four_standard_errors(0.373, no_sims))
    # against the analytical deltas the tolerance also has to cover the
    # finite difference bias, gamma q / 2 = 0.007
    assert_almost_equal(delta_call, 0.700208, atol=0.007 + four_standard_errors(0.579, no_sims))
    assert_almost_equal(delta_put, -0.299792, atol=0.007 + four_standard_errors(0.373, no_sims))
    # the book's sample size
    random_seed(1430)
    no_sims = 5000
    delta_call = option_price_delta_call_european_simulated(S, K, r, sigma, time, no_sims)
    delta_put = option_price_delta_put_european_simulated(S, K, r, sigma, time, no_sims)
    assert_almost_equal(delta_call, fd_call, atol=four_standard_errors(0.579, no_sims))
    assert_almost_equal(delta_put, fd_put, atol=four_standard_errors(0.373, no_sims))


def test_payoff_functions() raises:
    # C++ Code 14.4 and 14.8
    assert_almost_equal(payoff_call(110.0, 100.0), 10.0, atol=1e-12)
    assert_almost_equal(payoff_call(90.0, 100.0), 0.0, atol=1e-12)
    assert_almost_equal(payoff_put(110.0, 100.0), 0.0, atol=1e-12)
    assert_almost_equal(payoff_put(90.0, 100.0), 10.0, atol=1e-12)
    assert_almost_equal(payoff_cash_or_nothing_call(110.0, 100.0), 1.0, atol=1e-12)
    assert_almost_equal(payoff_cash_or_nothing_call(100.0, 100.0), 1.0, atol=1e-12)
    assert_almost_equal(payoff_cash_or_nothing_call(90.0, 100.0), 0.0, atol=1e-12)
    assert_almost_equal(payoff_asset_or_nothing_call(110.0, 100.0), 110.0, atol=1e-12)
    assert_almost_equal(payoff_asset_or_nothing_call(100.0, 100.0), 100.0, atol=1e-12)
    assert_almost_equal(payoff_asset_or_nothing_call(90.0, 100.0), 0.0, atol=1e-12)


def test_generic_simulation() raises:
    # C++ Code 14.5 with the book's 50000 simulations. Book output: call
    # 14.995 (Black Scholes 14.9758), put 5.5599 (Black Scholes 5.45954).
    # 4 standard errors: 0.356 for the call, 0.165 for the put.
    random_seed(145)
    var no_sims = 50000
    var c = derivative_price_simulate_european_option_generic(S, K, r, sigma, time, payoff_call, no_sims)
    var p = derivative_price_simulate_european_option_generic(S, K, r, sigma, time, payoff_put, no_sims)
    assert_almost_equal(c, 14.9758, atol=four_standard_errors(19.92, no_sims))
    assert_almost_equal(p, 5.45954, atol=four_standard_errors(9.199, no_sims))


def test_generic_simulation_with_control_variate() raises:
    # C++ Code 14.6 with 50000 simulations. The control variate is the at the
    # money call, so for the at the money call itself the simulated payoffs
    # cancel and the Black Scholes price is returned whatever the random
    # numbers: the book prints "Simulated call option price, CV = 14.9758".
    random_seed(146)
    var no_sims = 50000
    var c = derivative_price_simulate_european_option_generic_with_control_variate(
        S, K, r, sigma, time, payoff_call, no_sims
    )
    assert_almost_equal(c, 14.9758, atol=1e-4)
    assert_almost_equal(c, option_price_call_black_scholes(S, K, r, sigma, time), atol=1e-9)
    # Put: book output 5.42541 (Black Scholes 5.45954). The adjusted payoff,
    # put minus call, has standard deviation 25.40, so 4 standard errors
    # are 0.454.
    var p = derivative_price_simulate_european_option_generic_with_control_variate(
        S, K, r, sigma, time, payoff_put, no_sims
    )
    assert_almost_equal(p, 5.45954, atol=four_standard_errors(25.40, no_sims))


def test_generic_simulation_with_antithetic_variate() raises:
    # C++ Code 14.7 with 50000 pairs. Book output: call 14.9919 (Black
    # Scholes 14.9758), put 5.46043 (Black Scholes 5.45954). The average of
    # an antithetic pair has standard deviation 9.449 (call) and 5.235 (put),
    # so 4 standard errors are 0.169 and 0.094.
    random_seed(147)
    var no_sims = 50000
    var c = derivative_price_simulate_european_option_generic_with_antithetic_variate(
        S, K, r, sigma, time, payoff_call, no_sims
    )
    var p = derivative_price_simulate_european_option_generic_with_antithetic_variate(
        S, K, r, sigma, time, payoff_put, no_sims
    )
    assert_almost_equal(c, 14.9758, atol=four_standard_errors(9.449, no_sims))
    assert_almost_equal(p, 5.45954, atol=four_standard_errors(5.235, no_sims))


def test_binary_options() raises:
    # Example of section 14.6 with the book's 5000 simulations.
    # Analytical values: cash or nothing call exp(-r T) N(d2) = 0.550442,
    # asset or nothing call S N(d1) = 70.0200.
    var time_sqrt = sqrt(time)
    var d1 = (log(S / K) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt
    var d2 = d1 - sigma * time_sqrt
    var cash = exp(-r * time) * N(d2)
    var asset = S * N(d1)
    assert_almost_equal(cash, 0.550442, atol=1e-5)
    assert_almost_equal(asset, 70.0200, atol=1e-3)
    random_seed(148)
    var no_sims = 5000
    # Cash or nothing. Book output: 0.547427, control variate 1.02552,
    # antithetic variate 0.549598. Standard deviation of one term: 0.4417
    # plain, 19.66 with the control variate (the at the money call is far
    # noisier than the binary payoff, which is why the book's control variate
    # estimate is so poor), 0.1864 for an antithetic pair.
    var v = derivative_price_simulate_european_option_generic(
        S, K, r, sigma, time, payoff_cash_or_nothing_call, no_sims
    )
    assert_almost_equal(v, cash, atol=four_standard_errors(0.4417, no_sims))
    v = derivative_price_simulate_european_option_generic_with_control_variate(
        S, K, r, sigma, time, payoff_cash_or_nothing_call, no_sims
    )
    assert_almost_equal(v, cash, atol=four_standard_errors(19.66, no_sims))
    v = derivative_price_simulate_european_option_generic_with_antithetic_variate(
        S, K, r, sigma, time, payoff_cash_or_nothing_call, no_sims
    )
    assert_almost_equal(v, cash, atol=four_standard_errors(0.1864, no_sims))
    # Asset or nothing. Book output: 70.5292, control variate 69.8451,
    # antithetic variate 70.2205. Standard deviation of one term: 58.39
    # plain, 44.17 with the control variate, 16.46 for an antithetic pair.
    v = derivative_price_simulate_european_option_generic(
        S, K, r, sigma, time, payoff_asset_or_nothing_call, no_sims
    )
    assert_almost_equal(v, asset, atol=four_standard_errors(58.39, no_sims))
    v = derivative_price_simulate_european_option_generic_with_control_variate(
        S, K, r, sigma, time, payoff_asset_or_nothing_call, no_sims
    )
    assert_almost_equal(v, asset, atol=four_standard_errors(44.17, no_sims))
    v = derivative_price_simulate_european_option_generic_with_antithetic_variate(
        S, K, r, sigma, time, payoff_asset_or_nothing_call, no_sims
    )
    assert_almost_equal(v, asset, atol=four_standard_errors(16.46, no_sims))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
